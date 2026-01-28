from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from uipath_langchain.retrievers import ContextGroundingRetriever
from langchain_core.documents import Document
from uipath.platform.common import InvokeProcess
from uipath.platform.errors import IngestionInProgressException
import httpx
from uipath_langchain.chat.models import UiPathAzureChatOpenAI
from langchain_openai import ChatOpenAI
import logging
import time

from dotenv import load_dotenv
load_dotenv()


class IndexNotFound(Exception):
    pass

logger = logging.getLogger(__name__)

# Define system prompt for company policy agent
system_prompt = """
You are an advanced AI assistant specializing in answering questions about internal company policies. 
Structure your responses as follows:

Answer:
<1–3 sentence conclusion>

Policy evidence:
- "<exact quote>"
  Reference: <section/bullet>

Conditions/Exceptions:
- <condition or threshold>

Next steps:
- <action 1>
- <action 2>

If not covered:
<one line stating it’s not covered and what info is missing>
"""

# llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
# llm = UiPathAzureChatOpenAI()
llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    max_tokens=4000,
    timeout=30,
    max_retries=2,
)


async def get_context_data_async(retriever: ContextGroundingRetriever, question: str) -> list[Document]:
    no_of_retries = 5
    context_data = None
    data_queried = False
    while no_of_retries != 0:
        try:
            context_data = await retriever.ainvoke(question)
            data_queried = True
            break
        except IngestionInProgressException as ex:
            logger.info(ex.message)
            no_of_retries -= 1
            logger.info(f"{no_of_retries} retries left")
            time.sleep(5)
        except httpx.HTTPStatusError as err:
            if err.response.status_code == 404:
                raise IndexNotFound
            raise
    if not data_queried:
        raise Exception("Ingestion is taking too long.")
    return  context_data


class GraphState(BaseModel):
    """State for the company policy agent graph."""
    question: str
    category: str | None = None
    # Human-in-the-loop fields (used by procurement flow)
    email: str | None = None
    code: str | None = None


class GraphOutput(BaseModel):
    """Output for the company policy agent graph."""
    response: str


def is_valid_company_email(email: str | None) -> bool:
    """Validate that the email is in the uipath.com domain."""
    if not email or "@" not in email:
        return False
    return email.strip().lower().endswith("@uipath.com")


def is_valid_code(code: str | None) -> bool:
    """Validate a 4-digit code that is not all the same digit and not consecutive (e.g., 1234)."""
    if not code or len(code) != 4 or not code.isdigit():
        return False
    # Reject all identical digits (e.g., 1111)
    if len(set(code)) == 1:
        return False
    # Reject strictly ascending consecutive sequences (e.g., 0123, 1234, ..., 6789)
    digits = [int(c) for c in code]
    if all(digits[i + 1] - digits[i] == 1 for i in range(3)):
        return False
    return True


async def policy_node(state: GraphState) -> GraphOutput:
    retriever = ContextGroundingRetriever(
                index_name="company-policy-index",
                folder_path="WellsFargoCodedAgents",
                number_of_results=100,
            )
    try:
        context_docs = await get_context_data_async(retriever, state.question)
    except IndexNotFound:
        context_docs = []
    

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.question},
    ]
    if state.category:
        messages.append({"role": "user", "content": f"Category: {state.category}"})
    if context_docs:
        messages.append({"role":"user", "content": context_docs[0].page_content})

    """Process a company policy question and return a structured answer."""
    result = await llm.ainvoke(messages)
    return GraphOutput(response=result.content)


async def supervisor_node(state: GraphState) -> GraphState:
    """Supervisor: classify the question and store the category for routing."""
    categories = [
        "Company Policy",
        "Procurement",
    ]
    messages = [
        {"role": "system", "content": (
            "Classify the user's question into one of: Company Policy, Procurement. "
            "Reply with only the exact label."
        )},
        {"role": "user", "content": state.question},
    ]
    result = await llm.ainvoke(messages)
    label = (result.content or "").strip()
    normalized = {c.lower(): c for c in categories}
    category = normalized.get(label.lower(), "Company Policy")
    return GraphState(question=state.question, category=category)


def route_by_category(state: GraphState) -> str:
    """Return the route key to dispatch to the specialized agent."""
    if state.category and state.category.lower() == "procurement":
        return "procurement"
    return "policy"




async def procurement_node(state: GraphState) -> GraphOutput:
    """Procurement specialized agent (assumes prior verification)."""
    email = state.email
    code = state.code

    # Defensive guard: if credentials are missing/invalid, exit immediately.
    if not is_valid_company_email(email) or not is_valid_code(code):
        return GraphOutput(response="Failed: E-mail and CODE are required for procurement.")

    retriever = ContextGroundingRetriever(
                index_name="procurement-index",
                folder_path="WellsFargoCodedAgents",
                number_of_results=100,
            )
    try:
        context_docs = await get_context_data_async(retriever, state.question)
    except IndexNotFound:
        context_docs = []

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.question},
    ]
    messages.append({"role": "user", "content": "Agent: Procurement"})
    if state.category:
        messages.append({"role": "user", "content": f"Category: {state.category}"})
    # Attach verified email/code context (not used as secrets; only provenance)
    if email:
        messages.append({"role": "user", "content": f"Verified email: {email}"})
    if code:
        messages.append({"role": "user", "content": "Verification code provided (stored transiently)"})
    if context_docs:
        messages.append({"role":"user", "content": context_docs[0].page_content})

    result = await llm.ainvoke(messages)
    return GraphOutput(response=result.content)


async def verify_credentials_node(state: GraphState):
    """Verification gate: validate optional email/code for procurement path.

    - If missing: fail fast with concise message
    - If invalid: fail fast with concise message
    - If valid: pass state through unchanged
    """
    email = state.email
    code = state.code
    if email is None or code is None:
        return GraphOutput(response="Failed: E-mail and CODE are required for procurement.")
    if not is_valid_company_email(email) or not is_valid_code(code):
        return GraphOutput(response="Failed: E-mail and CODE are invalid.")
    return state


def route_after_verification(state) -> str:
    """Route based on verification result.

    - If verification failed, prior node returned GraphOutput → end
    - If verification passed, prior node returned GraphState → procurement
    """
    # LangGraph passes the state as a dict here. If verification failed,
    # the node returned GraphOutput, which surfaces as {'response': ...}.
    if isinstance(state, GraphOutput):
        return "end"
    if isinstance(state, dict) and "response" in state:
        return "end"
    # If we received a GraphState but credentials are invalid, end as well
    if isinstance(state, GraphState):
        if not is_valid_company_email(state.email) or not is_valid_code(state.code):
            return "end"
    return "procurement"

# Build the state graph with conditional routing
builder = StateGraph(GraphState, output=GraphOutput)
builder.add_node("supervisor", supervisor_node)
builder.add_node("policy", policy_node)
builder.add_node("verify_credentials", verify_credentials_node)
builder.add_node("procurement", procurement_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_by_category,
    {
        "policy": "policy",
        "procurement": "verify_credentials",
    },
)
builder.add_edge("policy", END)
builder.add_conditional_edges(
    "verify_credentials",
    route_after_verification,
    {
        "procurement": "procurement",
        "end": END,
    },
)
builder.add_edge("procurement", END)

# Compile the graph
graph = builder.compile()


