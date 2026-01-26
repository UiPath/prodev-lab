from langchain_anthropic import ChatAnthropic
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
 
import os
import json

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
                folder_path="Shared",
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
        "HR",
    ]
    messages = [
        {"role": "system", "content": (
            "Classify the user's question into one of: Company Policy, Procurement, HR. "
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
    if state.category and state.category.lower() == "hr":
        return "hr"
    return "policy"




async def procurement_node(state: GraphState) -> GraphOutput:
    """Procurement specialized agent (assumes prior verification)."""
    email = state.email
    code = state.code

    retriever = ContextGroundingRetriever(
                index_name="procurement-index",
                folder_path="Shared",
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


async def hr_node(state: GraphState) -> GraphOutput:
    """HR specialized agent (no context grounding).

    Chooses an HR system/tool (e.g., Workday) based on the question using:
    - A local JSON catalog file if present (./hr_tools.json or ./resources/hr_tools.json)
    - Otherwise a built-in default mapping
    The model is prompted to return a compact JSON plan with the chosen tool and action.
    """

    # Load optional local catalog
    default_catalog = {
        "workday": {
            "areas": [
                "pto", "leave", "vacation", "time off", "absence",
                "benefits", "payroll", "compensation", "recruiting", "onboarding"
            ]
        },
        "servicenow_hrsd": {
            "areas": ["hr ticket", "policy request", "case", "issue", "access request"]
        },
        "bamboohr": {
            "areas": ["employee directory", "time off", "basic hr records"]
        },
        "greenhouse": {
            "areas": ["recruiting", "interview", "schedule", "offer"]
        }
    }

    catalog = default_catalog
    for path in [
        os.path.join(os.getcwd(), "hr_tools.json"),
        os.path.join(os.getcwd(), "resources", "hr_tools.json"),
    ]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    catalog = json.load(f)
                break
            except Exception:
                catalog = default_catalog

    planning_prompt = (
        "You are an HR routing module. Choose the best target system/tool from the provided catalog "
        "for the user's request and propose a single concrete action. Respond with ONLY valid JSON "
        "in the following schema: {\n"
        "  \"tool\": <one of the catalog keys>,\n"
        "  \"action\": <short command-like phrase>,\n"
        "  \"parameters\": {<key-value minimal fields>},\n"
        "  \"reason\": <one sentence rationale>\n"
        "}. If out of scope, set tool to \"none\" and explain in reason."
    )

    messages = [
        {"role": "system", "content": planning_prompt},
        {"role": "user", "content": f"Catalog: {json.dumps(catalog)}"},
        {"role": "user", "content": f"Question: {state.question}"},
    ]

    result = await llm.ainvoke(messages)
    content = result.content or "{}"

    plan = None
    try:
        plan = json.loads(content)
    except Exception:
        # Try to recover from responses wrapped in code fences
        if "{" in content and "}" in content:
            try:
                start = content.find("{")
                end = content.rfind("}") + 1
                plan = json.loads(content[start:end])
            except Exception:
                plan = None

    if not isinstance(plan, dict):
        return GraphOutput(response="Tool selected for the ask is: none")

    tool = plan.get("tool", "none")
    return GraphOutput(response=f"Tool selected for the ask is: {tool}")


async def permission_check_local_DB(state: GraphState):
    """HR permission gate using a local JSON DB of allowed emails.

    File lookup order: ./hr_auth.json, ./resources/hr_auth.json
    Expected format: {"allowed_emails": ["user@company.com", ...]}

    - If email is missing or not in the allow-list: return GraphOutput("Not authorised")
    - If allowed: pass state through unchanged
    """
    email = (state.email or "").strip().lower()
    allowlist: list[str] = []

    for path in [
        os.path.join(os.getcwd(), "hr_auth.json"),
        os.path.join(os.getcwd(), "resources", "hr_auth.json"),
    ]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    allowlist = [e.strip().lower() for e in data.get("allowed_emails", []) if isinstance(e, str)]
                break
            except Exception:
                allowlist = []

    if not email or email not in allowlist:
        return GraphOutput(response="Not authorised")

    return state

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


# Conditional routing after verification to respect LangGraph pattern
def route_after_verification(state) -> str:
    """Route based on verification result.

    - If verification failed, prior node returned GraphOutput → end
    - If verification passed, prior node returned GraphState → procurement
    """
    if isinstance(state, GraphOutput):
        return "end"
    if isinstance(state, dict) and "response" in state:
        return "end"
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
builder.add_node("hr", hr_node)
builder.add_node("permission_check_local_DB", permission_check_local_DB)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_by_category,
    {
        "policy": "policy",
        "procurement": "verify_credentials",
        "hr": "permission_check_local_DB",
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
builder.add_edge("permission_check_local_DB", "hr")
builder.add_edge("hr", END)

# Compile the graph
graph = builder.compile()


