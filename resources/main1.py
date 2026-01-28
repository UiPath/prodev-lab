from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from uipath_langchain.retrievers import ContextGroundingRetriever
from langchain_core.documents import Document
from uipath.platform.common import InvokeProcess
from uipath.platform.errors import IngestionInProgressException
import httpx
from uipath_langchain.chat.models import UiPathAzureChatOpenAI
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


class IndexNotFound(Exception):
    pass


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

# policy_agent = create_react_agent(llm, tools=[], prompt=system_prompt)


class GraphInput(BaseModel):
    """Input for the company policy agent graph."""
    question: str


class GraphOutput(BaseModel):
    """Output for the company policy agent graph."""
    response: str

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

async def policy_node(state: GraphInput) -> GraphOutput:
    retriever = ContextGroundingRetriever(
                index_name="company-policy-index",
                folder_path="WellsFargoCodedAgents",
                number_of_results=100,
            )
    try:
        context_data= await get_context_data_async(retriever, state.question)
    except IndexNotFound:
        context_data = system_prompt
    

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.question},
        {"role":"user", "content": context_data[0].page_content}
    ]

    """Process a company policy question and return a structured answer."""
    user_message = f"""Please answer the following company policy question. Use clear section headers for each relevant policy area (Procurement Policy, Equipment Acquisition, PTO Policy).If the question is outside these topics, politely state your scope.\n\nQuestion: {state.question}"""
    # result = await policy_agent.ainvoke(new_state)
    result = await llm.ainvoke(messages)
    return GraphOutput(response=result.content)


# Build the state graph
builder = StateGraph(GraphInput, output=GraphOutput)
builder.add_node("policy", policy_node)

builder.add_edge(START, "policy")
builder.add_edge("policy", END)

# Compile the graph
graph = builder.compile()
