from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ---- LangSmith tracing (uses your .env key) ----
if not os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "langgraph_reflection_agent")

#Message placeholder is used to indicate where the messages from the previous steps in the chain should be inserted into the prompt. This allows the agent to have access to the conversation history and use it to generate more informed critiques and recommendations for the user's tweet.

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a viral twitter influencer generating a tweet based on the critique and recommendations provided. Always follow the recommendations provided to generate a viral tweet."
                       " Generate the best twitter post possible for the user's request."
                       "If the user provides critique, respond with a revised version of your previous attempts."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="gpt-5.2", temperature=0.7, max_tokens=500)

generativechain = generation_prompt | llm
reflectionchain = reflection_prompt | llm

