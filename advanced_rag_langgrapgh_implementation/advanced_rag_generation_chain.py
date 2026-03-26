from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from advanced_rag_state import MessageGraph
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-5.2", temperature=0, max_tokens=500)

# Basic RAG prompt (replacement for LangChain Hub prompt)
llm_prompt = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are a helpful assistant. Use the provided context to answer the question. "
			"If the answer is not in the context, say you don't know.",
		),
		(
			"human",
			"Context:\n\n{context}\n\nQuestion: {question}",
		),
	]
)

generation_chain = llm_prompt | llm | StrOutputParser()