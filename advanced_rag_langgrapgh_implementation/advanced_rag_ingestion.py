from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
# get the docs from the sublist and create a single list of docs
docs_list = [doc for sublist in docs for doc in sublist]

#split the content of the docs into smaller chunks to be ingested into the vector database. This is done to ensure that the content is not too large for the model to process and to improve retrieval performance.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs_list)

#Initialize the embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents= split_docs, 
    collection_name ='rag-chroma-collection',
    embedding=embeddings,
    persist_directory="./chroma_db"
    )

#vectorstore.add_documents(split_docs)

retriever = Chroma(
    collection_name="rag-chroma-collection",
    persist_directory="./chroma_db",
    #embedding_function=OpenAIEmbeddings(),
    embedding_function=embeddings,
).as_retriever()

print("collections:", vectorstore._collection.name)
print("count:", vectorstore._collection.count())

