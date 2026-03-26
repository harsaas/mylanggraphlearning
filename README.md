**Agentic RAG with LangGraph **
Implementation of Adaptive RAG tailored towards developers and production-oriented applications for learning LangGraph



Langsmith Trace:
https://smith.langchain.com/public/555c8bd7-8fe1-4375-806e-ba08f71f5dca/r

Graph with Agentic RAG :

<img width="394" height="495" alt="image" src="https://github.com/user-attachments/assets/42cbb4e6-d681-4ccb-bfdd-5506b88dfb62" />

	Files	Description	Key Components
			
1	-> Setting up the project foundation	Initialize project structure; Configure Poetry for dependency management

2	-> advanced_rag_ingestion.py	Setting up the vector database	Create ingestion pipeline; Implement vector store with Chroma and OpenAI embeddings

3	-> advanced_rag_state.py	Defining the state management	Create GraphState class; Set up typed dictionaries for state tracking

4	-> advanced_rag_retrieve.py	Implementing the document retrieval	Build retrieve node; Connect retrieval to vector database

5	-> advanced_rag_grader_docs.py	Evaluating document relevance	Create document grading functionality; Implement decision logic for document relevance

6	-> advanced_rag_websearch.py	Adding external search capability	Integrate Tavily search API; Implement fallback for insufficient local knowledge

7	-> advanced_rag_generate.py ,advanced_rag_generation_chain.py	Creating the answer generation component	Build generate node; Implement context-aware response generation

8	-> advanced_rag_graph.py	Constructing the complete LangGraph workflow	Connect all nodes into workflow; Implement conditional edges for adaptive behavior

9	-> advanced_rag_answer_grader.py , advanced_rag_hallucination_grader.py , advanced_rag_retrievegrader.py	Adding self-evaluation capabilities	Implement hallucination detection; Create feedback loops for answer improvement

10 -> advanced_rag_router_prompt.py	Smart query routing	Create intelligent routing between retrieval and web search; Optimize entry point for different query types

11 -> Final code formatting and cleanup	Code optimization; Final documentation improvements

