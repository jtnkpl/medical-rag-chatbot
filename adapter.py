import os
from typing import List
from dotenv import load_dotenv, find_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

from app.core.interfaces import RAGPipeline

# Load environment variables
load_dotenv(find_dotenv())

# Smart path resolution: detects if run from repository root or chatbot subfolder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(CURRENT_DIR, "medical-chatbot")):
    # Run from root folder
    DB_FAISS_PATH = os.path.join(CURRENT_DIR, "medical-chatbot", "vectorstore", "db_faiss")
else:
    # Run from subfolder
    DB_FAISS_PATH = os.path.join(CURRENT_DIR, "vectorstore", "db_faiss")


class MedicalRAGAdapter(RAGPipeline):
    """
    Adapter bridging the medical-rag-chatbot codebase to the evaluation system.
    Implements the standard RAGPipeline interface.
    """

    def __init__(self):
        # 1. Initialize the LLM (using Groq as configured in the chatbot)
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            # Fallback to general API keys if GROQ_API_KEY is not set
            groq_api_key = os.environ.get("FREE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            
        groq_model_name = "llama-3.1-8b-instant"
        self.llm = ChatGroq(
            model=groq_model_name,
            temperature=0.5,
            max_tokens=512,
            api_key=groq_api_key,
        )

        # 2. Load the FAISS vector store
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        # 3. Setup the document stuffing chain
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        self.combine_docs_chain = create_stuff_documents_chain(
            self.llm,
            retrieval_qa_chat_prompt
        )

    def retrieve(self, query: str) -> List[str]:
        """
        Retrieves relevant document chunks from the FAISS database.
        """
        retriever = self.db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        return [doc.page_content for doc in docs]

    def generate(self, query: str, docs: List[str]) -> str:
        """
        Generates an answer using the combined documents and user query.
        """
        # Convert raw string docs back to LangChain Document objects
        langchain_docs = [Document(page_content=doc_content) for doc_content in docs]
        
        # Invoke document combiner chain synchronously
        response = self.combine_docs_chain.invoke(
            {
                "input": query,
                "context": langchain_docs
            }
        )
        return response.get("answer", "")
