import os
import sys
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()

# Global Configuration Variables
TEMPERATURE = 0.1  # Set strictly low for clinical accuracy
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
DB_FAISS_PATH = os.path.join("vectorstore", "db_faiss")
ENABLE_HYBRID_SEARCH = True
TOP_K = 3


def get_hybrid_retriever(db, top_k=TOP_K, enable_hybrid=ENABLE_HYBRID_SEARCH):
    """Creates an EnsembleRetriever combining FAISS (dense) and BM25 (sparse) searches."""
    dense_retriever = db.as_retriever(search_kwargs={'k': top_k})
    
    if not enable_hybrid:
        return dense_retriever
        
    # Extract all documents from the FAISS docstore to construct the BM25 sparse index
    all_docs = list(db.docstore._dict.values())
    if not all_docs:
        print("Warning: FAISS docstore is empty. Falling back to dense semantic retriever.")
        return dense_retriever
        
    # Initialize BM25 retriever
    sparse_retriever = BM25Retriever.from_documents(all_docs)
    sparse_retriever.k = top_k
    
    # Ensemble combining dense and sparse search
    # Weight settings: 70% dense semantic, 30% sparse keyword matching
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.7, 0.3]
    )
    return ensemble_retriever


def main():
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY is not set in environment variables.")
        sys.exit(1)

    print(f"Initializing LLM '{GROQ_MODEL_NAME}' with temperature = {TEMPERATURE}...")
    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=512,
        api_key=GROQ_API_KEY,
    )

    print(f"Loading FAISS database from '{DB_FAISS_PATH}'...")
    if not os.path.exists(DB_FAISS_PATH):
        print(f"Error: Vector store path '{DB_FAISS_PATH}' does not exist. Please run create_memory_for_llm.py first.")
        sys.exit(1)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    print("Building Hybrid Retrieval Pipeline (FAISS Dense + BM25 Sparse)...")
    retriever = get_hybrid_retriever(db, top_k=TOP_K, enable_hybrid=ENABLE_HYBRID_SEARCH)

    # Step 3: Build RAG chain
    print("Pulling prompt templates from hub...")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Document combiner chain (stuff documents into prompt)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    # Retrieval chain (retriever + doc combiner)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Now invoke with a single query
    user_query = input("\nWrite Query Here: ")
    if not user_query.strip():
        print("Empty query. Exiting.")
        return

    print("Querying Medical GPT Pipeline...")
    response = rag_chain.invoke({'input': user_query})
    
    print("\nRESULT: ", response["answer"])
    print("\nSOURCE DOCUMENTS:")
    for doc in response["context"]:
        source = doc.metadata.get("source_name") or os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
        print(f"- {source} (Page {page}) -> {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()