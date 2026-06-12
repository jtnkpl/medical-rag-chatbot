import os
import json
import datetime
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data"
DB_FAISS_PATH = os.path.join("vectorstore", "db_faiss")
METADATA_PATH = os.path.join("vectorstore", "metadata.json")


def load_pdf_files(data_dir):
    """Load raw PDF files from the specified directory."""
    loader = DirectoryLoader(
        data_dir,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def create_chunks(extracted_data):
    """Chunk the extracted pages using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def get_embedding_model():
    """Retrieve and cache the HuggingFace embeddings model."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


def load_metadata(metadata_path=METADATA_PATH):
    """Load the metadata catalog JSON. Return empty structure if not found."""
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "indexed_files": {},
        "total_chunks": 0,
        "last_updated": "Never"
    }


def save_metadata(metadata, metadata_path=METADATA_PATH):
    """Save the metadata catalog to JSON with strict context management to release locks immediately."""
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)


def initialize_metadata(db, metadata_path=METADATA_PATH):
    """Scan existing FAISS database docstore to rebuild metadata if missing."""
    metadata = load_metadata(metadata_path)
    
    # If index is populated but metadata catalog is empty, let's sync them
    if db and not metadata["indexed_files"]:
        all_docs = list(db.docstore._dict.values())
        file_chunks = {}
        for doc in all_docs:
            src = doc.metadata.get("source_name") or os.path.basename(doc.metadata.get("source", "unknown"))
            file_chunks[src] = file_chunks.get(src, 0) + 1
            
        current_time = datetime.datetime.now().isoformat()
        for filename, count in file_chunks.items():
            file_path = os.path.join(DATA_PATH, filename)
            file_size_mb = 0.0
            if os.path.exists(file_path):
                file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
            else:
                uploads_path = os.path.join(DATA_PATH, "uploads", filename)
                if os.path.exists(uploads_path):
                    file_size_mb = round(os.path.getsize(uploads_path) / (1024 * 1024), 2)

            metadata["indexed_files"][filename] = {
                "file_size_mb": file_size_mb,
                "ingestion_timestamp": current_time,
                "chunks_count": count
            }
            
        metadata["total_chunks"] = db.index.ntotal
        metadata["last_updated"] = current_time
        save_metadata(metadata, metadata_path)
    return metadata


def ingest_file(file_path, db_path=DB_FAISS_PATH, metadata_path=METADATA_PATH):
    """Parse, chunk, and embed a PDF or TXT file into the FAISS database."""
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)

    # 1. Load document
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif ext == '.txt':
        from langchain_core.documents import Document
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        documents = [Document(page_content=text, metadata={"source": file_path, "page": 0})]
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Only PDF and TXT are supported.")

    if not documents:
        return 0

    # 2. Chunk documents
    text_chunks = create_chunks(documents)
    
    # Enrich metadata for each chunk
    current_time = datetime.datetime.now().isoformat()
    for i, chunk in enumerate(text_chunks):
        chunk.metadata["source_name"] = filename
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1
        chunk.metadata["chunk_index"] = i
        chunk.metadata["ingestion_timestamp"] = current_time

    # 3. Load or create FAISS vectorstore
    embedding_model = get_embedding_model()
    if os.path.exists(db_path):
        db = FAISS.load_local(
            db_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        db.add_documents(text_chunks)
    else:
        db = FAISS.from_documents(text_chunks, embedding_model)

    # Re-save index, which handles local file serialization and automatically closes handles
    db.save_local(db_path)

    # 4. Update metadata.json catalog
    metadata = load_metadata(metadata_path)
    metadata["indexed_files"][filename] = {
        "file_size_mb": file_size_mb,
        "ingestion_timestamp": current_time,
        "chunks_count": len(text_chunks)
    }
    metadata["total_chunks"] = db.index.ntotal
    metadata["last_updated"] = current_time
    save_metadata(metadata, metadata_path)

    # Explicitly clear variables to encourage Python garbage collector to free DLL handles
    del db
    del embedding_model
    import gc
    gc.collect()

    return len(text_chunks)


if __name__ == "__main__":
    print("Initializing FAISS Vector Store from default data directory...")
    os.makedirs(DATA_PATH, exist_ok=True)
    
    # Load default files
    docs = load_pdf_files(data_dir=DATA_PATH)
    if docs:
        print(f"Loaded {len(docs)} pages.")
        chunks = create_chunks(docs)
        print(f"Created {len(chunks)} chunks.")
        
        current_time = datetime.datetime.now().isoformat()
        for i, chunk in enumerate(chunks):
            src_file = os.path.basename(chunk.metadata.get("source", "unknown"))
            chunk.metadata["source_name"] = src_file
            chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1
            chunk.metadata["chunk_index"] = i
            chunk.metadata["ingestion_timestamp"] = current_time
            
        emb_model = get_embedding_model()
        db = FAISS.from_documents(chunks, emb_model)
        db.save_local(DB_FAISS_PATH)
        print(f"FAISS database successfully created and saved to {DB_FAISS_PATH}")
        
        initialize_metadata(db, METADATA_PATH)
        print(f"Metadata catalog initialized at {METADATA_PATH}")
    else:
        print(f"No PDF files found in data directory: {DATA_PATH}")