Antigravity RAG Analyser
A local CI/CD Evaluation and Optimization System for Retrieval-Augmented Generation (RAG) Pipelines built using FastAPI, SQLModel, RAGAS, and Google Gemini.

This platform allows developers to submit a Git repository URL containing a RAG pipeline implementation, automatically clones the repository, sets up an isolated environment, runs automated quality evaluations using RAGAS metrics, and invokes an AI Improvement Agent (powered by Gemini) to analyze scores and recommend specific optimization fixes.

Table of Contents
Understanding Retrieval-Augmented Generation (RAG)
Core Concepts
How RAG Works
Advantages of RAG
Project Architecture & Features
Evaluation Metrics & RAGAS
The AI Improvement Agent
Getting Started & Installation
How to Use the System
1. Implementing
adapter.py
2. Triggering Evaluation via API
3. Fetching Results & Audit Reports
Running Tests
1. Understanding Retrieval-Augmented Generation (RAG)
Core Concepts
Large Language Models (LLMs) are incredibly powerful, but they have key limitations:

Static Knowledge: Their knowledge is frozen at the time of pre-training.
Hallucinations: They may generate convincing-sounding but completely false facts when they lack information.
Lack of Domain Context: They do not have access to private enterprise data or proprietary documents.
Retrieval-Augmented Generation (RAG) addresses these gaps by dynamically fetching relevant external information and injecting it into the prompt before sending it to the LLM.

How RAG Works
The typical RAG process consists of three main phases:

Ingestion: Source documents are split into smaller pieces (chunking), converted into vector embeddings, and indexed in a vector store.
Retrieval: When a user queries the system, the query is converted into a vector. The system performs a similarity search to retrieve the most relevant chunks from the database.
Generation: The original query and the retrieved documents are combined into a structured prompt (e.g., "Answer the question using ONLY the provided text..."). The LLM processes this prompt to produce a grounded, factual response.
Mermaid diagram
Advantages of RAG
Eliminates Retraining/Fine-Tuning Costs: Dynamically references new data without the massive expense of training custom models.
Drastically Reduces Hallucinations: Constrains the model's generation to actual facts present in the retrieved source documents.
Real-Time / Dynamic Data Access: If documents in the source database are updated, the RAG system retrieves the updated version instantly.
Traceability and Citations: Since documents are fetched explicitly, the system can cite the specific files and sections used to formulate the answer.
2. Project Architecture & Features
This project serves as an automated testing/audit harness for RAG pipelines. It functions like a CI/CD server:

Dynamic Code Loader: Clones external Git repositories containing a RAG pipeline, constructs a fresh Python virtual environment (venv), installs its specific dependencies, and runs the code inside an isolated subprocess.
FastAPI API Gateway: Exposes endpoints to start evaluations and check results.
Relational Database (SQLModel): Stores jobs, evaluation metrics, and AI-generated improvement reports locally in SQLite (or PostgreSQL).
RAGAS Evaluation Framework: Automatically tests the custom pipeline on a standard QA benchmark dataset.
AI Audit & Optimization Recommendations: Utilizes Google Gemini to parse results and generate a structured audit report detailing weaknesses and actionable fixes.
3. Evaluation Metrics & RAGAS
Evaluating RAG systems is hard because they consist of both a retrieval component (search) and a generation component (LLM). This system uses RAGAS (Retrieval Augmented Generation Assessment) to evaluate the pipeline across four dimensions:

Metric	Component Evaluated	Description
Faithfulness	Generator	Measures if the generated answer is strictly grounded in the retrieved context. Low scores suggest the LLM is hallucinating or using pre-trained knowledge instead of the provided context.
Answer Relevancy	Generator	Measures how directly the generated answer addresses the user's question. Low scores indicate the LLM is digressing, being redundant, or avoiding the question.
Context Precision	Retriever	Evaluates if the retrieved chunks are sorted correctly by relevance (most relevant chunks first). Low scores imply the ranker/search index is suboptimal.
Context Recall	Retriever	Measures if the retriever successfully fetched all critical facts required to formulate the reference answer (ground truth). Low scores mean chunks are missing from the retrieved context.
4. The AI Improvement Agent
After RAGAS calculates the scores, the Improvement Agent (powered by Gemini) acts as an expert systems architect. It analyzes the scores and generates a structured, Pydantic-validated JSON report containing:

Overall Summary: A high-level assessment of the pipeline's performance.
Weaknesses: Identifies the exact bottlenecks (e.g., High severity weakness: Context Recall is low (0.64) due to small chunk sizes or missing keywords).
Actionable Fixes: Provides concrete engineering solutions (e.g., Implement a reranker (e.g., Cohere or BGE), increase chunk overlap, or restrict the system prompt to avoid hallucinations), along with their expected impact.
5. Getting Started & Installation
Prerequisites
Python 3.12 or higher
Git (installed and available on your system path)
A Google Gemini API Key (you can get a free one from Google AI Studio)
Installation
Clone this repository:

bash

git clone <repository_url>
cd RAG_ANALYSER
Initialize the project virtual environment and install dependencies:

bash

# If using 'uv' (recommended for speed)
uv sync

# Or using standard pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
Configure your environment variables in .env:

ini

# Save this in a .env file at the root of the project
FREE_API_KEY="YOUR_GEMINI_API_KEY"
DATABASE_URL="sqlite:///./antigravity.db"
Running the API Server
Start the FastAPI application using Uvicorn:

bash

# If using uv
uv run uvicorn app.main:app --reload

# If using standard python venv
uvicorn app.main:app --reload
The API will start at http://127.0.0.1:8000. You can access the interactive API docs at http://127.0.0.1:8000/docs.

6. How to Use the System
1. Implementing adapter.py
To evaluate your custom RAG pipeline, you must place an adapter.py file at the root of your Git repository. It must implement a class that inherits from the abstract base class RAGPipeline (defined in the analyzer), implementing retrieve and generate:

python

# adapter.py (in YOUR repository)
from typing import List

class MyCustomRAGPipeline:
    def retrieve(self, query: str) -> List[str]:
        # Connect to your vector store/database and retrieve context chunks
        # Return a list of strings (raw text chunks)
        return [
            "Retrieval-Augmented Generation (RAG) reduces LLM hallucinations.",
            "RAG fetches external, up-to-date knowledge bases to answer user queries."
        ]

    def generate(self, query: str, docs: List[str]) -> str:
        # Construct prompt, query your LLM, and return the generated answer
        return "The primary advantage of RAG is that it reduces hallucinations by grounding responses."
Make sure your repository has a requirements.txt listing any dependencies your adapter needs (e.g., langchain, pinecone-client, openai).

2. Triggering Evaluation via API
Send a POST request to /api/v1/evaluate with your repository URL:

bash

curl -X POST "http://127.0.0.1:8000/api/v1/evaluate" \
     -H "Content-Type: application/json" \
     -d '{"repo_url": "https://github.com/your-username/my-rag-pipeline.git"}'
The system will:

Clone your repository to a temporary workspace.
Create an isolated python venv.
Install your repository's dependencies (requirements.txt or pyproject.toml).
Execute your RAG pipeline's methods inside a subprocess.
Perform RAGAS evaluation on standard benchmark tasks.
Query Gemini to build an optimization blueprint.
3. Fetching Results & Audit Reports
If successful, the endpoint returns a detailed output immediately (as it runs synchronously). You can also fetch the results later using the Job ID:

bash

curl -X GET "http://127.0.0.1:8000/api/v1/jobs/1"
Sample Response Structure:
json

{
  "id": 1,
  "repo_url": "https://github.com/your-username/my-rag-pipeline.git",
  "commit_hash": "a1b2c3d4e5f6...",
  "status": "completed",
  "error_message": null,
  "created_at": "2026-06-13T00:00:00",
  "updated_at": "2026-06-13T00:05:00",
  "metrics": {
    "id": 1,
    "faithfulness": 0.88,
    "answer_relevancy": 0.91,
    "context_precision": 0.85,
    "context_recall": 0.83,
    "created_at": "2026-06-13T00:05:00"
  },
  "report": {
    "id": 1,
    "overall_summary": "The RAG pipeline exhibits moderate performance. The primary bottleneck is context recall, followed by minor hallucinations.",
    "weaknesses": [
      {
        "metric": "context_recall",
        "finding": "Context recall is relatively low (0.83), indicating that the retriever often fails to retrieve all critical information necessary for the ground truth answers.",
        "severity": "High"
      }
    ],
    "recommended_fixes": [
      {
        "weakness_ref": "context_recall",
        "actionable_fix": "Increase retrieval top-k from 3 to 5, and implement a hybrid dense-sparse search index to capture both keyword and semantic matches.",
        "expected_impact": "Should raise context recall from 0.83 to > 0.92."
      }
    ],
    "created_at": "2026-06-13T00:05:00"
  }
}
7. Running Tests
To run the unit tests:

bash

# If using uv
uv run pytest

# If using standard python venv
pytest

