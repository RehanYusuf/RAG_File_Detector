import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


class OpenRouterCohereEmbeddings(Embeddings):

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

        self.model = "cohere/embed-5-fast"

    def embed_documents(self, texts):

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            extra_body={"input_type": "search_document"}
        )

        return [
            item.embedding
            for item in sorted(response.data, key=lambda x: x.index)
        ]

    def embed_query(self, text):

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            extra_body={"input_type": "search_query"}
        )

        return response.data[0].embedding


router = APIRouter(
    tags=["PDF Upload"]
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



@router.post("/upload-and-query")
async def upload_and_query(
    question: str,
    file: UploadFile = File(...)
):

    # Validate PDF
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty."
        )

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF must be smaller than 20 MB."
        )

    file_id = str(uuid4())
    pdf_path = UPLOAD_DIR / f"{file_id}.pdf"

    pdf_path.write_bytes(content)

    try:
        # Load PDF
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        # Split PDF into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in PDF."
            )

        # Generate embeddings
        embedding_model = OpenRouterCohereEmbeddings()

        collection_name = f"pdf_{file_id.replace('-', '')}"

        # Store vectors
        vector_db = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=collection_name,
            url="http://localhost:6333"
        )

        # Retrieve relevant chunks
        search_results = vector_db.similarity_search(
            query=question,
            k=4
        )

        context = "\n\n".join([
            f"[Page {doc.metadata.get('page', 0) + 1}]\n"
            f"{doc.page_content}"
            for doc in search_results
        ])

        # Generate answer
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are a PDF question-answering assistant.

                    Answer the user's question using only the
                    provided PDF context.

                    If the answer is not available, say:
                    I don't know based on the uploaded PDF.

                    Mention page numbers wherever applicable.

                    PDF Context:
                    {context}
                    """
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        return {
            "filename": file.filename,
            "question": question,
            "answer": response.choices[0].message.content,
            "sources": [
                {
                    "page": doc.metadata.get("page", 0) + 1,
                    "content": doc.page_content
                }
                for doc in search_results
            ]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
