from unittest import result

from fastapi import APIRouter
from openai import OpenAI
from langchain_core.embeddings import Embeddings
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

load_dotenv()


router = APIRouter(
    prefix="/query",
    tags=["[query]"]
)

openrouter_client = OpenAI(
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

embedding_model = OpenRouterCohereEmbeddings()

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name="sample_collection",
    url="http://localhost:6333"
)



@router.post("/")
async def query(question: str):
    search_results = vector_db.similarity_search(query = question)
    
    # print(f"Search results for question : {question}: {search_results}")

    context = " ".join([
        f"[Page {result.metadata.get('page')}] {result.page_content}"
        for result in search_results
    ])

    SYSTEM_PROMPT = f"""
    You are a helpful assistant that provides information based on the provided context. If the context does not contain the answer, respond with 'I don't know.

    Also include the page number of the context if applicable .
    Context: 
    {context}
    """

    response = openrouter_client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct",
        messages=[
            {"role": "system","content": SYSTEM_PROMPT},
            {"role": "user","content": question},  
        ],
    )




    return {"question": question, "answer": response.choices[0].message.content}

