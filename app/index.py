from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

pdf_path = Path(__file__).parent / "data" / "sample.pdf"

from langchain_qdrant import QdrantVectorStore

from openai import OpenAI

from langchain_core.embeddings import Embeddings

import os
from dotenv import load_dotenv

load_dotenv()


# Pdf loader 
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()


# Splitting the data into chunks 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 400
)

chunks = text_splitter.split_documents(documents=docs)


# Embedding the chunks 


# embedding_model = OpenAIEmbeddings(
#     model="text-embedding-3-large"
# )

# embedding_model = OpenAIEmbeddings(
#     model="cohere/embed-5-fast",
#     api_key=os.getenv("OPENROUTER_KEY"),
#     base_url="https://openrouter.ai/api/v1"
# )


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




# Create a Qdrant vector store 

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="sample_collection",
    url="http://localhost:6333"
)

print("Vector store created and data inserted successfully.")