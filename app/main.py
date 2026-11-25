from fastapi import FastAPI
from app.routes.query import router as query_router
from app.routes.pdf_upload import router as pdf_upload_router


apper = FastAPI(
    title="Rag Api",
    description="This is a Rag Api",
    version="0.1.0"
)

apper.include_router(query_router)

apper.include_router(pdf_upload_router)
