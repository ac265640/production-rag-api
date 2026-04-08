# main.py

from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Production RAG API")

app.include_router(router)


@app.get("/")
def root():
    return {"message": "RAG API running"}