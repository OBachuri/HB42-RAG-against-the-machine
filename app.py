
from fastapi import FastAPI
# from pydantic import BaseModel

from src.__main__ import RagCLI
from src.r_data_model import AskRequest, MinimalAnswer


app = FastAPI(
    title="RAG API",
    version="0.1.0",
)

rag = RagCLI()
rag.cache = True


@app.post("/ask", response_model=MinimalAnswer)
def ask(request: AskRequest):

    result = rag.answer(request.question, _return_value=True)

    return result
