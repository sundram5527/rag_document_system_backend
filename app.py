from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from rag import ingest_document, generate_answer

app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rag-document-system-11.vercel.app"],  # Or ["*"] for all origins
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()

    ingest_document(data, file.filename)

    return{"filename":file.filename,"status":"success"}

@app.get("/ask")
def ask(question: str):
    
    return generate_answer(question)