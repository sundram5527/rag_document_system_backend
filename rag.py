#1.PDF Loader
import fitz
from config import GROQ_API_KEY

def load_pdf(file_bytes: bytes):
    doc =fitz.open(stream=file_bytes, filetype="pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    return text

#2.HTML Loader
from bs4 import BeautifulSoup
def load_html(file_bytes: bytes):
    
    
    html = file_bytes.decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    for script in soup(["script","style"]):
        script.decompose()

    text = soup.get_text(separator=" ")

    return text
#text cleaning
import re

def clean(text):

    text = re.sub(r"\s+"," ",text)
    return text.strip()
#chunking

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def create_chunks(text):
    return splitter.split_text(text)

#embedding
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def generate_embedding(chunks):

    embeddings = embedding_model.encode(chunks, normalize_embeddings=True)

    return embeddings

#metadata generation

def create_metadata(source, chunk_id, text):

    return{"source":source, "chunk_id":chunk_id, "length":len(text)}

#Qdrant creation
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")

from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size= 384,
        distance= Distance.COSINE
    )
)

#store chunks
from qdrant_client.models import PointStruct
import uuid

def store_chunks(chunks, embeddings, source):

    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):

        points.append(PointStruct(id = str(uuid.uuid4()),
                                  vector = emb.tolist(),
                                  payload = {
                                      "text":chunk,
                                      "source":source,
                                      "chunk_id":i
                                  }))
        
    client.upsert(collection_name="documents", points= points)

#full ingestion pipeline
import os

def ingest_document(file_bytes: bytes, filename:str):

    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        doc = load_pdf(file_bytes)

    elif lower_name.endswith(".html"):
        doc = load_html(file_bytes)

    else:
        raise ValueError("unsupported file")

    text_clean = clean(doc)

    text_chunk = create_chunks(text_clean)

    embeddings = generate_embedding(text_chunk)

    store_chunks(text_chunk, embeddings, filename)
    return("document uploaded successfully")


#Rag pipeline
#user query => query embedding => similiarity search => top k chunks => retrieve ranking => prompt augementation => LLM =>final answer 

#from qdrant_client import QdrantClient

#client = QdrantClient(":memory:")

collection_name1 = "documents"

from sentence_transformers import SentenceTransformer

embedding_model = None

def get_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return embedding_model


def embed_query(query):

    return embedding_model.encode(query, normalize_embeddings=True).tolist()

def retrieve(query, k=5):

    query_vector1 = embed_query(query)

    results = client.query_points(collection_name="documents", query=query_vector1, limit=k)

    return results.points



def context(retrieve):
    context= []
    sources= []

    for r in retrieve:
        context.append(r.payload['text'])
        sources.append(r.payload['source'])

    return "\n\n".join(context), sources

custom_prompt = """
You are a helpful AI Assistant. Only answer from the given context. If not able to answer say so.

Context:
{context}

Question:
{question}

Answer:
"""

from langchain_groq import ChatGroq



llm = ChatGroq(model="llama-3.3-70b-versatile",api_key=GROQ_API_KEY, temperature=1)

def generate_answer(question):

    retrieve1 = retrieve(question)
    context1, sources = context(retrieve1)
    custom_prompt1 = custom_prompt.format(context= context1, question= question)
    response = llm.invoke(custom_prompt1)

    return {"answer":response.content,"sources":list(set(sources))}




