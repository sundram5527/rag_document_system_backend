#1.PDF Loader
import fitz
from dotenv import load_dotenv
import os

#load env variables
load_dotenv()

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


import voyageai

client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))


def generate_embedding(chunks):

    embeddings = client.embed(chunks, model="voyage-3-lite").embeddings
    return embeddings

#metadata generation

def create_metadata(source, chunk_id, text):

    return{"source":source, "chunk_id":chunk_id, "length":len(text)}

#Qdrant creation
from qdrant_client import QdrantClient

client1 = QdrantClient(":memory:")

from qdrant_client.models import Distance, VectorParams

client1.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size= 512,
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
                                  vector = emb,
                                  payload = {
                                      "text":chunk,
                                      "source":source,
                                      "chunk_id":i
                                  }))
        
    client1.upsert(collection_name="documents", points= points)

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

def embed_query(query):
    query_embedding = client.embed(query, model="voyage-3-lite").embeddings[0]
     
    return query_embedding

def retrieve(query, k=5):

    query_vector1 = embed_query(query)

    results = client1.query_points(collection_name="documents", query=query_vector1, limit=k)

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



llm = ChatGroq(model="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY"), temperature=1)

def generate_answer(question):

    retrieve1 = retrieve(question)
    context1, sources = context(retrieve1)
    custom_prompt1 = custom_prompt.format(context= context1, question= question)
    response = llm.invoke(custom_prompt1)

    return {"answer":response.content,"sources":list(set(sources))}




