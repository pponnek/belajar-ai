from fastapi import FastAPI
from pydantic import BaseModel

from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

import os
from dotenv import load_dotenv


DATA_PATH = "./data/data.txt"
INDEX_PATH = "faiss_index"

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY tidak ditemukan")

client = genai.Client(api_key=API_KEY)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


def build_index():
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_text(text)

    db = FAISS.from_texts(docs, embeddings)
    db.save_local(INDEX_PATH)

def load_index():
    if not os.path.exists(INDEX_PATH):
        build_index()

    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


db = load_index()


def retrieve(query, k=5):
    return db.similarity_search(query, k=k)

def rerank(query, docs, top_k=3):
    query_terms = set(query.lower().split())

    scored = []
    for doc in docs:
        content_terms = set(doc.page_content.lower().split())
        score = len(query_terms & content_terms)
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [doc for _, doc in scored[:top_k]]

def generate_answer(query, context):
    prompt = f"""
Anda adalah sistem QA berbasis dokumen.

ATURAN:
- Jawab hanya dari konteks
- Jangan menambahkan informasi luar
- Jika tidak ada, jawab: "Tidak ditemukan dalam konteks"

KONTEKS:
{context}

PERTANYAAN:
{query}

JAWABAN:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# =========================
# API ENDPOINT
# =========================
@app.post("/ask")
def ask(request: QueryRequest):
    query = request.query

    docs = retrieve(query)
    docs = rerank(query, docs)

    context = "\n".join([d.page_content for d in docs])

    answer = generate_answer(query, context)

    return {
        "query": query,
        "answer": answer,
        "context": context
    }