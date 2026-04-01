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
    raise ValueError("GOOGLE_API_KEY tidak ditemukan di .env")

client = genai.Client(api_key=API_KEY)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_index():
    print("[INFO] Building FAISS index...")

    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()

    splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    )

    docs = splitter.split_text(text)

    db = FAISS.from_texts(docs, embeddings)
    db.save_local(INDEX_PATH)

    print("[INFO] Index saved.")

def load_index():
    if not os.path.exists(INDEX_PATH):
        build_index()

    print("[INFO] Loading FAISS index...")
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve(query, db, k=5):
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
- Jangan menambahkan informasi dari luar
- Jika tidak ditemukan, jawab: "Tidak ditemukan dalam konteks"

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
# MAIN PIPELINE
# =========================
def main():
    db = load_index()

    query = "Apa itu FastAPI?"

    print("\n[QUERY]")
    print(query)

    docs = retrieve(query, db)

    print("\n[RETRIEVED DOCS]")
    for d in docs:
        print("-", d.page_content[:100])

    docs = rerank(query, docs)

    context = "\n".join([d.page_content for d in docs])

    print("\n[FINAL CONTEXT]")
    print(context)

    answer = generate_answer(query, context)

    print("\n[ANSWER]")
    print(answer)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()