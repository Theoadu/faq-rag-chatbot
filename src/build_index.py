import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from utils import logger, setup_logging

load_dotenv()
setup_logging()

DOCUMENT_PATH = "data/faq_document.txt"
CHROMA_PATH = "data/chroma"
CHUNKS_PATH = "data/chunks.json"

def load_document(path: str) -> str:
    if not os.path.exists(path):
        logger.error(f"Document not found at {path}")
        raise FileNotFoundError(f"Document not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    logger.info(f"Loaded document: {len(text)} characters")
    return text

def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    logger.info(f"Created {len(chunks)} chunks.")
    if len(chunks) < 20:
        raise ValueError("Document too short: fewer than 20 chunks generated.")
    return chunks

def build_and_save_chroma_index(chunks: list, embedding_model: str = "text-embedding-3-small"):
    os.makedirs("data", exist_ok=True)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    logger.info("Building ChromaDB index...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="hr_faq"
    )
    logger.info(f"ChromaDB index saved to {CHROMA_PATH}")

def main():
    try:
        text = load_document(DOCUMENT_PATH)
        chunks = chunk_text(text)
        build_and_save_chroma_index(chunks)
    except Exception as e:
        logger.exception("Failed to build index")
        raise

if __name__ == "__main__":
    main()