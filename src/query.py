import os
import json
import argparse
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from utils import logger, setup_logging, moderate_content

load_dotenv()
setup_logging()

CHROMA_PATH = "data/chroma"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_relevant_chunks(query: str, k: int = 2):
    if not os.path.exists(CHROMA_PATH):
        raise RuntimeError("ChromaDB index not found. Run `python src/build_index.py` first.")
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="hr_faq"
    )
    docs = vectorstore.similarity_search(query, k=k)
    chunks = [doc.page_content for doc in docs]
    logger.debug(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
    return chunks

def generate_answer(question: str, chunks: list) -> str:
    context = "\n\n".join(chunks)
    prompt = (
        "You are an HR support assistant. Answer the question using ONLY the provided context. "
        "If the context does not contain the answer, say: 'I don't know based on the provided documentation.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=250
    )
    answer = response.choices[0].message.content.strip()
    logger.info(f"Generated answer for: {question[:50]}... -> {answer[:60]}...")
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True, help="User question")
    args = parser.parse_args()

    question = args.question.strip()
    if not question:
        logger.error("Empty question provided.")
        print(json.dumps({
            "error": "Question cannot be empty.",
            "user_question": "",
            "system_answer": "Error: Empty question.",
            "chunks_related": []
        }, indent=2))
        return

    if moderate_content(question):
        logger.warning(f"Blocked unsafe query: {question}")
        result = {
            "user_question": question,
            "system_answer": "I cannot assist with that request.",
            "chunks_related": [],
            "moderation_flagged": True
        }
        print(json.dumps(result, indent=2))
        return

    try:
        logger.info(f"Processing query: {question}")
        chunks = retrieve_relevant_chunks(question, k=2)
        answer = generate_answer(question, chunks)

        result = {
            "user_question": question,
            "system_answer": answer,
            "chunks_related": chunks
        }
        logger.info(f"Successfully answered: {question[:50]}...")
        print(json.dumps(result, indent=2))

    except Exception as e:
        logger.exception(f"Error processing query: {question}")
        print(json.dumps({
            "error": str(e),
            "user_question": question,
            "system_answer": "An error occurred while processing your request.",
            "chunks_related": []
        }, indent=2))

if __name__ == "__main__":
    main()