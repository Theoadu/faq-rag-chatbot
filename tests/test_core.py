import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.build_index import load_document, chunk_text

def test_document_loading():
    text = load_document("data/faq_document.txt")
    assert len(text) > 1000

def test_chunk_count():
    text = load_document("data/faq_document.txt")
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 20

def test_chunk_content():
    text = load_document("data/faq_document.txt")
    chunks = chunk_text(text)
    assert "PTO" in chunks[1]  