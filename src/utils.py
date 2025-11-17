import logging
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

moderation_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAGChatbot")

def setup_logging():
    """Ensure log directory exists."""
    os.makedirs("logs", exist_ok=True)

def moderate_content(text: str) -> bool:
    """
    Use OpenAI Moderation API to check if input is safe.
    Returns True if content is flagged (unsafe), False if safe.
    """
    try:
        response = moderation_client.moderations.create(input=text)
        flagged = response.results[0].flagged
        if flagged:
            categories = response.results[0].categories
            logger.warning(f"🚨 Moderation flagged input: {text[:100]}... | Categories: {categories}")
        else:
            logger.debug("Input passed moderation.")
        return flagged
    except Exception as e:
        logger.error(f"Moderation API error: {e}")
        # Fail-safe: allow if moderation fails (or block — your policy)
        return False  