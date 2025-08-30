import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME: str = os.getenv("DATABASE_NAME", "yad2")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "listings")
TOTAL_PAGES: int = int(os.getenv("TOTAL_PAGES", 10))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 5))
MAX_CONCURRENCY: int = int(os.getenv("MAX_CONCURRENCY", 10))
SLEEP_BETWEEN_BATCHES: int = int(os.getenv("SLEEP_BETWEEN_BATCHES", 60))
TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", 120))