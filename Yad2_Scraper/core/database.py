from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError, DuplicateKeyError
from config.settings import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME
from logger import logger
from typing import Dict, Any

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
logger.info("MongoDB connection initialized")

db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

collection.create_index(
    [("ad_url", ASCENDING)],
    unique=True,
    name="unique_ad_url"
)

collection.create_index(
    [("city", ASCENDING), ("price", ASCENDING), ("square_meters", ASCENDING)],
    name="city_price_size_idx"
)

def insert_listing(listing: Dict[str, Any]) -> None:
    try:
        required_fields = ["price", "address"]
        missing = [field for field in required_fields if not listing.get(field)]
        if missing:
            logger.warning(f"Listing at {listing.get('ad_url', 'N/A')} missing required fields: {missing}. Skipping insert.")
            return
        if collection.find_one({"ad_url": listing["ad_url"]}):
            logger.info(f"Listing already exists: {listing['ad_url']}")
            return
        collection.insert_one(listing)
        logger.info(f"Inserted new listing: {listing['ad_url']}")
    except DuplicateKeyError:
        logger.warning(f"Duplicate key encountered during insert: {listing['ad_url']}")
    except PyMongoError as e:
        logger.error(f"[DB] MongoDB insert error: {e}")
