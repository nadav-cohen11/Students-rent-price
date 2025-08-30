import asyncio
import aiohttp
import os
from logger import setup_logger
from core.scraper import scrape_page
from dotenv import load_dotenv
from typing import Any

load_dotenv()

logger = setup_logger()

def get_config_value(key: str, default: Any) -> Any:
    return type(default)(os.getenv(key, default))

async def run_batch(session: aiohttp.ClientSession, start_page: int, end_page: int) -> None:
    logger.info(f"Starting batch: {start_page} to {end_page}")
    tasks = [scrape_page(session, page) for page in range(start_page, end_page + 1)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for idx, result in enumerate(results, start=start_page):
        if isinstance(result, Exception):
            logger.error(f"Error scraping page {idx}: {result}")
    logger.info(f"Completed batch: {start_page} to {end_page}")

async def main() -> None:
    total_pages = int(get_config_value('TOTAL_PAGES', 1000))
    batch_size = int(get_config_value('BATCH_SIZE', 15))
    max_concurrency = int(get_config_value('MAX_CONCURRENCY', 10))
    sleep_between_batches = int(get_config_value('SLEEP_BETWEEN_BATCHES', 60))
    timeout_seconds = int(get_config_value('TIMEOUT_SECONDS', 120))

    connector = aiohttp.TCPConnector(ssl=False, limit=max_concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for batch_start in range(1, total_pages + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, total_pages)
            try:
                await run_batch(session, batch_start, batch_end)
            except asyncio.CancelledError:
                logger.warning("Batch cancelled by user.")
                break
            except Exception as e:
                logger.exception(f"Batch error for pages {batch_start}-{batch_end}: {e}")
            logger.info(f"Sleeping for {sleep_between_batches} seconds after batch {batch_start}-{batch_end}")
            await asyncio.sleep(sleep_between_batches)

    logger.info("All batches completed.")

if __name__ == "__main__":
    logger.info("Script initialization.")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user.")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    logger.info("Script terminated.")
