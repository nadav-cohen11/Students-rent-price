import aiohttp
import random
from bs4 import BeautifulSoup
from aiohttp import ClientSession
from utils.constants import USER_AGENTS, COMMON_HEADERS
from core.parser import parse_listing
from core.database import insert_listing
from logger import logger
import asyncio
from typing import Optional, Set, Dict
from core.selenium_fetcher import fetch_listing_selenium

FAILED_PAGES: Set[int] = set()
fail_count: Dict[str, int] = {}

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    modal = soup.find("div", class_="bz-modal")
    if modal:
        modal.decompose()
    return str(soup)

async def fetch_page(
    session: ClientSession,
    url: str,
    retries: int = 2,
    backoff: float = 1.5
) -> Optional[str]:
    key = url.split("?")[0]
    fail_count[key] = fail_count.get(key, 0)

    if fail_count[key] >= 5:
        logger.warning("Too many failures for %s, skipping...", url)
        return None
    
    for attempt in range(retries):
        try:
            headers = COMMON_HEADERS.copy()
            headers["User-Agent"] = random.choice(USER_AGENTS)
            headers["Accept-Encoding"] = random.choice(["gzip, deflate, br", "identity"])
            headers["Referer"] = f"https://www.yad2.co.il/realestate/rent?page={random.randint(1, 5)}"

            async with session.get(url, headers=headers) as response:
                if response.status == 429:
                    delay = backoff * (attempt + 1)
                    logger.warning("429 Too Many Requests – Retrying %s in %.1fs", url, delay)
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return await response.text()

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
            await asyncio.sleep(backoff * (attempt + 1))
        except Exception as e:
            logger.error("Unexpected error fetching %s: %s", url, e)
            await asyncio.sleep(backoff * (attempt + 1))

    raise RuntimeError(f"Max retries exceeded for: {url}")

async def scrape_page(session: ClientSession, page_number: int) -> None:
    if page_number in FAILED_PAGES:
        return
    
    url = f'https://www.yad2.co.il/realestate/rent?topArea=2&zoom=10&page={page_number}'
    
    try:
        html = await fetch_page(session, url)
        if not html:
            logger.warning(f"No HTML returned for page {page_number}")
            FAILED_PAGES.add(page_number)
            return
        html = clean_html(html)
        soup = BeautifulSoup(html, 'html.parser')
        listings = soup.select('div.card_cardBox__KLi9I')

        for listing_html in listings:
            try:
                link_tag = listing_html.select_one('a.item-layout_itemLink__CZZ7w')
                if link_tag and link_tag.get('href'):
                    ad_url = f"https://www.yad2.co.il{link_tag['href']}"
                    logger.debug("Fetching ad detail: %s", ad_url)
                    ad_html = fetch_listing_selenium(ad_url)
                    await asyncio.sleep(random.uniform(2, 4))
                    parsed = parse_listing(ad_html, ad_url=ad_url) if ad_html else None
                    if parsed:
                        parsed["ad_url"] = ad_url
                        insert_listing(parsed)
                    else:
                        logger.warning("Failed to parse listing at %s", ad_url)
            except Exception as e:
                logger.error("Listing parse error on page %d: %s", page_number, e)
    except Exception as e:
        FAILED_PAGES.add(page_number)
        logger.exception("Page scrape error for page %d: %s", page_number, e)
