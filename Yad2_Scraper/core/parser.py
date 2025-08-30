import re
from bs4 import BeautifulSoup
from logger import logger
from core.embeddings import generate_embedding
from typing import Optional, Dict, Any

def parse_listing(listing_html: str, ad_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        print(listing_html[:500])
        soup = BeautifulSoup(listing_html, 'html.parser')
        description = extract_description(soup)
        description_embedding = generate_embedding(description) if description else None
        result = {
            "price": extract_price(soup),
            "address": extract_address(soup),
            "description_embedding": description_embedding,
            **extract_location(soup),
            **extract_rooms_floors_size(soup),
            "published_at": extract_published_date(soup),
            "type": "rent",
            "ad_url": ad_url,
        }
        features = extract_features(soup)
        property_details = extract_property_details(soup)
        result.update(features)
        result.update(property_details)
        logger.debug(f"Parsed listing ad_url: {result['ad_url']}")
        return result
    except Exception as e:
        logger.error(f"[Parser] Error parsing listing: {e}")
        return None

def extract_price(soup: BeautifulSoup) -> Optional[str]:
    tag = soup.find('span', {'data-testid': 'price'})
    if tag:
        price = tag.get_text(strip=True)
        if price == "לא צוין מחיר":
            return None
        return re.sub(r"[₪,]", "", price)
    return None

def extract_address(soup: BeautifulSoup) -> Optional[str]:
    tag = soup.find('h1', {'data-testid': 'heading'})
    return tag.get_text(strip=True) if tag else None

def extract_location(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    tag = soup.find('h2', {'data-testid': 'address'})
    apartment_style, neighborhood, city = None, None, None
    if tag:
        parts = [p.strip() for p in tag.get_text(strip=True).split(',')]
        if len(parts) == 3:
            apartment_style, neighborhood, city = parts
        elif len(parts) == 2:
            apartment_style, city = parts
        elif len(parts) == 1:
            apartment_style = parts[0]
    return {
        "apartment_style": apartment_style,
        "neighborhood": neighborhood,
        "city": city
    }

def extract_rooms_floors_size(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    rooms = floor = total_floors = size_sqm = None
    span_tags = soup.find_all("span")
    for tag in span_tags:
        if "חדרים" in tag.text:
            previous = tag.find_previous_sibling("span")
            if previous and previous.get("data-testid") == "building-text":
                rooms = previous.text.strip()
        if tag.get_text(strip=True) == "קומה":
            next_span = tag.find_next_sibling("span")
            if next_span:
                match = re.match(r"(\d+)\s*/\s*(\d+)", next_span.get_text(strip=True))
                if match:
                    floor, total_floors = match.groups()
                else:
                    floor = next_span.get_text(strip=True)
        if tag.get("data-testid") == "building-text":
            next_span = tag.find_next_sibling("span")
            if next_span and "מ״ר" in next_span.get_text():
                size_sqm = tag.get_text(strip=True)
    return {
        "rooms": rooms,
        "floor": floor,
        "total_floors": total_floors,
        "size_sqm": size_sqm
    }

def extract_published_date(soup: BeautifulSoup) -> Optional[str]:
    tag = soup.find("span", class_="report-ad_createdAt__tqSM6")
    if tag:
        match = re.search(r"\d{2}/\d{2}/\d{2}", tag.get_text())
        if match:
            return match.group(0)
    return None

def extract_features(soup: BeautifulSoup) -> Dict[str, bool]:
    feature_map = {
        "מעלית": "elevator",
        "גישה לנכים": "wheelchair_access",
        "מזגן טורנדו": "tornado_ac",
        "דלתות רב-בריח": "multi_bolt_doors",
        "מיזוג": "air_conditioning",
        "סורגים": "bars",
        "מחסן": "storage",
        "דוד שמש": "solar_water_heater",
        "משופצת": "renovated",
        "ממ\"ד": "mamad"
    }
    features: Dict[str, bool] = {}
    ul = soup.find("ul", {"data-testid": "in-property-grid"})
    if ul:
        items = ul.find_all("li", {"data-testid": "in-property-item"})
        for item in items:
            text_el = item.find("span", class_="in-property-item_text__aLvx0")
            if text_el:
                hebrew_feature = text_el.get_text(strip=True)
                english_feature = feature_map.get(hebrew_feature)
                if english_feature:
                    disabled = "in-property-item_disabled__gc5Gt" in item.get("class", [])
                    features[english_feature] = not disabled
    return features

def extract_property_details(soup: BeautifulSoup) -> Dict[str, Any]:
    label_map = {
        "מצב הנכס": "property_condition",
        "חניות": "parking",
        "מחיר למ\"ר": "price_per_meter"
    }
    result: Dict[str, Any] = {}
    section = soup.find("section", class_="")
    if section:
        labels = section.find_all("dd", class_="item-detail_label__FnhAu")
        values = section.find_all("dt", class_="item-detail_value__QHPml")
        for label_tag, value_tag in zip(labels, values):
            hebrew_label = label_tag.get_text(strip=True)
            english_key = label_map.get(hebrew_label)
            if not english_key:
                continue
            value = value_tag.get_text(strip=True)
            if english_key == "price_per_meter":
                clean_value = re.sub(r"[^\d]", "", value)
                if clean_value.isdigit():
                    result[english_key] = int(clean_value)
            elif english_key == "entry_date" and value == "כניסה גמישה":
                continue
            else:
                result[english_key] = value
    return result

def extract_description(soup: BeautifulSoup) -> Optional[str]:
    tag = soup.find('p', class_='description_description__9t6rz')
    return tag.get_text(strip=True) if tag else None