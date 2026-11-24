from pathlib import Path
import sys
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.nlp.preprocessor import clean_text

ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "Product Quality": [
        "quality", "build", "durability", "durable", "material", "fabric",
        "sturdy", "cheap", "broken", "defective", "flawless", "plastic",
        "stitching", "craftsmanship", "freshness", "fresh", "moldy", "stale"
    ],
    "Delivery & Shipping": [
        "delivery", "deliver", "shipping", "shipped", "arrive", "arrived",
        "fast", "late", "delayed", "package", "packaging", "box", "driver",
        "days", "on time", "cold", "spilled"
    ],
    "Price & Value": [
        "price", "pricing", "cost", "value", "penny", "expensive", "overpriced",
        "worth", "money", "affordable", "steal", "budget", "ripoff"
    ],
    "Customer Service": [
        "service", "support", "staff", "customer care", "response", "helpful",
        "rude", "reception", "agent", "refund", "replacement", "resolved"
    ],
    "Product Usability": [
        "easy to use", "setup", "set up", "interface", "manual", "confusing",
        "ergonomic", "battery", "charging", "fit", "sizing", "lag", "buggy", "software"
    ],
    "Taste & Food": [
        "taste", "tasty", "flavor", "delicious", "bland", "greasy",
        "portion", "menu", "spicy", "sweet", "buffet", "food"
    ],
}


def extract_aspects_list(text: str, category: str = "General") -> List[str]:
    """Extract matched operational aspects as a deduplicated list."""
    norm = clean_text(text)
    if not norm:
        return ["General Experience"]

    matched: List[str] = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in norm for kw in keywords):
            matched.append(aspect)

    if not matched:
        cat_lower = str(category or "").lower()
        if "food" in cat_lower or "grocery" in cat_lower:
            matched.append("Taste & Food")
        elif "delivery" in cat_lower:
            matched.append("Delivery & Shipping")
        else:
            matched.append("General Experience")

    return matched


def extract_aspects(text: str, category: str = "General") -> str:
    """Return semicolon-separated aspect labels compatible with DB and dashboards."""
    return ";".join(extract_aspects_list(text, category))


if __name__ == "__main__":
    print(extract_aspects("Product is good but delivery took too long and support was slow"))