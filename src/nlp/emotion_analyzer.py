from pathlib import Path
import sys
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.nlp.preprocessor import clean_text

EMOTION_LEXICON: Dict[str, List[str]] = {
    "Angry": [
        "horrible", "terrible", "worst", "disaster", "disgusting", "ripoff",
        "scam", "furious", "hate", "unacceptable", "pathetic", "outrageous",
        "stolen", "trash", "junk", "garbage", "dreadful", "rotten", "moldy"
    ],
    "Disappointed": [
        "disappointed", "disappointing", "let down", "expected better",
        "underwhelming", "below expectations", "not worth", "waste of money"
    ],
    "Frustrated": [
        "frustrat", "annoy", "slow", "late", "delayed", "broken", "damaged",
        "missing", "defective", "confusing", "buggy", "lag", "freeze",
        "freez", "wrong item", "ruined", "inconvenient", "stale", "cold"
    ],
    "Happy": [
        "incredible", "amazing", "delighted", "love", "loved", "phenomenal",
        "stellar", "flawless", "exceeded", "perfection", "best purchase",
        "superb", "brilliant", "fantastic", "wonderful", "awesome", "adore"
    ],
    "Satisfied": [
        "good", "nice", "solid", "decent", "reliable", "satisfied", "worth",
        "works well", "fine", "helpful", "pleasant", "fresh", "tasty",
        "comfortable", "acceptable", "recommend"
    ],
}


def analyze_emotion(text: str, rating: int = 3, sentiment: str = "Neutral") -> str:
    """Return dominant customer emotion for a review."""
    norm = clean_text(text)
    if not norm:
        return "Neutral"

    scores = {emo: 0 for emo in EMOTION_LEXICON}
    for emo, keywords in EMOTION_LEXICON.items():
        for kw in keywords:
            if kw in norm:
                scores[emo] += 1

    sent_clean = str(sentiment or "Neutral").strip().title()
    rating_val = int(rating) if rating is not None else 3

    if rating_val <= 1 or scores["Angry"] > 0:
        if scores["Angry"] >= max(scores["Frustrated"], scores["Disappointed"]):
            return "Angry"
        if rating_val == 1 and scores["Frustrated"] == 0 and scores["Disappointed"] == 0:
            return "Angry"

    if sent_clean == "Negative" or rating_val <= 2:
        if scores["Disappointed"] > scores["Frustrated"] and scores["Disappointed"] > scores["Angry"]:
            return "Disappointed"
        if scores["Frustrated"] > 0 or scores["Disappointed"] > 0:
            return "Frustrated" if scores["Frustrated"] >= scores["Disappointed"] else "Disappointed"
        return "Angry" if rating_val == 1 else "Frustrated"

    if sent_clean == "Positive" or rating_val >= 4:
        if scores["Happy"] > 0 or (rating_val == 5 and "!" in text):
            return "Happy"
        if scores["Satisfied"] > 0 or rating_val >= 4:
            return "Satisfied"

    best_emo = max(scores.items(), key=lambda x: x[1])
    if best_emo[1] > 0:
        return best_emo[0]
    return "Neutral"


if __name__ == "__main__":
    print(analyze_emotion("Delivery was very late and support was unhelpful", rating=2, sentiment="Negative"))