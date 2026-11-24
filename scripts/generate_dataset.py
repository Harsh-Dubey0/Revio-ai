import os
import random
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Business categories and corresponding product names and review aspects
CATEGORIES_METADATA = {
    "E-commerce": {
        "products": ["Wireless Earbuds", "Ergonomic Desk Chair", "Electric Kettle", "Waterproof Backpack", "Resistance Bands"],
        "sources": ["Website", "Amazon Store", "Mobile App"],
        "aspects": ["Product Quality", "Delivery Speed", "Value for Money", "Packaging"],
        "templates": {
            "Positive": [
                "The {product} is absolutely incredible. Fast delivery and high build quality.",
                "Best purchase I've made this year. The {product} exceeded my expectations and arrived in 2 days.",
                "Very happy with this {product}! Extremely durable and worth every penny.",
                "Excellent service and solid packaging. The {product} works like a charm."
            ],
            "Neutral": [
                "The {product} is okay. It works as expected but is nothing special.",
                "Decent {product}. Arrived on time but the packaging was slightly bent.",
                "It's an average {product}. Not the best, but not the worst either.",
                "Good enough for daily use, but does have a few minor flaws."
            ],
            "Negative": [
                "Extremely disappointed with the {product}. It broke after only two days of use!",
                "Terrible customer service. The {product} arrived late and damaged.",
                "Save your money! The {product} is very cheaply made and doesn't work well.",
                "The {product} is of poor quality. Delivery took almost two weeks!"
            ],
            "Mixed": [
                "The {product} itself is fantastic, but the delivery took forever and the box was ruined.",
                "Great price for the {product}, however, the instruction manual was missing and customer service was slow.",
                "Decent performance from the {product}, but it gets hot very quickly. Mixed feelings.",
                "The {product} works perfectly, but it feels quite plasticky and fragile."
            ]
        }
    },
    "Food delivery": {
        "products": ["Margherita Pizza", "Butter Chicken with Naan", "Sushi Combo Platter", "Double Cheeseburger Meal", "Vegan Buddha Bowl"],
        "sources": ["Deliveroo", "UberEats", "DoorDash", "Direct Mobile App"],
        "aspects": ["Food Taste", "Delivery Time", "Packaging", "Temperature"],
        "templates": {
            "Positive": [
                "The {product} was delicious and piping hot! Arrived 15 minutes early.",
                "Hands down the best {product} I've ordered. Packaged beautifully and taste was amazing.",
                "Super fresh and tasty! Will definitely order this {product} again.",
                "Very fast delivery, and the {product} was mouth-watering."
            ],
            "Neutral": [
                "The {product} was decent. Portion size was okay but taste was a bit bland.",
                "Delivery was average. The {product} was warm but not hot.",
                "It's a standard {product}. Nothing amazing, but satisfied my hunger.",
                "Food delivered in reasonable time. Taste was completely run-of-the-mill."
            ],
            "Negative": [
                "Horrible experience! The {product} arrived ice cold and spilled all over the bag.",
                "Wait time was over 1.5 hours and the {product} tasted stale. Avoid!",
                "The {product} was extremely greasy and undrinkable/uneatable. Waste of money.",
                "Terrible service. They got my order wrong and the {product} was missing ingredients."
            ],
            "Mixed": [
                "The taste of the {product} was incredible, but it arrived almost cold because delivery was late.",
                "Delivery was lightning fast, but the {product} was missing the extra dipping sauce I paid for.",
                "Tasty {product}, but the packaging was squashed and some food spilled out.",
                "Affordable and fast, but the food quality could be better."
            ]
        }
    },
    "Electronics": {
        "products": ["4K Ultra HD Smart TV", "Noise Cancelling Headphones", "Mechanical Gaming Keyboard", "Portable Power Bank 20k", "Dual-Band Wi-Fi 6 Router"],
        "sources": ["TechStore", "E-Store", "Mobile App", "In-Store Terminal"],
        "aspects": ["Build Quality", "Performance", "Battery Life", "Ease of Use"],
        "templates": {
            "Positive": [
                "The {product} is flawless. Exceptional performance and very sleek design.",
                "Absolutely stellar product. Super easy to set up and the performance is incredible.",
                "I am amazed by the battery life/features of this {product}. High quality build.",
                "Highly recommend! This {product} runs incredibly fast and looks stunning."
            ],
            "Neutral": [
                "The {product} is alright. It gets the job done but does not stand out.",
                "Good build quality but setup was a bit confusing. Decent overall.",
                "The battery life is decent, performance is average for this {product}.",
                "An acceptable device. Not sure if it's worth the full price though."
            ],
            "Negative": [
                "Absolute junk. The {product} stopped working after a software update.",
                "The screen/performance is awful and the {product} kept freezing.",
                "Dreadful experience. Returned it because the {product} was defective out of the box.",
                "Not user-friendly at all. Feels cheap and lags constantly."
            ],
            "Mixed": [
                "Amazing performance, but the battery life on the {product} is very short.",
                "The hardware is top-notch, but the companion software for the {product} is buggy.",
                "Beautiful display, but the built-in speakers of the {product} are very tinny.",
                "Super fast charging, but the cable supplied is incredibly short."
            ]
        }
    },
    "Fashion": {
        "products": ["Slim-Fit Denim Jeans", "Classic White Cotton Tee", "Lightweight Running Shoes", "Windproof Winter Coat", "Leather Minimalist Wallet"],
        "sources": ["FashionHub", "Amazon Store", "Mobile App", "Social Media Shop"],
        "aspects": ["Material Quality", "Size & Fit", "Style & Look", "Comfort"],
        "templates": {
            "Positive": [
                "The fit of this {product} is absolutely perfect! Extremely comfortable material.",
                "Super stylish and feels high quality. Love the color and the fit.",
                "Exceeded my expectations! Very soft fabric and washes well. Love the {product}!",
                "Got so many compliments wearing this {product}. Outstanding comfort and style."
            ],
            "Neutral": [
                "The {product} is okay. The size runs slightly large, but it's usable.",
                "Average quality shirt/jeans. It feels comfortable enough for casual wear.",
                "The material of this {product} is somewhat thin, but it fits fine.",
                "Nothing special. Just a basic {product} that does the job."
            ],
            "Negative": [
                "The sizing of this {product} is completely wrong. It shrunk after the first wash!",
                "Very poor quality stitching. The fabric ripped within a few hours.",
                "Incredibly uncomfortable material. Feels like sandpaper. Avoid the {product}.",
                "The {product} looks nothing like the pictures online. Ruined after one wash."
            ],
            "Mixed": [
                "The {product} looks fantastic and stylish, but the material is a bit stiff and uncomfortable.",
                "The fabric feels extremely soft and premium, but the sizing runs way too small.",
                "Very comfortable {product}, but the color started fading after just two washes.",
                "Fits like a glove, but the zipper feels fragile and cheaply made."
            ]
        }
    },
    "Grocery": {
        "products": ["Organic Avocado Pack", "Fresh Strawberries Box", "Whole Grain Sourdough Bread", "Premium Almond Milk", "Extra Virgin Olive Oil"],
        "sources": ["GroceryGo", "Mobile App", "In-Store Pick", "Instacart"],
        "aspects": ["Freshness", "Packaging", "Value", "Expiration Date"],
        "templates": {
            "Positive": [
                "The {product} was extremely fresh and delicious! Perfect ripeness.",
                "Excellent quality, delivered cold with insulated bag. The {product} is perfect.",
                "Best organic {product} I have bought in a long time. Highly recommend!",
                "Fresh, on-time delivery, and the sourdough/fruit was superb."
            ],
            "Neutral": [
                "The {product} is alright. Freshness is average, but it's acceptable.",
                "Decent price for the {product}, though the bread was slightly dry.",
                "Average grocery order. The {product} arrived with reasonable expiry date.",
                "It's regular {product}. Nothing much to complain about."
            ],
            "Negative": [
                "The {product} arrived moldy and completely rotten! Absolutely disgusting.",
                "The milk was expired and the bread was stale. Unacceptable quality control.",
                "The avocados were bruised and completely smashed due to terrible packaging.",
                "Awful! Ordered fresh fruit/vegetables and received wilted, spoiled goods."
            ],
            "Mixed": [
                "The {product} tastes amazing and fresh, but the packaging was torn.",
                "Great price for {product}, but they arrived with an expiration date of tomorrow.",
                "Super fast grocery delivery, but the avocados were hard as rocks and need days to ripen.",
                "Fresh lettuce/milk, but the delivery person handled the bags roughly."
            ]
        }
    },
    "Hospitality": {
        "products": ["Luxury Suite Night", "Spa and Wellness Package", "Seafood Buffet Experience", "Weekend Getaway Stay", "Conference Hall Rental"],
        "sources": ["Booking.com", "TripAdvisor", "Direct Website", "Expedia"],
        "aspects": ["Service Quality", "Room Cleanliness", "Amenities", "Location"],
        "templates": {
            "Positive": [
                "Incredible stay! The staff was incredibly welcoming and the service was top-tier.",
                "Beautiful, spotless room with amazing views. The amenities were phenomenal.",
                "The spa/buffet was absolute heaven. Exceeded all of our expectations!",
                "Highly professional staff, excellent hygiene, and perfect location."
            ],
            "Neutral": [
                "The service was average. The room was clean but quite outdated.",
                "A decent hotel/restaurant experience. It was comfortable but overpriced.",
                "Everything was okay. Room service took a little long, but food was acceptable.",
                "It's a fine place for a short stopover. Standard breakfast options."
            ],
            "Negative": [
                "Worst hotel experience ever. The room was dirty, and the staff was extremely rude.",
                "The air conditioning did not work and the room smelled like damp. Disastrous stay.",
                "Total ripoff! Noise all night, breakfast was cold, and service was non-existent.",
                "Dirty sheets, broken bathroom fixtures, and they lost my booking."
            ],
            "Mixed": [
                "The location was absolutely breathtaking, but the service at reception was dreadfully slow.",
                "The room was exceptionally clean and modern, but the walls were paper-thin and noisy.",
                "Delicious food and spa services, but the receptionist was rude and unhelpful.",
                "Beautiful amenities, but they charge extra for literally everything including basic Wi-Fi."
            ]
        }
    }
}

# Spelling mistakes helper to simulate realistic user reviews
SPELLING_MISTAKES = {
    "absolutely": ["absolutly", "absoluteley"],
    "incredible": ["incridible", "incredable"],
    "recommend": ["recomended", "recomend"],
    "delivery": ["delvery", "deleivery"],
    "excellent": ["exelent", "excelent"],
    "definitely": ["definately", "definitly"],
    "disappointed": ["disapointed", "dissapointed"],
    "quality": ["qality", "quaity"],
    "service": ["servise", "serivce"],
    "terrible": ["terible", "terrable"],
    "beautiful": ["beatiful", "beautifull"],
    "experience": ["experence", "expirience"]
}

def introduce_spelling_mistakes(text):
    words = text.split()
    for i, word in enumerate(words):
        clean_word = "".join(filter(str.isalpha, word.lower()))
        if clean_word in SPELLING_MISTAKES and random.random() < 0.3:
            mistake = random.choice(SPELLING_MISTAKES[clean_word])
            # Keep capitalization of first letter if present
            if word[0].isupper():
                mistake = mistake.capitalize()
            words[i] = word.replace(clean_word, mistake)
    return " ".join(words)

def generate_random_date(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 86400)
    return start_date + timedelta(days=random_days, seconds=random_seconds)

def generate_review_row(review_id, customer_id, category, sentiment=None):
    metadata = CATEGORIES_METADATA[category]
    product = random.choice(metadata["products"])
    source = random.choice(metadata["sources"])

    if sentiment is None:
        sentiment = random.choice(["Positive", "Neutral", "Negative", "Mixed"])

    rating_map = {
        "Positive": [5, 5, 5, 4],
        "Neutral": [3, 3, 4, 3],
        "Negative": [1, 1, 2, 1],
        "Mixed": [3, 4, 2, 3]
    }

    rating = random.choice(rating_map[sentiment])
    templates = metadata["templates"][sentiment]
    text_template = random.choice(templates)

    review_text = text_template.format(product=product)

    # Introduce spelling mistakes in 15% of reviews
    if random.random() < 0.15:
        review_text = introduce_spelling_mistakes(review_text)

    # Introduce random length extensions sometimes
    if random.random() < 0.1:
        review_text += f" I bought it for my family. They had various thoughts, but overall it is a prominent aspect of our routine."

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 11, 24)
    review_date = generate_random_date(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S")

    # Select randomly 1 to 3 aspects from metadata["aspects"]
    num_aspects = random.randint(1, 2)
    aspects = random.sample(metadata["aspects"], num_aspects)
    aspects_str = ";".join(aspects)

    return {
        "review_id": f"REV_{review_id:06d}",
        "customer_id": f"CUST_{customer_id:05d}",
        "review_text": review_text,
        "rating": rating,
        "review_date": review_date,
        "product_category": category,
        "product_name": product,
        "source": source,
        "sentiment_label": sentiment,
        "aspect_labels": aspects_str
    }

def generate_datasets():
    print("Generating synthetic datasets...")
    os.makedirs("data/generated", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)

    # Generate 10,000 clean training reviews
    reviews = []
    category_list = list(CATEGORIES_METADATA.keys())

    # Ensure balanced or realistic sentiment distribution
    # ~45% Positive, ~20% Neutral, ~20% Negative, ~15% Mixed
    sentiments_distribution = ["Positive"] * 45 + ["Neutral"] * 20 + ["Negative"] * 20 + ["Mixed"] * 15

    for i in range(1, 10001):
        category = random.choice(category_list)
        sentiment = random.choice(sentiments_distribution)
        row = generate_review_row(i, random.randint(1000, 99999), category, sentiment)
        reviews.append(row)

    df = pd.DataFrame(reviews)
    df.to_csv("data/generated/reviews_dataset.csv", index=False)
    print(f"Generated clean dataset of {len(df)} rows saved to data/generated/reviews_dataset.csv")

    # Generate valid sample for testing
    valid_sample = df.head(50).copy()
    valid_sample.to_csv("data/samples/valid_sample.csv", index=False)
    print("Generated data/samples/valid_sample.csv")

    # Create invalid rows for data/samples/invalid_reviews.csv
    # This must contain specific bad formatting for testing the CSV validation engine
    invalid_rows = []

    # Row 1: Normal row
    normal_row = generate_review_row(20000, 11111, "E-commerce", "Positive")
    invalid_rows.append(normal_row)

    # Row 2: Missing review_text (CRITICAL)
    bad_text_row = generate_review_row(20001, 11112, "Electronics", "Negative")
    bad_text_row["review_text"] = ""
    invalid_rows.append(bad_text_row)

    # Row 3: Missing rating (CRITICAL)
    bad_rating_row = generate_review_row(20002, 11113, "Fashion", "Positive")
    bad_rating_row["rating"] = ""
    invalid_rows.append(bad_rating_row)

    # Row 4: Invalid rating out of range (CRITICAL)
    bad_rating_val_row = generate_review_row(20003, 11114, "Grocery", "Neutral")
    bad_rating_val_row["rating"] = 10 # Ratings are 1-5
    invalid_rows.append(bad_rating_val_row)

    # Row 5: Invalid date format (WARNING)
    bad_date_row = generate_review_row(20004, 11115, "Hospitality", "Mixed")
    bad_date_row["review_date"] = "24-11-2026 14:00:00" # Not %Y-%m-%d %H:%M:%S
    invalid_rows.append(bad_date_row)

    # Row 6: Non-numeric rating (CRITICAL)
    bad_rating_type_row = generate_review_row(20005, 11116, "Food delivery", "Negative")
    bad_rating_type_row["rating"] = "Five Stars"
    invalid_rows.append(bad_rating_type_row)

    # Row 7: Duplicate of Row 1 (WARNING/CRITICAL)
    dup_row = normal_row.copy()
    invalid_rows.append(dup_row)

    # Row 8: Missing optional field product_category (WARNING)
    missing_cat_row = generate_review_row(20006, 11117, "E-commerce", "Positive")
    missing_cat_row["product_category"] = ""
    invalid_rows.append(missing_cat_row)

    df_invalid = pd.DataFrame(invalid_rows)
    df_invalid.to_csv("data/samples/invalid_reviews.csv", index=False)
    print("Generated data/samples/invalid_reviews.csv")

if __name__ == "__main__":
    generate_datasets()
