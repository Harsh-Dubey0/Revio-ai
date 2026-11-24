import pytest
import pandas as pd
from src.validation.csv_validator import CSVValidator

def test_validator_with_valid_data():
    valid_data = pd.DataFrame([
        {
            "review_id": "REV_000001",
            "customer_id": "CUST_00001",
            "review_text": "Excellent and fast service! Loved the meal.",
            "rating": 5,
            "review_date": "2026-11-24 14:00:00",
            "product_category": "Food delivery",
            "product_name": "Sushi combo",
            "source": "UberEats"
        }
    ])
    validator = CSVValidator(valid_data)
    result = validator.validate()

    assert result["is_valid"] is True
    assert result["valid_rows_count"] == 1
    assert len(result["critical_errors"]) == 0

def test_validator_with_invalid_data():
    invalid_data = pd.DataFrame([
        {
            "review_id": "REV_000002",
            "customer_id": "CUST_00002",
            "review_text": "", # Missing review text (CRITICAL)
            "rating": "Five", # Invalid rating (CRITICAL)
            "review_date": "24/11/2026",
            "product_category": "Grocery",
            "product_name": "Organic Apples",
            "source": "GroceryGo"
        }
    ])
    validator = CSVValidator(invalid_data)
    result = validator.validate()

    assert result["is_valid"] is False
    assert result["valid_rows_count"] == 0
    assert len(result["critical_errors"]) > 0
