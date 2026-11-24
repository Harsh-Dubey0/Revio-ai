# Revio AI Intelligence Engine — Testing Suite

Revio AI uses `pytest` for automated system testing.

## Running Tests
Run all tests from the project root:
```bash
pytest
```

## Test Coverage
- `tests/test_validator.py`: Verifies CSV schema, critical error detection (missing ratings, empty text), and warning generation.
- `tests/test_predictor.py`: Verifies inference pipeline model loading, sentiment prediction logic, aspect extraction, and priority classification rules.
- System Integration: Manual testing of the Streamlit dashboard verifies data flow from CSV upload through to the dashboard analytics and AI recommendation persistence.
