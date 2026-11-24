import pandas as pd

def format_number(val, decimal_places=2):
    """Format number to max 2 decimal places, removing trailing zeros."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        num = float(val)
        formatted = f"{num:.{decimal_places}f}".rstrip('0').rstrip('.')
        return formatted if formatted != "" else "0"
    except (ValueError, TypeError):
        return str(val)

def format_integer(val):
    """Format integer with comma separators."""
    if val is None or pd.isna(val):
        return "0"
    try:
        return f"{int(val):,}"
    except (ValueError, TypeError):
        return str(val)

def format_percentage(val):
    """Format float as percentage (e.g., 0.8765 -> 87.65%)."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        return f"{float(val)*100:.2f}%"
    except (ValueError, TypeError):
        return str(val)

def format_rating(val):
    """Format rating (e.g., 4.25 -> 4.25 / 5.0)."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        return f"{float(val):.2f} / 5.0"
    except (ValueError, TypeError):
        return str(val)
