import pandas as pd
import numpy as np
from src.validation.csv_validator import CSVValidator

print("=== CSV VALIDATOR RUNTIME VERIFICATION ===")

# 1. Valid CSV
df_valid = pd.DataFrame([{
    "review_id": "REV_001", "customer_id": "CUST_1", "review_text": "Great product!",
    "rating": 5, "review_date": "2026-11-24 10:00:00", "product_category": "E-commerce",
    "product_name": "Chair", "source": "Web"
}])
res_v = CSVValidator(df_valid).validate()
print(f"1. Valid CSV -> isValid: {res_v['is_valid']}, Valid Rows: {res_v['valid_rows_count']}")

# 2. Missing Required Column
df_missing_col = df_valid.drop(columns=["review_text"])
res_mc = CSVValidator(df_missing_col).validate()
print(f"2. Missing Required Column -> isValid: {res_mc['is_valid']}, Critical Errors: {len(res_mc['critical_errors'])}")

# 3. Missing Review Text
df_empty_text = df_valid.copy()
df_empty_text.loc[0, "review_text"] = ""
res_et = CSVValidator(df_empty_text).validate()
print(f"3. Missing Review Text -> isValid: {res_et['is_valid']}, Critical Errors: {len(res_et['critical_errors'])}")

# 4. Rating out of bounds (6)
df_bad_rating = df_valid.copy()
df_bad_rating.loc[0, "rating"] = 6
res_br = CSVValidator(df_bad_rating).validate()
print(f"4. Rating out of bounds (6) -> isValid: {res_br['is_valid']}, Critical Errors: {len(res_br['critical_errors'])}")

# 5. Invalid Rating Type (string)
df_str_rating = df_valid.copy()
df_str_rating.loc[0, "rating"] = "Excellent"
res_sr = CSVValidator(df_str_rating).validate()
print(f"5. Invalid Rating Type -> isValid: {res_sr['is_valid']}, Critical Errors: {len(res_sr['critical_errors'])}")

# 6. Invalid Date Format
df_bad_date = df_valid.copy()
df_bad_date.loc[0, "review_date"] = "24/13/2026 99:99"
res_bd = CSVValidator(df_bad_date).validate()
print(f"6. Invalid Date Format -> Warnings: {len(res_bd['warnings'])}, Valid Rows: {res_bd['valid_rows_count']}")

# 7. Duplicate Rows / Review IDs
df_dups = pd.concat([df_valid, df_valid], ignore_index=True)
res_dup = CSVValidator(df_dups).validate()
print(f"7. Duplicate Review IDs -> Warnings: {len(res_dup['warnings'])}")

# 8. Unknown Columns
df_unknown = df_valid.copy()
df_unknown["extra_column_xyz"] = "unexpected"
res_un = CSVValidator(df_unknown).validate()
print(f"8. Unknown Columns -> Warnings: {len(res_un['warnings'])}, Valid Rows: {res_un['valid_rows_count']}")

# 9. Empty CSV
df_empty = pd.DataFrame(columns=["review_id", "customer_id", "review_text", "rating", "review_date", "product_category", "product_name", "source"])
res_emp = CSVValidator(df_empty).validate()
print(f"9. Empty CSV -> isValid: {res_emp['is_valid']}, Total Rows: {res_emp['total_rows']}")
