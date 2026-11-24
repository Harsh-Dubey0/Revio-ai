from __future__ import annotations

from dataclasses import dataclass, field
import io
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


COLUMN_ALIASES: Dict[str, List[str]] = {
    "review_id": ["review_id", "id", "reviewid", "review _id", "order_id"],
    "review_text": ["review_text", "review", "text", "comment", "comments", "feedback", "body"],
    "rating": ["rating", "stars", "star_rating", "score", "rate"],
    "product_category": ["product_category", "category", "cat", "department", "vertical", "product_type"],
    "created_at": ["created_at", "date", "review_date", "timestamp", "created"],
}

REQUIRED_CANONICAL_COLS = ["review_text", "rating"]


@dataclass
class ValidationReport:
    is_valid: bool
    cleaned_df: pd.DataFrame
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "stats": dict(self.stats),
            "row_count": int(len(self.cleaned_df)),
        }


class CSVValidator:
    """Validates, normalizes, and cleans uploaded review CSV datasets before inference."""

    def __init__(
        self,
        max_file_mb: float = 25.0,
        max_rows: int = 50000,
        min_rating: int = 1,
        max_rating: int = 5,
        drop_empty_text: bool = True,
        deduplicate_review_ids: bool = False,
    ) -> None:
        self.max_file_mb = float(max_file_mb)
        self.max_rows = int(max_rows)
        self.min_rating = int(min_rating)
        self.max_rating = int(max_rating)
        self.drop_empty_text = bool(drop_empty_text)
        self.deduplicate_review_ids = bool(deduplicate_review_ids)

    def _read_csv_source(
        self,
        source: Union[str, Path, io.BytesIO, io.StringIO, Any],
        errors: List[str],
        warnings: List[str],
    ) -> Optional[pd.DataFrame]:
        if isinstance(source, pd.DataFrame):
            return source.copy()

        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                errors.append(f"CSV file not found at path: {p}")
                return None
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > self.max_file_mb:
                errors.append(f"File size ({size_mb:.2f} MB) exceeds limit ({self.max_file_mb:.2f} MB).")
                return None
        elif hasattr(source, "size") and getattr(source, "size") is not None:
            size_mb = float(getattr(source, "size")) / (1024 * 1024)
            if size_mb > self.max_file_mb:
                errors.append(f"Uploaded file size ({size_mb:.2f} MB) exceeds limit ({self.max_file_mb:.2f} MB).")
                return None

        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        for enc in encodings:
            try:
                if hasattr(source, "seek"):
                    source.seek(0)
                df = pd.read_csv(source, encoding=enc)
                if enc != "utf-8":
                    warnings.append(f"CSV decoded using fallback encoding '{enc}'.")
                return df
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                errors.append("Uploaded CSV file is completely empty (no header or data).")
                return None
            except Exception as exc:
                errors.append(f"Failed to parse CSV file: {exc}")
                return None

        errors.append("Unable to decode CSV file with supported encodings (UTF-8, CP1252, Latin-1).")
        return None

    def _normalize_columns(self, df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        rename_map: Dict[str, str] = {}
        lower_to_orig = {str(col).strip().lower().replace("-", "_"): col for col in df.columns}
        already_mapped_targets = set()

        for canonical, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                norm_alias = alias.strip().lower().replace("-", "_")
                if norm_alias in lower_to_orig and canonical not in already_mapped_targets:
                    orig_col = lower_to_orig[norm_alias]
                    if orig_col != canonical:
                        rename_map[orig_col] = canonical
                        warnings.append(f"Mapped column '{orig_col}' -> '{canonical}'.")
                    already_mapped_targets.add(canonical)
                    break

        out = df.rename(columns=rename_map).copy()
        out.columns = [str(c).strip() for c in out.columns]
        return out

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationReport:
        errors: List[str] = []
        warnings: List[str] = []

        stats: Dict[str, Any] = {
            "raw_rows": int(len(df)) if df is not None else 0,
            "raw_columns": int(len(df.columns)) if df is not None else 0,
            "dropped_empty_text": 0,
            "dropped_invalid_rating": 0,
            "duplicate_review_ids": 0,
            "auto_generated_ids": 0,
            "defaulted_categories": 0,
        }

        if df is None or df.empty:
            errors.append("CSV contains zero rows.")
            return ValidationReport(False, pd.DataFrame(), errors, warnings, stats)

        if len(df) > self.max_rows:
            errors.append(f"Row count ({len(df):,}) exceeds maximum allowed ({self.max_rows:,}).")
            return ValidationReport(False, pd.DataFrame(), errors, warnings, stats)

        norm_df = self._normalize_columns(df, warnings)

        missing_required = [c for c in REQUIRED_CANONICAL_COLS if c not in norm_df.columns]
        if missing_required:
            errors.append(
                f"Missing required column(s): {', '.join(missing_required)}. "
                f"Available columns: {list(norm_df.columns)}"
            )
            return ValidationReport(False, pd.DataFrame(), errors, warnings, stats)

        working = norm_df.copy()

        # 1. Clean review_text
        working["review_text"] = working["review_text"].fillna("").astype(str).str.strip()
        empty_mask = working["review_text"].eq("") | working["review_text"].str.lower().isin({"nan", "none", "null"})
        empty_cnt = int(empty_mask.sum())
        stats["dropped_empty_text"] = empty_cnt

        if empty_cnt > 0:
            if self.drop_empty_text:
                warnings.append(f"Removed {empty_cnt:,} row(s) with empty/missing review_text.")
                working = working[~empty_mask].copy()
            else:
                warnings.append(f"Found {empty_cnt:,} row(s) with empty review_text; keeping with neutral fallback.")
                working.loc[empty_mask, "review_text"] = ""

        if working.empty:
            errors.append("No valid review rows remain after removing empty review_text entries.")
            return ValidationReport(False, pd.DataFrame(), errors, warnings, stats)

        # 2. Validate & coerce rating
        parsed_rating = pd.to_numeric(working["rating"], errors="coerce")
        valid_rating_mask = (
            parsed_rating.notna()
            & (parsed_rating >= self.min_rating)
            & (parsed_rating <= self.max_rating)
            & ((parsed_rating % 1) == 0)
        )
        invalid_rating_cnt = int((~valid_rating_mask).sum())
        stats["dropped_invalid_rating"] = invalid_rating_cnt

        if invalid_rating_cnt > 0:
            warnings.append(
                f"Dropped {invalid_rating_cnt:,} row(s) with invalid ratings "
                f"(expected whole numbers between {self.min_rating} and {self.max_rating})."
            )
            working = working[valid_rating_mask].copy()
            parsed_rating = parsed_rating[valid_rating_mask]

        if working.empty:
            errors.append(
                f"No valid rows remain after rating validation ({self.min_rating}-{self.max_rating})."
            )
            return ValidationReport(False, pd.DataFrame(), errors, warnings, stats)

        working["rating"] = parsed_rating.astype(int)

        # 3. Ensure review_id
        if "review_id" not in working.columns:
            working["review_id"] = [f"REV-{i + 1:05d}" for i in range(len(working))]
            stats["auto_generated_ids"] = int(len(working))
            warnings.append("Column 'review_id' not found; auto-generated sequential IDs.")
        else:
            missing_id_mask = (
                working["review_id"].isna()
                | working["review_id"].astype(str).str.strip().isin({"", "nan", "None", "null"})
            )
            gen_cnt = int(missing_id_mask.sum())
            if gen_cnt > 0:
                gen_ids = [f"REV-AUTO-{idx + 1}" for idx, is_miss in enumerate(missing_id_mask) if is_miss]
                working.loc[missing_id_mask, "review_id"] = gen_ids
                stats["auto_generated_ids"] = gen_cnt
                warnings.append(f"Auto-generated {gen_cnt:,} missing review_id value(s).")
            working["review_id"] = working["review_id"].astype(str).str.strip()

        dup_cnt = int(working["review_id"].duplicated().sum())
        stats["duplicate_review_ids"] = dup_cnt
        if dup_cnt > 0:
            if self.deduplicate_review_ids:
                working = working.drop_duplicates(subset=["review_id"], keep="first").copy()
                warnings.append(f"Removed {dup_cnt:,} duplicate review_id row(s).")
            else:
                warnings.append(f"Detected {dup_cnt:,} duplicate review_id value(s) in batch.")

        # 4. Ensure product_category
        if "product_category" not in working.columns:
            working["product_category"] = "General"
            stats["defaulted_categories"] = int(len(working))
            warnings.append("Column 'product_category' not found; defaulted to 'General'.")
        else:
            cat_series = working["product_category"].fillna("").astype(str).str.strip()
            blank_cat_mask = cat_series.eq("") | cat_series.str.lower().isin({"nan", "none", "null"})
            def_cnt = int(blank_cat_mask.sum())
            stats["defaulted_categories"] = def_cnt
            if def_cnt > 0:
                cat_series[blank_cat_mask] = "General"
                warnings.append(f"Defaulted {def_cnt:,} blank category value(s) to 'General'.")
            working["product_category"] = cat_series

        # 5. Optional created_at normalization
        if "created_at" in working.columns:
            parsed_dt = pd.to_datetime(working["created_at"], errors="coerce", utc=False)
            bad_dt = int(parsed_dt.isna().sum() - working["created_at"].isna().sum())
            if bad_dt > 0:
                warnings.append(f"{bad_dt:,} 'created_at' value(s) could not be parsed as dates.")
            working["created_at"] = parsed_dt

        working = working.reset_index(drop=True)
        stats["valid_rows"] = int(len(working))
        stats["rating_breakdown"] = working["rating"].value_counts().sort_index().to_dict()
        stats["category_breakdown"] = working["product_category"].value_counts().head(10).to_dict()

        return ValidationReport(
            is_valid=True,
            cleaned_df=working,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    def validate_csv(
        self,
        source: Union[str, Path, io.BytesIO, io.StringIO, Any],
    ) -> ValidationReport:
        errors: List[str] = []
        warnings: List[str] = []

        raw_df = self._read_csv_source(source, errors, warnings)
        if raw_df is None:
            return ValidationReport(
                is_valid=False,
                cleaned_df=pd.DataFrame(),
                errors=errors,
                warnings=warnings,
                stats={"raw_rows": 0},
            )

        report = self.validate_dataframe(raw_df)
        report.errors = errors + report.errors
        report.warnings = warnings + report.warnings
        return report


ReviewCSVValidator = CSVValidator


def validate_reviews_csv(
    source: Union[str, Path, io.BytesIO, io.StringIO, Any],
    **kwargs: Any,
) -> ValidationReport:
    """Convenience function for Streamlit upload pages and batch scripts."""
    validator = CSVValidator(**kwargs)
    return validator.validate_csv(source)