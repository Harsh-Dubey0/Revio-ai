import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(dataset_path="data/generated/reviews_dataset.csv", output_dir="reports/eda"):
    print(f"Running Exploratory Data Analysis on {dataset_path}...")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)

    eda_results = {}

    # 1. Dataset Shape & Schema
    eda_results["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
    eda_results["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # 2. Missing Value Analysis
    missing_counts = df.isnull().sum().to_dict()
    eda_results["missing_values"] = missing_counts

    # 3. Duplicate Analysis
    duplicate_rows = int(df.duplicated().sum())
    eda_results["duplicate_rows"] = duplicate_rows
    eda_results["duplicate_review_ids"] = int(df.duplicated(subset=["review_id"]).sum())

    # 4. Rating Distribution
    rating_counts = df["rating"].value_counts().sort_index().to_dict()
    eda_results["rating_distribution"] = rating_counts

    # 5. Sentiment Distribution
    sentiment_counts = df["sentiment_label"].value_counts().to_dict()
    eda_results["sentiment_distribution"] = sentiment_counts

    # 6. Category Distribution
    category_counts = df["product_category"].value_counts().to_dict()
    eda_results["category_distribution"] = category_counts

    # 7. Review Length Analysis
    df["review_length"] = df["review_text"].astype(str).apply(len)
    eda_results["review_length_stats"] = {
        "mean": float(df["review_length"].mean()),
        "median": float(df["review_length"].median()),
        "min": int(df["review_length"].min()),
        "max": int(df["review_length"].max()),
        "std": float(df["review_length"].std())
    }

    # Save summary json
    summary_path = os.path.join(output_dir, "eda_summary.json")
    with open(summary_path, "w") as f:
        json.dump(eda_results, f, indent=4)
    print(f"EDA summary saved to {summary_path}")

    # Generate charts
    sns.set_theme(style="whitegrid")

    # Chart 1: Sentiment Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="sentiment_label", palette="viridis", order=["Positive", "Neutral", "Mixed", "Negative"])
    plt.title("Sentiment Distribution of Synthetic Reviews", fontsize=14, fontweight="bold")
    plt.xlabel("Sentiment Label", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentiment_distribution.png"), dpi=300)
    plt.close()

    # Chart 2: Rating Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="rating", palette="magma")
    plt.title("Rating Distribution (1 to 5 Stars)", fontsize=14, fontweight="bold")
    plt.xlabel("Rating", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rating_distribution.png"), dpi=300)
    plt.close()

    # Chart 3: Category Distribution
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=df, y="product_category", palette="crest", order=df["product_category"].value_counts().index)
    plt.title("Review Count by Product Category", fontsize=14, fontweight="bold")
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Product Category", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_distribution.png"), dpi=300)
    plt.close()

    # Chart 4: Review Length Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df["review_length"], bins=50, kde=True, color="purple")
    plt.title("Review Character Length Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Review Length (Characters)", fontsize=12)
    plt.ylabel("Density / Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "review_length_distribution.png"), dpi=300)
    plt.close()

    print("EDA charts successfully generated in reports/eda/")
    return eda_results

if __name__ == "__main__":
    run_eda()
