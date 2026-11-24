# Revio AI Intelligence Engine — Database Schema

Revio AI uses PostgreSQL (hosted on Supabase) with schema migrations managed under `supabase/migrations/`. Below is the relational schema specification.

## Tables

### 1. `upload_batches`
Tracks user upload sessions and ingestion status.
- `batch_id`: UUID (Primary Key)
- `filename`: VARCHAR(255)
- `uploaded_at`: TIMESTAMP WITH TIME ZONE
- `row_count`: INTEGER
- `status`: VARCHAR(20) (`pending`, `processing`, `incomplete`, `completed`, `failed`)
- `model_version`: VARCHAR(50)

### 2. `reviews`
Stores customer review records linked to an upload batch.
- `review_id`: VARCHAR(50) (Primary Key)
- `batch_id`: UUID (Foreign Key references `upload_batches(batch_id)` ON DELETE CASCADE)
- `customer_id`: VARCHAR(50)
- `review_text`: TEXT
- `rating`: INTEGER (CHECK `1 <= rating <= 5`)
- `review_date`: TIMESTAMP WITH TIME ZONE
- `product_category`: VARCHAR(100)
- `product_name`: VARCHAR(255)
- `source`: VARCHAR(100)
- `created_at`: TIMESTAMP WITH TIME ZONE

### 3. `predictions`
Stores ML/NLP model inference outputs per review.
- `prediction_id`: UUID (Primary Key)
- `review_id`: VARCHAR(50) (Foreign Key references `reviews(review_id)` ON DELETE CASCADE, UNIQUE)
- `sentiment`: VARCHAR(20) (`Positive`, `Neutral`, `Negative`, `Mixed`)
- `emotion`: VARCHAR(50)
- `confidence`: DOUBLE PRECISION
- `aspect_labels`: TEXT (Semicolon-separated aspects)
- `priority`: VARCHAR(20) (`High`, `Medium`, `Low`)
- `predicted_at`: TIMESTAMP WITH TIME ZONE

### 4. `ai_recommendations`
Stores generated executive insights and action plans per batch.
- `recommendation_id`: UUID (Primary Key)
- `batch_id`: UUID (Foreign Key references `upload_batches(batch_id)` ON DELETE CASCADE)
- `executive_summary`: TEXT
- `complaints_summary`: TEXT
- `positives_summary`: TEXT
- `action_plan`: JSONB
- `generated_at`: TIMESTAMP WITH TIME ZONE

### 5. `processing_logs` & `model_versions`
Audit logging and model registry tables.
- `processing_logs`: `log_id`, `batch_id`, `stage`, `level`, `message`, `created_at`
- `model_versions`: `version_id`, `model_name`, `version_tag`, `metrics`, `artifact_path`, `created_at`

## Indexes & Integrity Notes
- Index `reviews(batch_id)` and `predictions(review_id)` for fast batch verification, analytics aggregations, and cleanup.
- Use `ON DELETE CASCADE` (or transactional child-first deletion) so deleting a batch cleanly removes its reviews, predictions, and recommendations.