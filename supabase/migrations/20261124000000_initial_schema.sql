-- Revio AI Intelligence Engine Database Migration Schema
-- Supabase PostgreSQL migration file

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Users table (or User accounts / profiles)
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(50) NOT NULL UNIQUE,
    architecture VARCHAR(100) NOT NULL,
    accuracy DOUBLE PRECISION,
    macro_f1 DOUBLE PRECISION,
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'inactive' CHECK (status IN ('active', 'inactive'))
);

-- 3. Upload batches table
CREATE TABLE IF NOT EXISTS upload_batches (
    batch_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    row_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    model_version VARCHAR(50) DEFAULT 'v1.0.0'
);

-- 4. Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    review_id VARCHAR(50) PRIMARY KEY, -- or generated UUID, but fits input review_id
    batch_id UUID REFERENCES upload_batches(batch_id) ON DELETE CASCADE,
    customer_id VARCHAR(50),
    review_text TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review_date TIMESTAMP WITH TIME ZONE,
    product_category VARCHAR(100),
    product_name VARCHAR(255),
    source VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    review_id VARCHAR(50) REFERENCES reviews(review_id) ON DELETE CASCADE UNIQUE,
    sentiment VARCHAR(20) NOT NULL CHECK (sentiment IN ('Positive', 'Neutral', 'Negative', 'Mixed')),
    confidence DOUBLE PRECISION NOT NULL,
    aspect_labels TEXT, -- Semicolon-separated list or JSON array
    priority VARCHAR(20) NOT NULL CHECK (priority IN ('High', 'Medium', 'Low')),
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. AI Recommendations table
CREATE TABLE IF NOT EXISTS ai_recommendations (
    recommendation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_id UUID REFERENCES upload_batches(batch_id) ON DELETE CASCADE,
    executive_summary TEXT,
    complaints_summary TEXT,
    positives_summary TEXT,
    action_plan JSONB, -- list of action items / suggestions
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. Processing logs table
CREATE TABLE IF NOT EXISTS processing_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_id UUID REFERENCES upload_batches(batch_id) ON DELETE CASCADE,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- CREATE INDEXES for performance tuning
CREATE INDEX IF NOT EXISTS idx_reviews_batch_id ON reviews(batch_id);
CREATE INDEX IF NOT EXISTS idx_reviews_product_category ON reviews(product_category);
CREATE INDEX IF NOT EXISTS idx_predictions_sentiment ON predictions(sentiment);
CREATE INDEX IF NOT EXISTS idx_predictions_priority ON predictions(priority);
CREATE INDEX IF NOT EXISTS idx_predictions_review_id ON predictions(review_id);
CREATE INDEX IF NOT EXISTS idx_ai_recs_batch_id ON ai_recommendations(batch_id);
CREATE INDEX IF NOT EXISTS idx_logs_batch_id ON processing_logs(batch_id);
