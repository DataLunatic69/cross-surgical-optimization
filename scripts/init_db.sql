-- Initialize database schemas for Surgical Optimizer

-- Create schemas
CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS hospital;
CREATE SCHEMA IF NOT EXISTS training;
CREATE SCHEMA IF NOT EXISTS prediction;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Add extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Grant permissions
GRANT ALL ON SCHEMA auth TO surgical_admin;
GRANT ALL ON SCHEMA hospital TO surgical_admin;
GRANT ALL ON SCHEMA training TO surgical_admin;
GRANT ALL ON SCHEMA prediction TO surgical_admin;
GRANT ALL ON SCHEMA analytics TO surgical_admin;

-- Create indexes for better performance (will be added after tables are created)
-- These are placeholders that will be executed after SQLAlchemy creates tables

COMMENT ON SCHEMA auth IS 'Authentication and authorization related tables';
COMMENT ON SCHEMA hospital IS 'Hospital management and configuration tables';
COMMENT ON SCHEMA training IS 'Federated learning training session tables';
COMMENT ON SCHEMA prediction IS 'Surgical prediction and outcome tables';
COMMENT ON SCHEMA analytics IS 'Analytics and aggregated metrics tables';