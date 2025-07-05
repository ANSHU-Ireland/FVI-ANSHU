-- Database initialization script for FVI Analytics
-- This script creates the database and initial tables

-- Create database (if using PostgreSQL)
-- CREATE DATABASE fvi_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_data_sources_name ON data_sources(name);
CREATE INDEX IF NOT EXISTS idx_metric_definitions_slug ON metric_definitions(metric_slug);
CREATE INDEX IF NOT EXISTS idx_metric_definitions_sheet ON metric_definitions(sheet_name);
CREATE INDEX IF NOT EXISTS idx_metric_values_metric_id ON metric_values(metric_definition_id);
CREATE INDEX IF NOT EXISTS idx_metric_values_entity ON metric_values(sub_industry, country);
CREATE INDEX IF NOT EXISTS idx_metric_values_time ON metric_values(year, quarter, month);
CREATE INDEX IF NOT EXISTS idx_composite_scores_entity ON composite_scores(sub_industry, country);
CREATE INDEX IF NOT EXISTS idx_composite_scores_time ON composite_scores(year, horizon);
CREATE INDEX IF NOT EXISTS idx_model_runs_name ON model_runs(model_name);
CREATE INDEX IF NOT EXISTS idx_model_runs_deployed ON model_runs(is_deployed);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);

-- Create full-text search indexes
CREATE INDEX IF NOT EXISTS idx_metric_definitions_details_fts ON metric_definitions USING gin(to_tsvector('english', details));
CREATE INDEX IF NOT EXISTS idx_data_sources_description_fts ON data_sources USING gin(to_tsvector('english', description));

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_metric_values_entity_metric_time ON metric_values(sub_industry, metric_definition_id, year);
CREATE INDEX IF NOT EXISTS idx_composite_scores_entity_horizon_time ON composite_scores(sub_industry, horizon, year);

-- Insert initial data sources
INSERT INTO data_sources (name, description, api_available, created_at, updated_at) 
VALUES 
    ('IEA', 'International Energy Agency', true, NOW(), NOW()),
    ('EIA', 'U.S. Energy Information Administration', true, NOW(), NOW()),
    ('World Bank', 'World Bank Open Data', true, NOW(), NOW()),
    ('Quandl', 'Quandl Financial Data', true, NOW(), NOW()),
    ('Yahoo Finance', 'Yahoo Finance API', true, NOW(), NOW()),
    ('Alpha Vantage', 'Alpha Vantage Financial Data', true, NOW(), NOW()),
    ('USGS', 'U.S. Geological Survey', false, NOW(), NOW()),
    ('OECD', 'Organisation for Economic Co-operation and Development', false, NOW(), NOW()),
    ('IMF', 'International Monetary Fund', false, NOW(), NOW()),
    ('BP Statistical Review', 'BP Statistical Review of World Energy', false, NOW(), NOW())
ON CONFLICT (name) DO NOTHING;

-- Insert sample metric definitions for coal industry
INSERT INTO metric_definitions (
    sheet_name, title, metric_slug, details, 
    structured_availability, country_level_availability, 
    sub_industry_availability, volume_of_data, 
    genai_ml_fillability, readiness_status, 
    created_at, updated_at
) VALUES 
    ('1 Necessity Score (Core)', 'Energy Security Dependence', 'energy_security_dependence', 
     'Measures how dependent a country/region is on coal for energy security', 
     4, 5, 4, 4, 3, 'Complete', NOW(), NOW()),
    ('1 Necessity Score (Core)', 'Industrial Process Necessity', 'industrial_process_necessity', 
     'Measures the necessity of coal for industrial processes like steel production', 
     3, 4, 5, 3, 4, 'Complete', NOW(), NOW()),
    ('2 Resource Extraction & Scarcity Score', 'Reserve to Production Ratio', 'reserve_production_ratio', 
     'Coal reserves divided by annual production (years of supply remaining)', 
     4, 5, 4, 4, 2, 'Complete', NOW(), NOW()),
    ('2 Resource Extraction & Scarcity Score', 'Mining Cost Trend', 'mining_cost_trend', 
     'Trend in coal mining costs over time', 
     3, 4, 4, 3, 4, 'Needs Data', NOW(), NOW()),
    ('3 Artificial Support Score', 'Fossil Fuel Subsidies', 'fossil_fuel_subsidies', 
     'Government subsidies for coal industry (USD per unit)', 
     3, 4, 4, 3, 4, 'Needs Data', NOW(), NOW()),
    ('4 Emissions Score', 'CO2 Emissions Intensity', 'co2_emissions_intensity', 
     'CO2 emissions per unit of energy produced (tonnes CO2/MWh)', 
     5, 5, 5, 5, 2, 'Complete', NOW(), NOW()),
    ('4 Emissions Score', 'Carbon Tax Exposure', 'carbon_tax_exposure', 
     'Potential financial impact of carbon pricing mechanisms', 
     3, 4, 4, 3, 4, 'Draft', NOW(), NOW()),
    ('5 Ecological Score', 'Land Use Impact', 'land_use_impact', 
     'Area of land disturbed per unit of coal extracted', 
     2, 3, 3, 2, 5, 'Draft', NOW(), NOW()),
    ('5 Ecological Score', 'Water Usage Intensity', 'water_usage_intensity', 
     'Water consumption per unit of coal processed', 
     3, 3, 4, 3, 4, 'Needs Data', NOW(), NOW()),
    ('8 Workforce Transition Score', 'Employment Dependency', 'employment_dependency', 
     'Number of jobs dependent on coal industry', 
     3, 4, 4, 3, 3, 'Complete', NOW(), NOW()),
    ('8 Workforce Transition Score', 'Reskilling Feasibility', 'reskilling_feasibility', 
     'Feasibility of retraining coal workers for other industries', 
     2, 3, 3, 2, 5, 'Draft', NOW(), NOW()),
    ('9 Infrastructure Repurposing Score', 'Asset Stranding Risk', 'asset_stranding_risk', 
     'Risk of coal infrastructure becoming stranded assets', 
     3, 4, 4, 3, 4, 'Draft', NOW(), NOW()),
    ('11 Monopoly & Corporate Control Score', 'Market Concentration', 'market_concentration', 
     'Herfindahl-Hirschman Index for coal market concentration', 
     3, 4, 4, 3, 3, 'Needs Data', NOW(), NOW()),
    ('20 Economic Score', 'Price Volatility', 'price_volatility', 
     'Standard deviation of coal prices over time', 
     4, 5, 4, 4, 3, 'Complete', NOW(), NOW()),
    ('20 Economic Score', 'Profitability Trend', 'profitability_trend', 
     'Trend in coal industry profitability margins', 
     3, 4, 4, 3, 4, 'Needs Data', NOW(), NOW()),
    ('24. Technological Disruption Score', 'Renewable Energy Cost Competitiveness', 'renewable_cost_competitiveness', 
     'Cost of renewable energy vs coal (LCOE comparison)', 
     4, 4, 4, 4, 3, 'Complete', NOW(), NOW()),
    ('24. Technological Disruption Score', 'Energy Storage Deployment', 'energy_storage_deployment', 
     'Rate of energy storage technology deployment', 
     3, 4, 3, 3, 4, 'Complete', NOW(), NOW())
ON CONFLICT (metric_slug) DO NOTHING;

-- Update final_data_confidence_score for all metrics
UPDATE metric_definitions 
SET final_data_confidence_score = (
    COALESCE(structured_availability, 0) + 
    COALESCE(country_level_availability, 0) + 
    COALESCE(sub_industry_availability, 0) + 
    COALESCE(volume_of_data, 0) + 
    COALESCE(genai_ml_fillability, 0)
) / 5.0
WHERE final_data_confidence_score IS NULL;

-- Create a view for FVI dashboard
CREATE OR REPLACE VIEW fvi_dashboard AS
SELECT 
    cs.sub_industry,
    cs.country,
    cs.year,
    cs.horizon,
    cs.fvi_score,
    cs.fvi_percentile,
    cs.necessity_score,
    cs.resource_scarcity_score,
    cs.artificial_support_score,
    cs.emissions_score,
    cs.ecological_score,
    cs.workforce_transition_score,
    cs.infrastructure_repurposing_score,
    cs.monopoly_control_score,
    cs.economic_score,
    cs.technological_disruption_score,
    cs.confidence_interval_lower,
    cs.confidence_interval_upper,
    cs.model_version,
    cs.created_at as score_calculated_at
FROM composite_scores cs
WHERE cs.created_at = (
    SELECT MAX(cs2.created_at)
    FROM composite_scores cs2
    WHERE cs2.sub_industry = cs.sub_industry
    AND cs2.country = cs.country
    AND cs2.horizon = cs.horizon
);

-- Create a view for metric summary
CREATE OR REPLACE VIEW metric_summary AS
SELECT 
    md.sheet_name,
    md.title,
    md.metric_slug,
    md.readiness_status,
    md.final_data_confidence_score,
    COUNT(mv.id) as value_count,
    COUNT(DISTINCT mv.sub_industry) as industry_count,
    COUNT(DISTINCT mv.country) as country_count,
    MIN(mv.year) as earliest_year,
    MAX(mv.year) as latest_year,
    AVG(mv.normalized_value) as avg_normalized_value,
    ds.name as data_source_name
FROM metric_definitions md
LEFT JOIN metric_values mv ON md.id = mv.metric_definition_id
LEFT JOIN data_sources ds ON md.data_source_id = ds.id
GROUP BY md.id, md.sheet_name, md.title, md.metric_slug, 
         md.readiness_status, md.final_data_confidence_score, ds.name;

-- Create function to calculate confidence score
CREATE OR REPLACE FUNCTION calculate_confidence_score(
    structured_avail INTEGER,
    country_avail INTEGER,
    sub_industry_avail INTEGER,
    volume_data INTEGER,
    genai_fillability INTEGER,
    longitudinal_avail INTEGER,
    bias_risk INTEGER,
    interdependence INTEGER
) RETURNS DECIMAL(3,2) AS $$
BEGIN
    RETURN (
        COALESCE(structured_avail, 0) + 
        COALESCE(country_avail, 0) + 
        COALESCE(sub_industry_avail, 0) + 
        COALESCE(volume_data, 0) + 
        COALESCE(genai_fillability, 0) + 
        COALESCE(longitudinal_avail, 0) + 
        COALESCE(6 - bias_risk, 0) + 
        COALESCE(6 - interdependence, 0)
    )::DECIMAL / 8.0;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to auto-update confidence scores
CREATE OR REPLACE FUNCTION update_confidence_score()
RETURNS TRIGGER AS $$
BEGIN
    NEW.final_data_confidence_score := calculate_confidence_score(
        NEW.structured_availability,
        NEW.country_level_availability,
        NEW.sub_industry_availability,
        NEW.volume_of_data,
        NEW.genai_ml_fillability,
        NEW.longitudinal_availability,
        NEW.data_verification_bias_risk,
        NEW.interdependence_with_other_metrics
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_confidence_score
    BEFORE INSERT OR UPDATE ON metric_definitions
    FOR EACH ROW
    EXECUTE FUNCTION update_confidence_score();

-- Create function to get latest FVI score
CREATE OR REPLACE FUNCTION get_latest_fvi_score(
    p_sub_industry VARCHAR,
    p_country VARCHAR DEFAULT NULL,
    p_horizon INTEGER DEFAULT 10
) RETURNS TABLE (
    fvi_score DECIMAL,
    fvi_percentile DECIMAL,
    calculated_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cs.fvi_score,
        cs.fvi_percentile,
        cs.created_at
    FROM composite_scores cs
    WHERE cs.sub_industry = p_sub_industry
    AND (p_country IS NULL OR cs.country = p_country)
    AND cs.horizon = p_horizon
    ORDER BY cs.created_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fvi_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fvi_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO fvi_user;

-- Create indexes for the views
CREATE INDEX IF NOT EXISTS idx_composite_scores_latest ON composite_scores(sub_industry, country, horizon, created_at DESC);

-- Add comments for documentation
COMMENT ON TABLE data_sources IS 'Registry of external data sources used in FVI calculations';
COMMENT ON TABLE metric_definitions IS 'Definitions of individual metrics that compose the FVI score';
COMMENT ON TABLE metric_values IS 'Actual values for metrics across different entities and time periods';
COMMENT ON TABLE composite_scores IS 'Calculated FVI scores and component scores';
COMMENT ON TABLE model_runs IS 'Record of machine learning model training runs';
COMMENT ON TABLE chat_sessions IS 'Chat sessions for the FVI analyst tool';
COMMENT ON TABLE chat_messages IS 'Individual messages within chat sessions';

COMMENT ON VIEW fvi_dashboard IS 'Latest FVI scores for dashboard display';
COMMENT ON VIEW metric_summary IS 'Summary statistics for each metric definition';

-- Final message
SELECT 'FVI Analytics database initialized successfully' as status;
