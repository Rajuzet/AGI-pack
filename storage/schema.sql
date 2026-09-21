-- =============================================================================
-- AGI Autonomous Agent & Model Adaptation Engine - Production Relational Schema
-- Database: agi_memory (UTF-8 utf8mb4)
-- Compatible with MySQL Workbench 8.0+ and Google Cloud SQL for MySQL
-- =============================================================================

CREATE DATABASE IF NOT EXISTS agi_memory
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE agi_memory;

-- -----------------------------------------------------------------------------
-- 1. Table: research_papers
-- Primary key ID, title, clean content/abstract, source, authors (JSON),
-- categories (JSON), PDF URL, and ingestion timestamp. Indexes on source and timestamp.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_papers (
    id VARCHAR(255) NOT NULL,
    title VARCHAR(512) NOT NULL,
    abstract LONGTEXT NOT NULL,
    source VARCHAR(64) NOT NULL,
    authors JSON,
    categories JSON,
    pdf_url VARCHAR(1024),
    ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_papers_source (source),
    INDEX idx_papers_ingested_at (ingested_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------------------------
-- 2. Table: training_runs
-- Run ID, base model ID, quantization type, dataset sample count, final training loss,
-- duration in seconds, GCS checkpoint URI, and execution timestamp.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS training_runs (
    run_id VARCHAR(128) NOT NULL,
    base_model VARCHAR(255) NOT NULL,
    quantization VARCHAR(64) NOT NULL DEFAULT '4-bit NF4',
    dataset_sample_count INT NOT NULL DEFAULT 0,
    training_loss FLOAT,
    duration_seconds FLOAT,
    gcs_checkpoint_uri VARCHAR(1024),
    executed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id),
    INDEX idx_training_executed_at (executed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------------------------
-- 3. Table: agent_sessions
-- Session ID, input objective, status (success/failed), total steps, retry count,
-- execution latency, and session timestamp.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id VARCHAR(128) NOT NULL,
    input_objective TEXT NOT NULL,
    status ENUM('success', 'failed') NOT NULL DEFAULT 'success',
    total_steps INT NOT NULL DEFAULT 0,
    retry_count INT NOT NULL DEFAULT 0,
    execution_latency FLOAT NOT NULL DEFAULT 0.0,
    session_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id),
    INDEX idx_sessions_status (status),
    INDEX idx_sessions_timestamp (session_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------------------------
-- 4. Table: agent_steps
-- Auto-increment ID, foreign key referencing agent_sessions(session_id) with
-- cascading deletes, step index, thought trace, action name, action input (JSON),
-- observation output, and step latency.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_steps (
    id BIGINT AUTO_INCREMENT NOT NULL,
    session_id VARCHAR(128) NOT NULL,
    step_index INT NOT NULL,
    thought_trace TEXT,
    action_name VARCHAR(128),
    action_input JSON,
    observation_output LONGTEXT,
    step_latency FLOAT NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_agent_steps_session
        FOREIGN KEY (session_id)
        REFERENCES agent_sessions(session_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    INDEX idx_steps_session_index (session_id, step_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
