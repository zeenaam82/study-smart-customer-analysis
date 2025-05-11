-- init.sql
CREATE ROLE myuser WITH LOGIN PASSWORD '8254';
ALTER ROLE myuser CREATEDB;

-- 데이터베이스가 없는 경우 생성
DO $$  
BEGIN  
    IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'customer_analysis') THEN  
        CREATE DATABASE customer_analysis OWNER myuser;  
    END IF;  
END $$;

GRANT ALL PRIVILEGES ON DATABASE customer_analysis TO myuser;

\c customer_analysis

-- 테이블이 없는 경우 생성
CREATE TABLE IF NOT EXISTS customer_data (
    id SERIAL PRIMARY KEY,
    review TEXT NOT NULL,
    label INTEGER NOT NULL
);