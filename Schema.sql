-- PostgreSQL schema for stocks and factors
CREATE TABLE IF NOT EXISTS stocks (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    region VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS factors (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    date DATE NOT NULL,
    factor_name VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS rankings (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    date DATE NOT NULL,
    composite_score FLOAT NOT NULL,
    rank INTEGER
);
