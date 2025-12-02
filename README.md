# Arcadia

# Multifactor Quant Stock Screener App

A full-stack application for screening, ranking, and backtesting stocks using multiple factors (fundamental, technical, sentiment). Built with React (frontend), FastAPI (backend), PostgreSQL (database), and Yahoo Finance integration.

## Features
- Stock universe selection (S&P 500, NASDAQ, HKEX, SGX, custom)
- Factor scoring, normalization, weighting, and ranking
- Backtesting module with rebalancing and performance metrics
- Interactive dashboard: ranking table, factor breakdown, charts
- Export results to CSV/XLSX/PDF

## Tech Stack
- Frontend: React, Tailwind CSS, shadcn/ui
- Backend: Python, FastAPI, Pandas, NumPy, scikit-learn, yfinance
- Database: PostgreSQL

## Setup Instructions

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. Database
- Create PostgreSQL database and run `database/schema.sql` to initialize tables.

### 3. Frontend
```bash
npm install
npm run dev
```

### 4. Configuration
- Edit `config/factors.yaml` to customize factor definitions and weights.

### 5. Example Universe
- Add universe files (e.g., S&P 500 tickers) in `config/`.

## Documentation
- See Jupyter notebooks in `notebooks/` for factor scoring and backtesting walkthroughs.

## Extensibility
- Modular backend for new factors, universes, and data sources
- Easily add new visualizations and export formats

---
For questions or contributions, see LICENSE and contact the maintainer.


### Troubleshooting

This section has moved here: [https://vitejs.dev/guide/troubleshooting.html](https://vitejs.dev/guide/troubleshooting.html)
