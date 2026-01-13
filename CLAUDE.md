# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PeopleOS is a **local-first, privacy-preserving Strategic People Analytics platform** for HR leaders. It features a Python FastAPI backend and Next.js React frontend. All data processing happens locally with optional local AI via Ollama.

## Commands

### Backend (Python 3.10+)
```bash
pip install -r requirements.txt              # Install dependencies
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload  # Start API server
pytest tests/                                # Run all tests
pytest tests/ --cov=src --cov-report=html   # Run with coverage
pytest tests/test_ml_engine.py -v           # Run specific test file
```

### Frontend (Node.js 18+)
```bash
cd web
npm install         # Install dependencies
npm run dev         # Development server (http://localhost:3000)
npm run build       # Production build
npm run lint        # ESLint check
```

### Full Stack Development
Run backend and frontend in separate terminals. Frontend proxies `/api/*` to backend via Next.js rewrites.

## Architecture

### Layered Structure

**API Layer** (`api/`):
- `main.py` - FastAPI app entry point
- `dependencies.py` - AppState singleton managing all engines and data
- `routes/` - 15+ route modules (predictions, analytics, survival, compensation, etc.)
- `schemas/` - Pydantic request/response models

**Business Logic Layer** (`src/`):
- **MLEngine** - Attrition prediction with RF/XGBoost/LightGBM, SMOTE, Optuna tuning, SHAP explanations
- **SurvivalEngine** - Kaplan-Meier curves, Cox Proportional Hazards models
- **CompensationEngine** - Pay equity analysis, Gini coefficients
- **SuccessionEngine** - 9-box matrix, readiness scoring
- **FairnessEngine** - EEOC four-fifths rule compliance
- **QualityOfHireEngine** - Source ROI, interview predictor correlations
- **NLPEngine** / **SentimentEngine** - Topic extraction, sentiment analysis
- **VectorEngine** - FAISS semantic search
- **StructuralEngine** - Org structure, span of control
- **TeamDynamicsEngine** - Team health, diversity metrics
- **LLMClient** - Ollama integration

**Data Layer** (`src/`):
- `data_loader.py` - CSV/DB loading
- `database.py` - SQLite persistence (`data/peopleos.db`)
- `preprocessor.py` - Feature engineering

**Frontend** (`web/`):
- Next.js 14 with App Router
- TanStack Query for data fetching
- Zustand for state management
- Tailwind CSS with custom design tokens
- Components in `components/ui/` (Shadcn-style)

### Key Patterns

**AppState Singleton**: Central state in `api/dependencies.py` holds loaded data and trained models. Call `get_app_state()` to access. `state.load_data(filepath)` initializes all engines.

**Feature Enablement**: Features auto-enable based on available columns:
- `predictive`: Requires `Attrition` column
- `nlp`: Requires `PerformanceText` column
- `llm`: Requires Ollama running

**Engine Pattern**: Each analytics domain has a dedicated engine. Engines are initialized once via AppState and reused across requests.

## Configuration

`config.yaml` controls thresholds and behavior:
- ML risk thresholds: high > 0.75, medium > 0.50
- EEOC four-fifths threshold: 0.8
- Min data rows: 50 (100+ recommended for reliable ML)
- Ollama model: configurable (default: gemma3:4b)

## Data Schema

**Required columns**: EmployeeID, Dept, Tenure, Salary, LastRating, Age

**Optional columns enabling features**:
- `Attrition` - Enables retention forecasting
- `PerformanceText` - Enables NLP sentiment analysis
- `HireSource`, `InterviewScores` - Enables quality of hire analytics
- `Gender` - Enables fairness/equity analysis
- `ManagerID` - Enables org structure analysis

## Testing

Tests are in `tests/`. Run specific test files with `pytest tests/test_<name>.py -v`. Major test files:
- `test_ml_engine.py` - ML pipeline tests
- `test_integration.py` - End-to-end workflows
- `test_compensation_engine.py`, `test_fairness_engine.py` - Compliance tests
