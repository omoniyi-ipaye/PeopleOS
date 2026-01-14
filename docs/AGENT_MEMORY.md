# Agent Memory & Development Guide

**Project:** PeopleOS (Strategic People Analytics Platform)
**Last Updated:** 2026-01-14

## 🧠 Critical Context "What We Often Miss"

This document serves as a persistent memory for the AI agent to avoid repeating past mistakes and to better understand the system's "personality".

### 1. The "Predictive Analytics Not Available" Error
*   **Symptom:** The UI shows "Predictive analytics not available" even when data seems correct.
*   **Root Cause 1 (Data):** The dataset MUST have an `Attrition` column (case-sensitive, though fuzzy matching helps) with *both* 0s and 1s (targets).
*   **Root Cause 2 (Code Crash):** The `MLEngine` training process runs in a `try/except` block in `dependencies.py`. If *any* error occurs during training (like a `KeyError`, `ValueError`, or `ZeroDivisionError`), the system logs the error to `logs/peopleos.log` but **does not crash the app**. It simply sets `is_trained = False`.
    *   **Lesson:** ALWAYS check `logs/peopleos.log` if the ML features are disabled. Do not assume the code is bug-free just because the server is running.
    *   **Specific Incident:** On 2026-01-14, training failed due to a `KeyError: 'warnings'` because a warnings list wasn't initialized before use.

### 2. Data Loading & "Golden Schema"
*   **Strictness:** The `DataLoader` is strict. It maps columns to a "Golden Schema".
*   **Pitfall:** Optional columns like `PerformanceText` or `InterviewScore` are crucial for advanced features (NLP, Quality of Hire). If they are missing or named weirdly, those specific engines will silently disable themselves.
*   **Visualization:** The "World Map" and "Demographics" require specific columns (`Country`, `Location`). `Country` must be normalized (e.g., "United States", not "US" or "USA" mixed) for best visualization.

### 3. Local-First Architecture
*   **State:** The state is held in `api/dependencies.py` (`AppState`).
*   **Persistence:** Data acts as if it's persistent (SQLite `peopleos.db`), but the *trained ML models* are often in-memory. Restarting `uvicorn` might require re-training (which happens automatically on startup if data exists, but takes time).
*   **Logs:** We have removed `uvicorn.log` from git tracking. The main application log is `logs/peopleos.log`.

### 4. UI/Frontend Nuances
*   **Tailwind:** We use Tailwind CSS.
*   **Components:** The UI is modular (`web/components/...`). A common error is "Hydration Mismatch" (Next.js) or "AnimatePresence" syntax errors (missing fragments `<>...</>`).
*   **Graphs:** We use Recharts. It requires data to be shaped exactly right. If a graph is empty, check the API response format.

## 🛠 Troubleshooting Checklist

1.  **Is the ML Engine trained?** Check `http://localhost:8000/api/health` or logs.
2.  **Are columns mapped?** Check `src/data_loader.py` logic vs. your CSV.
3.  **Did the server crash?** Check terminal output.
4.  **Is the UI expecting a different format?** Check `api/schemas/...` vs TypeScript interfaces.

## 📝 Recurring Commandments
1.  **Never Assume Success:** If an API returns 200, check the content. If a script runs, check the output file size.
2.  **Log Everything:** Since this is a local app, detailed logs in `peopleos.log` are the primary debug tool.
3.  **Respect the Schema:** The `Golden Schema` in `data_loader.py` is the single source of truth.
