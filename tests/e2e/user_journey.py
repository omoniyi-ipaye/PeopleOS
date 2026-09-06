"""End-to-end enterprise UI acceptance test for PeopleOS.

Runs against live FastAPI + Next.js processes in CI and exercises the full
user-facing route set after activating the sample dataset. The goal is not only
functional correctness: every route must render its enterprise information
hierarchy without browser/page failures.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


BASE_URL = "http://127.0.0.1:3000"
API_URL = "http://127.0.0.1:8000"
ARTIFACT_DIR = Path("artifacts/e2e")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def wait_http(url: str, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:  # pragma: no cover
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def assert_route(page: Page, path: str, heading: str, screenshot: str) -> None:
    page.goto(f"{BASE_URL}{path}", wait_until="networkidle", timeout=120_000)
    page.get_by_role("heading", name=heading).wait_for(timeout=30_000)
    body = page.locator("body").inner_text()
    assert "Application error" not in body, f"{path}: {body[-2000:]}"
    assert "Internal Server Error" not in body, f"{path}: {body[-2000:]}"
    page.screenshot(path=str(ARTIFACT_DIR / screenshot), full_page=True)


def main() -> None:
    wait_http(f"{API_URL}/api/health")
    wait_http(BASE_URL)

    health = fetch_json(f"{API_URL}/api/health")
    assert health.get("status") in {"healthy", "degraded"}, health

    console_errors: list[str] = []
    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        # 1. Activate the sample dataset through the real Data & Sources UI.
        assert_route(page, "/upload", "Know exactly what data PeopleOS is using", "01-data-sources.png")
        if page.get_by_text("Dataset active", exact=True).count() == 0:
            started = time.monotonic()
            page.get_by_role("button", name="Load sample").click()
            page.get_by_text("Dataset active", exact=True).wait_for(timeout=120_000)
            elapsed = time.monotonic() - started
            (ARTIFACT_DIR / "timings.json").write_text(
                json.dumps({"sample_data_activation_seconds": round(elapsed, 2)}, indent=2)
            )
        page.screenshot(path=str(ARTIFACT_DIR / "02-data-active.png"), full_page=True)

        status = fetch_json(f"{API_URL}/api/upload/status")
        assert status.get("has_data") is True, status
        assert int(status.get("employee_count", 0)) >= 50, status
        assert status.get("active_dataset_id"), status

        # 2. Every major user-facing route must render its enterprise hierarchy.
        routes = [
            ("/", "What deserves your attention?", "03-decision-cockpit.png"),
            ("/workforce-health", "Where is organisational pressure building?", "04-workforce-health.png"),
            ("/employee-experience", "How are people experiencing the organisation?", "05-employee-experience.png"),
            ("/quality-of-hire", "Which hiring inputs are associated with better outcomes?", "06-quality-of-hire.png"),
            ("/retention-forecast", "How is retention likely to evolve?", "07-retention-forecast.png"),
            ("/flight-risk", "Predictive retention signals are not active", "08-retention-signals.png"),
            ("/advisor", "Ask a workforce question and inspect the evidence", "09-people-intelligence.png"),
            ("/search", "Search the evidence in workforce text", "10-research.png"),
            ("/scenario-planner", "Explore workforce decisions before making them", "11-scenario-planner.png"),
            ("/platform", "Can I trust this analysis?", "12-trust-center.png"),
            ("/sessions", "Saved Investigations", "13-saved-investigations.png"),
            ("/settings", "System configuration and capability state", "14-settings.png"),
            ("/design-system", "PeopleOS Enterprise Design System", "15-design-system.png"),
        ]

        for path, heading, screenshot in routes:
            # Capability-dependent routes may render a deliberate alternate heading.
            if path == "/search":
                page.goto(f"{BASE_URL}{path}", wait_until="networkidle", timeout=120_000)
                body = page.locator("body").inner_text()
                assert "Search the evidence in workforce text" in body or "Search is not available for this dataset" in body, body[-2000:]
                page.screenshot(path=str(ARTIFACT_DIR / screenshot), full_page=True)
            elif path == "/flight-risk":
                page.goto(f"{BASE_URL}{path}", wait_until="networkidle", timeout=120_000)
                body = page.locator("body").inner_text()
                assert "Where is predictive retention pressure concentrated?" in body or "Predictive retention signals are not active" in body, body[-2000:]
                page.screenshot(path=str(ARTIFACT_DIR / screenshot), full_page=True)
            else:
                assert_route(page, path, heading, screenshot)

        # 3. Run a governed People Intelligence investigation after the route sweep.
        page.goto(f"{BASE_URL}/advisor", wait_until="networkidle", timeout=120_000)
        textarea = page.get_by_placeholder("Ask about turnover, workforce health, compensation equity or organisation structure…")
        textarea.fill("What are the most important workforce health signals right now?")
        page.get_by_role("button", name="Investigate", exact=True).click()
        page.get_by_text("Evidence ledger", exact=True).wait_for(timeout=180_000)
        page.get_by_text("Agent boundary", exact=True).wait_for(timeout=30_000)
        page.screenshot(path=str(ARTIFACT_DIR / "16-investigation.png"), full_page=True)

        body_text = page.locator("body").inner_text()
        assert "confidence" in body_text.lower()
        assert "read-only aggregate analysis" in body_text.lower()
        assert "Investigation unavailable" not in body_text, body_text[-2000:]

        browser.close()

    assert not page_errors, page_errors
    significant_console_errors = [error for error in console_errors if "favicon" not in error.lower()]
    assert not significant_console_errors, significant_console_errors

    print("ENTERPRISE UI E2E ROUTE AUDIT: PASS")


if __name__ == "__main__":
    main()
