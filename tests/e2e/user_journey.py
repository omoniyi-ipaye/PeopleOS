"""End-to-end user acceptance test for PeopleOS.

Runs against live FastAPI + Next.js processes in CI and exercises the primary product journey:
Data & Sources -> Decision Cockpit -> Trust Center -> People Intelligence.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


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

        # 1. Data onboarding and durable dataset activation.
        page.goto(f"{BASE_URL}/upload", wait_until="networkidle", timeout=120_000)
        page.get_by_role("heading", name="Make the data state obvious.").wait_for(timeout=30_000)
        page.screenshot(path=str(ARTIFACT_DIR / "01-data-sources.png"), full_page=True)

        if page.get_by_text("Dataset active", exact=True).count() == 0:
            started = time.monotonic()
            page.get_by_role("button", name="Load sample dataset").click()
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

        # 2. Decision Cockpit must present a decision-oriented entry point.
        page.goto(BASE_URL, wait_until="networkidle", timeout=120_000)
        page.get_by_role("heading", name="What deserves your attention?").wait_for(timeout=30_000)
        page.get_by_text("Priority briefing", exact=True).wait_for(timeout=30_000)
        page.get_by_role("link", name="Ask PeopleOS").wait_for(timeout=30_000)
        page.screenshot(path=str(ARTIFACT_DIR / "03-decision-cockpit.png"), full_page=True)

        # 3. Trust Center exposes lifecycle and governance in user language.
        page.goto(f"{BASE_URL}/platform", wait_until="networkidle", timeout=120_000)
        page.get_by_role("heading", name="Can I trust this analysis?").wait_for(timeout=30_000)
        page.get_by_text("Current trust verdict", exact=True).wait_for(timeout=30_000)
        page.screenshot(path=str(ARTIFACT_DIR / "04-trust-center.png"), full_page=True)

        # 4. Governed People Intelligence investigation.
        page.goto(f"{BASE_URL}/advisor", wait_until="networkidle", timeout=120_000)
        page.get_by_role("heading", name="People Intelligence Agent").wait_for(timeout=30_000)
        textarea = page.get_by_placeholder("Ask about turnover, workforce health, compensation equity, manager structure…")
        textarea.fill("What are the most important workforce health signals right now?")
        page.get_by_role("button", name="Investigate").click()

        page.get_by_text("Evidence ledger", exact=True).wait_for(timeout=180_000)
        page.get_by_text("Agent boundary", exact=True).wait_for(timeout=30_000)
        page.screenshot(path=str(ARTIFACT_DIR / "05-investigation.png"), full_page=True)

        body_text = page.locator("body").inner_text()
        assert "confidence" in body_text.lower()
        assert "Allowlisted tools" in body_text
        assert "Aggregate evidence only" in body_text
        assert "Investigation unavailable" not in body_text, body_text[-2000:]

        browser.close()

    assert not page_errors, page_errors
    significant_console_errors = [error for error in console_errors if "favicon" not in error.lower()]
    assert not significant_console_errors, significant_console_errors

    print("E2E USER JOURNEY: PASS")


if __name__ == "__main__":
    main()
