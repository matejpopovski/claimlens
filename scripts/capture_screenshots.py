"""Screenshot the running service through a real browser, one image per outcome.

Start the service first. Each shot is a claim typed into the page and answered by the
live API, and the script fails if the page lacks what the screenshot is meant to prove:
marked evidence sentences, and a stated reason on an abstention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


def run_claim(page: Page, claim: str, timeout_ms: int) -> str:
    page.fill("#claim", claim)
    with page.expect_response("**/api/verify", timeout=timeout_ms) as response:
        page.click("#submit")
    body = response.value.json()
    page.wait_for_function(
        "claim => !document.getElementById('finding').hidden && "
        "document.getElementById('submit').disabled === false",
        arg=claim,
    )
    return body["outcome"]


def check_page(page: Page, outcome: str) -> None:
    if page.locator(f"#finding.{outcome}").count() != 1:
        raise SystemExit(f"finding block does not show {outcome}")
    if page.locator("#evidence mark").count() == 0:
        raise SystemExit("no evidence sentence is marked")
    if outcome == "ABSTAINED" and page.locator(".reasons li").count() == 0:
        raise SystemExit("an abstention appeared without a reason")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:45411")
    parser.add_argument("--shots", type=Path, default=Path("docs/screens/shots.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/screens"))
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    args = parser.parse_args()

    shots = json.loads(args.shots.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 900}, color_scheme="light")
        page.goto(args.url)
        page.wait_for_selector("#examples button", state="attached")
        for shot in shots:
            outcome = run_claim(page, shot["claim"], args.timeout_ms)
            if outcome != shot["expect"]:
                raise SystemExit(f"{shot['claim']!r} gave {outcome}, expected {shot['expect']}")
            check_page(page, outcome)
            clip_height = shot.get("clip_height")
            clip = {"x": 0, "y": 0, "width": 1100, "height": clip_height} if clip_height else None
            page.screenshot(path=args.out / shot["file"], full_page=True, clip=clip)
            print(f"{shot['file']}: {outcome}")

        phone = browser.new_page(viewport={"width": 390, "height": 844}, color_scheme="light")
        phone.goto(args.url)
        run_claim(phone, shots[0]["claim"], args.timeout_ms)
        overflow = phone.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        if overflow > 0:
            raise SystemExit(f"page overflows a phone viewport by {overflow}px")
        print("phone width: no horizontal overflow")
        browser.close()


if __name__ == "__main__":
    main()
