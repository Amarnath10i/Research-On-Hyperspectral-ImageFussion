"""Start a Kaggle kernel via Playwright with API key cookie injection."""
import asyncio
import base64
from playwright.async_api import async_playwright

KERNEL_URL = "https://www.kaggle.com/code/amarnath10chinu/feinfn-cave-x4-fusion-benchmark-sota-52-47db"
USERNAME = "amarnath10chinu"
API_KEY = "2b9850a9d08d535269e8aef3acc65803"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Step 1: Visit kaggle.com to get XSRF token
        print("[1] Getting XSRF token...")
        await page.goto("https://www.kaggle.com", wait_until="networkidle", timeout=30000)
        cookies = await context.cookies()
        xsrf = None
        for c in cookies:
            if c["name"] == "XSRF-TOKEN":
                xsrf = c["value"]
                print(f"  XSRF token: {xsrf[:20]}...")
                break

        if not xsrf:
            print("  No XSRF token found, trying without...")

        # Step 2: Try to authenticate via internal API
        print("[2] Attempting API key authentication...")
        token_b64 = base64.b64encode(f"{USERNAME}:{API_KEY}".encode()).decode()

        # Use evaluate to make authenticated requests from the page context
        result = await page.evaluate(f"""
            async () => {{
                try {{
                    const resp = await fetch('/api/v1/kernels/list?mine=true&page=1&page_size=3', {{
                        headers: {{
                            'Authorization': 'Basic {token_b64}'
                        }},
                        credentials: 'include'
                    }});
                    return {{ status: resp.status, text: await resp.text() }};
                }} catch(e) {{
                    return {{ error: e.message }};
                }}
            }}
        """)
        print(f"  API list kernels: status={result.get('status')}")
        if "text" in result:
            print(f"  Response: {result['text'][:300]}")

        # Step 3: Try to set auth cookie and navigate to kernel
        print("[3] Setting auth cookies...")
        await context.add_cookies([
            {"name": "ka_sessionid", "value": "auto_" + API_KEY[:16], "domain": ".kaggle.com", "path": "/"},
        ])

        # Step 4: Try navigating to the notebook with all routes injecting auth
        auth_header = f"Basic {token_b64}"
        await page.route("**/api/v1/**", lambda route: route.continue_(
            headers={**route.request.headers, "Authorization": auth_header}
        ))

        print(f"[4] Loading kernel page...")
        resp = await page.goto(KERNEL_URL, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(5)
        print(f"  Status: {resp.status}, URL: {page.url}")

        # Check if we're logged in
        is_logged = await page.evaluate("""
            () => {
                const el = document.querySelector('[data-testid="user-menu"]') ||
                           document.querySelector('.current-user-display-name') ||
                           document.querySelector('header img[alt*="avatar"]');
                return el !== null;
            }
        """)
        print(f"  Logged in: {is_logged}")

        # Screenshot for debugging
        await page.screenshot(path="C:/Users/sande/AppData/Local/Temp/kaggle_page.png")
        print("[5] Screenshot saved")

        # Try clicking Run if visible
        all_buttons = await page.query_selector_all("button")
        for btn in all_buttons:
            text = (await btn.inner_text()).strip()
            if text:
                print(f"  Button: '{text[:60]}'")

        # Look for the Run/Play button
        run_button = None
        for selector in [
            'button:has-text("Run")',
            'button:has-text("Save Version")',
            '[class*="run"]',
            'button[aria-label*="run" i]',
            'button[aria-label*="Run"]',
        ]:
            run_button = await page.query_selector(selector)
            if run_button:
                break

        if run_button:
            print("[6] Found Run button, clicking...")
            await run_button.click()
            await asyncio.sleep(5)
            print(f"[7] After click URL: {page.url}")
        else:
            print("[6] No Run button found (not authenticated)")

        await browser.close()


asyncio.run(main())
