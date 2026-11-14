"""Reproduce the oversized-estimate via POST /api/ask."""

from __future__ import annotations

import argparse
import httpx
import asyncio
import json
import time


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--setting", default="chunk400-k5")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=60) as c:
        t0 = time.time()
        async with c.stream("POST", "/api/ask", json={
            "question": "What is scaled dot-product attention?",
            "source": "stand-in",
            "retrieval_setting": args.setting.replace("-", "_"),
        }) as r:
            print("HTTP", r.status_code)
            async for line in r.aiter_lines():
                if line.startswith("event:"):
                    print(f"{(time.time()-t0)*1000:8.1f}ms  {line}")
                elif line.startswith("data:"):
                    try:
                        d = json.loads(line[5:])
                        if "step" in d or "rung" in d or "status" in d:
                            print(f"            {line[5:][:200]}")
                    except Exception:
                        pass
    # if no 120k estimate seen, say so
    print("\nIf no step showed an estimate near 120k, the oversized estimate did not reproduce.")


asyncio.run(main())
