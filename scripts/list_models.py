"""List Groq models available to your key. No guessed limits.

Usage:
    python scripts/list_models.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY not set (.env)")
        return 1
    print(f"GET {base}/models")
    try:
        r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=15.0)
    except httpx.HTTPError as e:
        print(f"request failed: {type(e).__name__}: {e}")
        return 1
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:500])
        return 1
    data = r.json().get("data", [])
    print(f"\n{len(data)} models:\n")
    print(f"{'id':<45}{'owned_by':<18}{'context window':>15}")
    print("-" * 78)
    for m in sorted(data, key=lambda x: x.get("id", "")):
        mid = m.get("id", "?")
        owner = m.get("owned_by", "?")
        ctx = m.get("context_window") or m.get("context_length") or m.get("max_context_length")
        ctx_s = str(ctx) if ctx is not None else "not listed"
        print(f"{mid:<45}{owner:<18}{ctx_s:>15}")
    print("\nNote: request-rate limits are NOT in this response.")
    print("Check https://console.groq.com/settings/limits for the chosen model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
