"""
Small retry helper for hosted LLM APIs, which occasionally return transient
errors under load (rate limits, or a 503 "model currently experiencing high
demand" from Gemini) that succeed if you just wait a few seconds and try
again. Same pattern used in the RAG project's src/retry.py.
"""

import time


def with_retry(fn, max_retries: int = 4, base_delay: float = 5.0):
    """Call fn() with exponential backoff on failure. Re-raises the last
    exception if all attempts are exhausted."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, we retry any transient failure
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(f"  (transient error, retrying in {delay:.0f}s... [{attempt + 1}/{max_retries}])")
                time.sleep(delay)
    raise last_exc
