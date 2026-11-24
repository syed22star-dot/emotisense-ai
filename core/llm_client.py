"""
core/llm_client.py
------------------
Resilient Gemini 4 / 3.x Interactions API v1 + OpenAI fallback LLM client
with a 120s cooldown circuit-breaker so Pass 2 and follow-up turns execute fast.
Reads credentials cleanly from request/header or environment variables (.env).
"""

import json
import os
import re
import time
import requests


DEFAULT_GEMINI_CHAIN = ["gemini-4-hydrogen", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
_FAILED_UNTIL = {}


def extract_gemini_text(payload: dict) -> str:
    """Extract text from Gemini v1/interactions `steps[]` or fallback `output_text`."""
    if not isinstance(payload, dict):
        return ""
    texts = []
    for step in payload.get("steps") or []:
        if step.get("type") == "model_output":
            for part in step.get("content") or []:
                if part.get("type") == "text" and part.get("text"):
                    texts.append(part["text"])
    if texts:
        return "\n".join(texts).strip()
    return str(payload.get("output_text") or "").strip()


def parse_json_envelope(raw_text: str) -> dict:
    """Safely extract JSON object from raw LLM text (supports markdown code fences)."""
    cleaned = (raw_text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1)
    else:
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]
    return json.loads(cleaned)


def resolve_llm_credentials(req_data: dict, request_headers) -> tuple[str, str]:
    req_provider = (req_data.get("provider") or os.environ.get("LLM_PROVIDER") or "").lower().strip()
    session_key = (request_headers.get("X-LLM-API-Key") or req_data.get("api_key") or "").strip()
    env_gemini = os.environ.get("GEMINI_API_KEY", "").strip()
    env_openai = os.environ.get("OPENAI_API_KEY", "").strip()
    if req_provider == "openai":
        return "openai", (session_key or env_openai)
    if req_provider == "gemini":
        return "gemini", (session_key or env_gemini)
    if session_key:
        return ("openai" if session_key.startswith("sk-") else "gemini"), session_key
    if env_gemini:
        return "gemini", env_gemini
    if env_openai:
        return "openai", env_openai
    return "gemini", ""


def _build_gemini_model_chain() -> list[str]:
    primary = os.environ.get("GEMINI_MODEL", "gemini-4-hydrogen").strip()
    fallbacks_raw = os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.8-flash")
    base = [primary] if primary else []
    for item in [x.strip() for x in fallbacks_raw.split(",") if x.strip()] + DEFAULT_GEMINI_CHAIN:
        if item not in base:
            base.append(item)
    now = time.time()
    healthy = [m for m in base if _FAILED_UNTIL.get(m, 0.0) <= now]
    cooled = [m for m in base if _FAILED_UNTIL.get(m, 0.0) > now]
    return healthy + cooled


def call_llm_raw(provider: str, api_key: str, system_prompt: str, prompt_text: str) -> tuple[str, str, dict]:
    """
    Invoke LLM with automatic transient-overload (503/429/timeout) model fallback
    and a 120s cooldown circuit-breaker so Pass 2 executes without re-hitting overloaded models.
    Returns (response_text, model_used, telemetry_dict).
    """
    provider = (provider or "").lower().strip()
    t0 = time.perf_counter()

    if provider == "gemini":
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
            "Api-Revision": "2026-05-20",
        }
        chain = _build_gemini_model_chain()
        errors = []
        for idx, model_name in enumerate(chain):
            req_timeout = (3, 6) if idx == 0 and len(chain) > 1 else (4, 15)
            try:
                resp = requests.post(
                    "https://generativelanguage.googleapis.com/v1/interactions",
                    headers=headers,
                    json={"model": model_name, "system_instruction": system_prompt, "input": prompt_text},
                    timeout=req_timeout,
                )
                if resp.ok:
                    text = extract_gemini_text(resp.json())
                    if text:
                        _FAILED_UNTIL.pop(model_name, None)
                        return text, model_name, {
                            "attempts": idx + 1,
                            "fallback_used": idx > 0 or model_name != os.environ.get("GEMINI_MODEL", "gemini-4-hydrogen"),
                            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                        }
                    errors.append(f"{model_name}: empty output")
                else:
                    msg = resp.text[:180].replace("\n", " ")
                    errors.append(f"{model_name} HTTP {resp.status_code}: {msg}")
                    if resp.status_code in (400, 401, 403):
                        raise RuntimeError(errors[-1])
                    _FAILED_UNTIL[model_name] = time.time() + 120.0
            except requests.RequestException as exc:
                errors.append(f"{model_name} net: {exc}")
                _FAILED_UNTIL[model_name] = time.time() + 120.0
        raise RuntimeError(" | ".join(errors[-2:]))

    if provider == "openai":
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={
                "model": model_name,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_text}],
            },
            timeout=(5, 25),
        )
        if not resp.ok:
            raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text[:220]}")
        return (
            resp.json()["choices"][0]["message"]["content"].strip(),
            model_name,
            {"attempts": 1, "fallback_used": False, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)},
        )

    raise ValueError(f"Unsupported provider: {provider}")
