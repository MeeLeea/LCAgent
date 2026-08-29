import json
import os
import sys

import pytest
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONFIG_PATH = os.path.join(ROOT, "config", "llm_config.json")

with open(CONFIG_PATH, encoding="utf-8") as _f:
    _cfg = json.load(_f)

PROVIDERS = _cfg.get("providers", {})


def _load_provider(name):
    prov = PROVIDERS.get(name)
    if prov is None:
        pytest.skip(f"provider {name!r} not found in llm_config.json")
    api_key = prov.get("api_key") or os.environ.get(prov.get("env_key", ""))
    if not api_key:
        pytest.skip(f"provider {name!r} has empty api_key; set it to run live checks")
    return prov, api_key


def _chat_completion(prov, api_key, model, timeout=30):
    """Send a minimal chat completion request, return (ok, detail)."""
    base = prov["base_url"].rstrip("/")
    if base.endswith("/chat/completions"):
        url = base
    elif base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base + "/chat/completions"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8,
                "stream": False,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return True, None
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:  # 网络/解析等任何异常都算不可用(BLE001 已在项目配置中忽略)
        return False, f"{type(exc).__name__}: {exc}"


def unavailable_models(provider_name):
    """Return the model names that do not work for a provider."""
    prov, api_key = _load_provider(provider_name)
    bad = []
    for model in prov.get("models", []):
        ok, _detail = _chat_completion(prov, api_key, model)
        if not ok:
            bad.append(model)
    return bad


def pytest_generate_tests(metafunc):
    if "provider" not in metafunc.fixturenames:
        return

    requested = metafunc.config.getoption("--provider")
    if requested is not None and requested not in PROVIDERS:
        raise pytest.UsageError(
            f"unknown provider {requested!r}; available providers: "
            f"{', '.join(sorted(PROVIDERS))}"
        )

    providers = [requested] if requested else sorted(PROVIDERS)
    metafunc.parametrize("provider", providers)


def test_provider_models_usable(provider):
    """Check every configured model and report the failed model names."""
    bad = unavailable_models(provider)
    if bad:
        raise AssertionError(
            f"{provider} unavailable models: {bad}"
        )
    assert bad == []


if __name__ == "__main__":
    for provider in sorted(PROVIDERS):
        result = unavailable_models(provider)
        print(f"{provider} unavailable models: {result}")
