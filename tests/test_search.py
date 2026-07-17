import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# tools/__init__ 里 `from .search import search` 让 tools.search 指向工具对象,
# 因此通过 sys.modules 取到真正的模块
import tools.search  # noqa: F401  (确保子模块已导入)
search_module = sys.modules["tools.search"]


def test_no_key(monkeypatch):
    monkeypatch.setattr(search_module, "_get_tavily_key", lambda: None)
    r = search_module.search.invoke({"query": "test"})
    assert r["success"] is False
    assert "hint" in r


def test_mock_tavily(monkeypatch):
    monkeypatch.setattr(search_module, "_get_tavily_key", lambda: "fake")

    def fake_search(query, max_results=5, search_depth="basic"):
        return {
            "results": [{"title": "t", "url": "u", "content": "c"}],
            "answer": None,
        }

    fake_module = types.SimpleNamespace(
        TavilyClient=lambda api_key: types.SimpleNamespace(search=fake_search)
    )
    monkeypatch.setitem(sys.modules, "tavily", fake_module)

    r = search_module.search.invoke({"query": "q", "num_results": 3})
    assert r["success"] is True
    assert len(r["results"]) == 1
    assert r["results"][0]["title"] == "t"
    assert r["results"][0]["url"] == "u"
