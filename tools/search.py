"""
联网搜索工具 - 基于 Tavily Search API
使用 LangChain @tool 装饰器；未配置 Key 时优雅降级(明确提示,不返回假数据)
"""
from langchain_core.tools import tool
from typing import Dict, Any, Optional
import os
import json


def _get_tavily_key() -> Optional[str]:
    """获取 Tavily API Key: 优先环境变量,回退 llm_config.json 的 tavily.api_key"""
    key = os.environ.get("TAVILY_API_KEY")
    if key:
        return key
    cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "llm_config.json"
    )
    if os.path.exists(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("tavily", {}).get("api_key")
        except (json.JSONDecodeError, IOError):
            return None
    return None


@tool
def search(query: str, num_results: int = 5, search_depth: str = "basic") -> Dict[str, Any]:
    """
    联网搜索工具,基于 Tavily Search API 搜索互联网上的实时信息。

    当任务需要最新/外部知识(如新闻、文档、技术资料)时,应调用本工具。
    未配置 Tavily API Key 时会明确返回失败提示,而不会编造结果。

    Args:
        query: 搜索查询关键词
        num_results: 返回结果数量,默认 5(上限 20)
        search_depth: 搜索深度, 'basic'(默认) 或 'advanced'(更深入)

    Returns:
        包含搜索结果的字典: success / query / answer / results[{title,url,content}]
    """
    api_key = _get_tavily_key()
    if not api_key:
        return {
            "success": False,
            "error": "未配置 Tavily API Key",
            "hint": "请设置环境变量 TAVILY_API_KEY,或在 llm_config.json 中配置 tavily.api_key",
            "results": [],
        }

    try:
        from tavily import TavilyClient
    except ImportError:
        return {
            "success": False,
            "error": "未安装 tavily-python,请运行: pip install tavily-python",
            "results": [],
        }

    try:
        client = TavilyClient(api_key=api_key)
        depth = search_depth if search_depth in ("basic", "advanced") else "basic"
        resp = client.search(
            query=query,
            max_results=min(max(num_results, 1), 20),
            search_depth=depth,
        )
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in resp.get("results", [])
        ]
        return {
            "success": True,
            "query": query,
            "answer": resp.get("answer"),
            "results": results,
            "message": f"找到 {len(results)} 条相关结果",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Tavily 搜索失败: {e}",
            "results": [],
        }


if __name__ == "__main__":
    import json as _json
    result = search.invoke({"query": "Python教程", "num_results": 3})
    print(_json.dumps(result, ensure_ascii=False, indent=2))
