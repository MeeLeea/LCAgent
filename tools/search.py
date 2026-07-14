"""
联网搜索工具 - 使用LangChain @tool装饰器
"""
from langchain_core.tools import tool
from typing import Dict, Any


@tool
def search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    联网搜索工具，用于搜索互联网上的信息。

    Args:
        query: 搜索查询关键词
        num_results: 返回结果数量，默认5

    Returns:
        包含搜索结果的字典
    """
    # 模拟实现，实际项目中可接入真实搜索API
    return {
        "success": True,
        "query": query,
        "results": [
            {
                "title": f"搜索结果 {i+1}",
                "url": f"https://example.com/result/{i+1}",
                "snippet": f"这是关于 '{query}' 的搜索结果摘要 {i+1}..."
            }
            for i in range(num_results)
        ],
        "message": f"找到 {num_results} 条相关结果（模拟数据）"
    }


if __name__ == "__main__":
    import json
    result = search.invoke({"query": "Python教程", "num_results": 3})
    print(json.dumps(result, ensure_ascii=False, indent=2))
