#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取本地时间的工具
"""
from langchain_core.tools import tool
import datetime
import json
import sys

@tool
def get_local_time():
    """
    获取本地时间
    """
    try:
        # 获取当前本地时间
        current_time = datetime.datetime.now()
        
        # 返回格式化的时间字符串
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 返回结果
        result = {
            "success": True,
            "message": "获取本地时间成功",
            "data": {
                "local_time": time_str,
                "timestamp": current_time.timestamp(),
                "year": current_time.year,
                "month": current_time.month,
                "day": current_time.day,
                "hour": current_time.hour,
                "minute": current_time.minute,
                "second": current_time.second,
                "weekday": current_time.strftime("%A"),  # 星期几
                "week_number": current_time.strftime("%W")  # 年内第几周
            }
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        error_result = {
            "success": False,
            "message": f"获取本地时间失败: {str(e)}",
            "error": str(e)
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 如果作为脚本直接运行，输出结果
    output = get_local_time()
    print(output)
