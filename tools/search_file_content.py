from langchain_core.tools import tool
from typing import Dict, Any

@tool
def search_file_content(search_dir: str, pattern: str, file_pattern: str, case_sensitive: bool, show_context: int, max_results: int, recursive: bool) -> Dict[str, Any]:
    """
    在文件中搜索指定内容，支持正则表达式匹配。可以在指定目录下搜索文件，支持多种搜索模式和输出格式。适用于代码审查、日志分析、文档搜索等场景。

功能特性：
- 支持正则表达式搜索（默认启用）
- 支持按文件类型过滤（如 .py, .js, .md 等）
- 支持按文件大小过滤
- 显示匹配行号和上下文
- 可选择输出匹配的文件路径和行号
- 支持递归搜索子目录

参数：
- search_dir: 搜索目录路径（必需）
- pattern: 搜索内容或正则表达式（必需）
- file_pattern: 文件匹配模式，如 "*.py" 或 "test_*.txt"（可选）
- case_sensitive: 是否区分大小写（可选，默认 false）
- show_context: 显示匹配行前后几行上下文（可选，默认 2）
- max_results: 最大返回结果数（可选，默认 100）
- recursive: 是否递归搜索子目录（可选，默认 true）
    """
    results = []
    
    # 构建文件搜索模式
    if file_pattern:
        search_pattern = search_dir + '/' + file_pattern
    else:
        if recursive:
            search_pattern = search_dir + '/**/*'
        else:
            search_pattern = search_dir + '/*'
    
    # 使用 Python 内置的 glob 模块
    try:
        import glob
        import re
    except ImportError:
        return {
            'success': False,
            'error': '缺少必要的模块: glob, re',
            'results': []
        }
    
    # 获取所有匹配的文件
    try:
        files = glob.glob(search_pattern, recursive=recursive)
    except Exception as e:
        return {
            'success': False,
            'error': f'文件搜索失败: {str(e)}',
            'results': []
        }
    
    if not files:
        return {
            'success': True,
            'message': f'在 {search_dir} 中未找到匹配的文件',
            'results': []
        }
    
    # 编译正则表达式
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return {
            'success': False,
            'error': f'正则表达式错误: {str(e)}',
            'results': []
        }
    
    # 搜索每个文件
    total_matches = 0
    for file_path in files:
        # 检查是否为文件（简单检查）
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            continue
        
        matches = []
        for line_num, line in enumerate(lines, 1):
            if regex.search(line):
                match = {
                    'file': file_path,
                    'line': line_num,
                    'content': line.rstrip('\n\r'),
                    'match': regex.search(line).group(0)
                }
                matches.append(match)
                total_matches += 1
        
        if matches:
            # 添加上下文
            for i, match in enumerate(matches):
                start_line = max(0, match['line'] - show_context - 1)
                end_line = min(len(lines), match['line'] + show_context)
                
                context_lines = []
                for j in range(start_line, end_line):
                    prefix = ">>> " if j == match['line'] - 1 else "    "
                    context_lines.append(f"{prefix}{j + 1}: {lines[j].rstrip()}")
                
                match['context'] = '\n'.join(context_lines)
            
            results.extend(matches)
            
            if len(results) >= max_results:
                break
    
    # 按文件分组
    grouped_results = {}
    for match in results[:max_results]:
        file_key = match['file']
        if file_key not in grouped_results:
            grouped_results[file_key] = []
        grouped_results[file_key].append(match)
    
    # 格式化输出
    formatted_results = []
    for file_path, file_matches in grouped_results.items():
        file_result = {
            'file': file_path,
            'matches': len(file_matches),
            'details': []
        }
        
        for match in file_matches:
            detail = {
                'line': match['line'],
                'content': match['content'],
                'context': match.get('context', '')
            }
            file_result['details'].append(detail)
        
        formatted_results.append(file_result)
    
    return {
        'success': True,
        'search_dir': search_dir,
        'pattern': pattern,
        'total_files': len(formatted_results),
        'total_matches': total_matches,
        'results': formatted_results
    }
