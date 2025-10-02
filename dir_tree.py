import sys
from pathlib import Path


def generate_tree(start_path='.', prefix='', is_last=True, ignore_list=None,current_depth=0):
    if ignore_list is None:
        ignore_list = ['.git', '__pycache__', '*.pyc', '.DS_Store','.idea']

    path = Path(start_path)
    if not path.exists():
        return f"路径不存在: {start_path}"

    if current_depth == 0 and start_path == '.' :
        name = Path.cwd().name
    else:
        name = path.name

    # 检查是否应该忽略
    if any(name == pattern or (pattern.startswith('*') and name.endswith(pattern[1:])) for pattern in ignore_list):
        return ''

    # 当前项目显示
    if current_depth == 0:
        result = f"{name}/\n"
    else:
        connector = "└── " if is_last else "├── "
        result = f"{prefix}{connector}{name}/\n" if path.is_dir() else f"{prefix}{connector}{name}\n"

    # 如果是目录，递归处理子项
    if path.is_dir():
        children = [p for p in path.iterdir()
                    if not any(p.name == pattern or (pattern.startswith('*') and p.name.endswith(pattern[1:]))
                               for pattern in ignore_list)]
        children.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            new_prefix = prefix + ("    " if is_last else "│   ")
            result += generate_tree(child, new_prefix, is_last_child, ignore_list,current_depth + 1)
    return result


if __name__ == '__main__':
    # 如果命令行参数个数 > 1（即用户提供了参数），就使用第一个参数 sys.argv[1] 作为路径
    # 否则使用当前目录 '.' 作为默认路径
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    tree_str = generate_tree(path)
    print(tree_str)