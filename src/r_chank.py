import ast
import argparse
import sys
from pathlib import Path

from src.r_data_model import MinimalSource

_TEXT_SEPARATORS: dict[str, list[str]] = {
    ".txt":  ["\n\n", "\n", ". ", "; ", ", ", " ",],
    ".html": ["</div>", "</section>", "</p>", "</table>"
              "<br>", "</br>", "\n\n", "\n", " "],
    ".htm":  ["</div>", "</section>", "</p>", "</table>"
              "<br>", "</br>", "\n\n", "\n", " "],
    ".py":   ["\"\"\"", "\n\n", "\n", ";", " "],
    ".md":   ["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ",
              "\n\n", "\n", ". ", "; ", ", ", " ",],
    ".toml": ["\n[", "\n\n", "\n", " "],
    ".yaml": ["\n\n", "\n", " "],
    ".yml":  ["\n\n", "\n", " "],
    ".cfg":  ["\n[", "\n\n", "\n", " "],
    ".json": ["\n\n", "\n", " "],
    ".rst":  ["\n====", "\n----", "\n~~~~", "\n\n", "\n", " "]
}


def get_all_file_paths(directory_path: str | Path) -> list[Path]:
    base_dir = Path(directory_path).resolve()
    files = [file for file in base_dir.rglob("*") if file.is_file()]
    return files


def r_chank_txt(file: str,
                param: argparse.Namespace,
                source: str = "") -> list[MinimalSource]:
    return []


def r_get_end_line(start: int, line_offsets: list[int],
                   max_offset: int = 2000, start_line: int = 0) -> int:
    rez = -1
    for i in range(start_line, len(line_offsets)):
        if line_offsets[i] > start + max_offset:
            rez = i - 1
            return rez
    if start_line < len(line_offsets):
        return len(line_offsets) - 1
    return rez


def r_chank_py(file: str,
               param: argparse.Namespace,
               source: str = "") -> list[MinimalSource]:

    start_char = 0
    end_char = 0
    start_line = 0
    end_line = 0
    chanks: list[MinimalSource] = []

    print("---f:", file)

    if not source or not source.strip():
        try:
            with open(file) as f:
                source = f.read()
        except Exception as ex:
            print(f"Error: can't read file {file} \n({ex})", file=sys.stderr)
            return []

    try:
        tree = ast.parse(source)
    except Exception as ex:
        print(f"Warning: can't parse file {file} \n({ex}) as python code",
              file=sys.stderr)
        return r_chank_txt(file, param, source=source)

    top_level = [
        node for node in ast.walk(tree)
        if isinstance(node, (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        ))
        and hasattr(node, "lineno")
        and getattr(node, "parent", None) is None
    ]

    top_level.sort(key=lambda node: node.lineno)

    lines = source.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    i = 0
    for line in lines:
        line_offsets.append(offset)
        print("l-", i, "-", offset, ":", lines[i], end="")
        offset += len(line)
        i += 1

    while start_line < len(lines):
        if len(lines[start_line]) > param.max_chunk_size:
            chanks.extend(r_chank_txt(file, param, source=lines[start_line]))

        end_line = r_get_end_line(start_char, line_offsets,
                                  max_offset=param.max_chunk_size,
                                  start_line=start_line)
        print("end_line:", end_line)
        start_char = line_offsets[end_line] + len(lines[end_line])
        start_line = end_line + 1


    print("Top:", top_level)

    for node in top_level:
        if hasattr(node, "name"):
            parent = getattr(node, "parent", None)
            print("s:",
                  node.lineno,
                  node.col_offset,
                  node.end_lineno,
                  node.end_col_offset,
                  node.name, type(parent).__name__ if parent else None)
    return chanks


def r_index(param: argparse.Namespace) -> None:
    print("---index---")

    i = 1
    for f_ in get_all_file_paths(param.data_raw_path):
        extension = f_.suffix
        size_in_bytes = f_.stat().st_size
        if size_in_bytes <= param.max_chunk_size:
            print(f"{i:4} add:", size_in_bytes, f_)
        elif extension == '.py':
            print(f"{i:4} chank py:", size_in_bytes, f_)
            # r_chank_py(f_,param)
        elif extension == '.md':
            print(f"{i:4} chank md:", size_in_bytes, f_)
        else:
            print(f"{i:4} skip:", size_in_bytes, f_)
        i += 1

    print("-"*20)
    r_chank_py("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/examples/offline_inference/basic/chat.py", param)
    print("-"*20)
    r_chank_py("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/examples/others/tensorize_vllm_model.py", param)
