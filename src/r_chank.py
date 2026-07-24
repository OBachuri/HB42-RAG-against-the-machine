import ast
import argparse
import sys
from pathlib import Path

from src.r_data_model import MinimalSource

_TEXT_SEPARATORS: dict[str, list[str]] = {
    ".txt":  ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ",],
    ".html": ["</table>", "</div>", "</section>", "</p>",
              "<br>", "</br>", "\n\n", "\n", " "],
    ".htm":  ["</table>", "</div>", "</section>", "</p>",
              "<br>", "</br>", "\n\n", "\n", " "],
    ".py":   ["class ", "def ", "\n\n", "\n", ";", " "],
    ".md":   ["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ",
              "\n\n", "\n", ". ", "; ", ", ", " ",],
    ".toml": ["\n[", "\n\n", "\n", " "],
    ".yaml": ["\n\n", "\n", " "],
    ".sh":   ["\n\n", "\n", " "],
    ".yml":  ["\n\n", "\n", " "],
    ".cfg":  ["\n[", "\n\n", "\n", " "],
    ".json": ["\n\n", "\n", " "],
    ".cpp":  ["\n}\n", "}\n", "}", "\n\n", ";\n", ";", "\n", " "],
    ".c++":  ["\n}\n", "}\n", "}", "\n\n", ";\n", ";", "\n", " "],
    ".c":    ["\n}\n", "}\n", "}", "\n\n", ";\n", ";", "\n", " "],
    ".cu":   ["\n}\n", "}\n", "}", "\n\n", ";\n", ";", "\n", " "],
    ".js":   ["\n}\n", "}\n", "}", "\n\n", ";\n", ";", "\n", " "],
    ".rst":  ["\n====", "\n----", "\n~~~~", "\n\n", "\n", " "]
}

_DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ",
                                  "! ", "? ", "; ",
                                  ", ", " "]


def get_all_file_paths(directory_path: str | Path) -> list[Path]:
    base_dir = Path(directory_path).resolve()
    files = [file for file in base_dir.rglob("*") if file.is_file()]
    return files


def should_index(file: str) -> bool:

    try:
        with open(file, "rb") as f:
            sample = f.read(8192)  # Read only a small sample
    except Exception as ex:
        print(f"Error: can't read file {file} \n({ex})", file=sys.stderr)
        return False

    if not sample:  # skip empty files
        return False

    if b"\x00" in sample:
        return False      # binary

    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False      # probably binary

    return True


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


def r_chank_txt(file: str,
                param: argparse.Namespace,
                source: str = "",
                shift: int = 0) -> list[MinimalSource]:
    """ Chunk plain text files """

    start_char = 0
    end_char = 0
    chanks: list[MinimalSource] = []

    if not source or not source.strip():
        try:
            with open(file) as f:
                source = f.read()
        except Exception as ex:
            print(f"Error: can't read file {file} \n({ex})", file=sys.stderr)
            return []

    # print("---txt--parce  -- source len", len(source))

    if len(source) <= param.max_chunk_size:
        return [MinimalSource(file_path=file,
                              first_character_index=start_char + shift,
                              last_character_index=(shift + len(source)-1))]

    extension = Path(file).suffix.lower()
    txt_separatos = _TEXT_SEPARATORS.get(
        extension, _DEFAULT_SEPARATORS)

    end_char = len(source)-1

    while start_char < end_char:
        if end_char - start_char < param.max_chunk_size:
            if end_char - start_char >= param.min_chunk_size:
                chanks.append(
                    MinimalSource(file_path=file,
                                  first_character_index=start_char + shift,
                                  last_character_index=end_char + shift))
            else:
                end_c = end_char - param.min_chunk_size
                start_char = end_c - min(
                    param.max_chunk_size * param.max_overlap // 100,
                    param.max_chunk_size - param.min_chunk_size)
                index = 0
                for separator in txt_separatos:
                    index = source.rfind(separator, start_char, end_c)
                    if (index > start_char):
                        break
                if index == 0:
                    start_char = end_char - param.min_chunk_size
                else:
                    start_char = index
                chanks.append(
                    MinimalSource(file_path=file,
                                  first_character_index=start_char + shift,
                                  last_character_index=end_char + shift))

            # print("-- start:", start_char, "end:", end_char, "len:", 1 + end_char - start_char)
            #  print(source[start_char:end_char])

            return chanks

        end_c = min(start_char + param.max_chunk_size, end_char)
        index = 0
        for separator in txt_separatos:
            # rfind searches from right to left within
            # the [starts:end_of_search] slice
            index = source.rfind(separator, start_char, end_c)
            if (index > start_char) and (index > start_char
                                         + param.min_chunk_size):
                break
        if (index <= 0) or ((index - start_char) < param.min_chunk_size):
            index = end_c
        chanks.append(
            MinimalSource(file_path=file,
                          first_character_index=start_char + shift,
                          last_character_index=index + shift))
        # print("-- start:", start_char, "index:", index, "len:", index + 1 - start_char)
        # print(source[start_char:index])

        start_char = index

    return chanks


def r_chank_py(file: str,
               param: argparse.Namespace,
               source: str = "") -> list[MinimalSource]:
    """ Chunk Python files """

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

    if len(source) <= param.max_chunk_size:
        return [MinimalSource(file_path=file,
                              first_character_index=start_char,
                              last_character_index=(len(source)-1))]

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
            chanks.extend(r_chank_txt(file, param,
                                      source=lines[start_line],
                                      shift=line_offsets[start_line]))
        else:
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
    chanks: list[MinimalSource] = []
    for f_ in get_all_file_paths(param.data_raw_path):
        extension = f_.suffix
        size_in_bytes = f_.stat().st_size
        if extension == '.py':
            print(f"{i:4} chank py:", size_in_bytes, f_)
            # chanks.extend(r_chank_py(str(f_), param))
        elif extension in _TEXT_SEPARATORS.keys():
            print(f"{i:4} chank txt:", size_in_bytes, f_)
            chanks.extend(r_chank_txt(str(f_), param))
        elif should_index(str(f_)):
            print(f"{i:4} chank as txt:", size_in_bytes, f_)
            chanks.extend(r_chank_txt(str(f_), param))
        else:
            print(f"{i:4} skip:", size_in_bytes, f_)
        i += 1

    # print("-"*20)
    # r_chank_py("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/examples/offline_inference/basic/chat.py", param)
    print("-"*20)
    chanks = r_chank_py("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/examples/others/tensorize_vllm_model.py", param)
    # print("-"*30, " txt")
    # # chank = r_chank_txt("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/LICENSE", param)
    # # print(chank)
    # # print("-"*30, " txt")
    # # chanks = r_chank_txt("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/format.sh", param)
    # # print(chank)
    # chank = r_chank_txt("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/examples/others/tensorize_vllm_model.py", param)
    print(chanks)
