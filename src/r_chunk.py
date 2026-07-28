import ast
import argparse
import sys
from pathlib import Path
from pydantic import RootModel

from src.r_data_model import MinimalSource

_TEXT_SEPARATORS: dict[str, list[str]] = {
    ".txt":  ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ",],
    ".html": ["</table>", "</div>", "</section>", "</p>",
              "<br>", "</br>", "\n\n",
              ". ", "! ", "? ", "; ", ", ", "\n", " "],
    ".htm":  ["</table>", "</div>", "</section>", "</p>",
              "<br>", "</br>", "\n\n",
              ". ", "! ", "? ", "; ", ", ", "\n", " "],
    ".py":   ["\nclass ", "class ", "def ", "while ", "\n\n",
              "for ", "if ", "\n", ";", " "],
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

_SKIP_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "venv",
    ".venv",
}

_SKIP_EXTENSIONS = {
    ".pyc", ".pyo",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".wav", ".ogg", ".flac",
    ".mp4", ".avi", ".mov", ".mkv",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".pdf",
    ".exe", ".dll", ".so", ".o", ".a",
}

_MAX_FILE_SIZE = 4 * 1024 * 1024      # 4 MB


def _should_skip(path: Path) -> bool:
    # Skip directories in the path
    if any(part in _SKIP_DIRS for part in path.parts):
        return True

    # Skip by extension
    if path.suffix.lower() in _SKIP_EXTENSIONS:
        return True

    # Skip very large files
    if path.stat().st_size > _MAX_FILE_SIZE:
        return True

    return False


def get_all_file_paths(directory_path: str | Path) -> list[Path]:
    base_dir = Path(directory_path).resolve()
    return [
        file.relative_to(base_dir) for file in base_dir.rglob("*")
        if file.is_file() and not _should_skip(file)]


def should_index(file: str) -> bool:
    """  Check if file contains text data and must be indexed """

    try:
        with open(file, "rb") as f:
            sample = f.read(8192)  # Read only a small sample
    except Exception as ex:
        print(f"Error: can't read file {file} \n({ex})", file=sys.stderr)
        return False

    if not sample:        # skip empty files
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


def r_chunk_txt(file: str,
                param: argparse.Namespace,
                source: str = "",
                shift: int = 0,
                chunk_id: int = 1,
                txt_separatos: list[str] = [],
                parent_id: int = 0
                ) -> list[MinimalSource]:
    """ Chunk plain text files """

    start_char = 0
    end_char = 0
    chunks: list[MinimalSource] = []

    if not source:   # or not source.strip():
        try:
            f_path = Path(param.data_raw_path).resolve() / file
            with open(f_path) as f:
                source = f.read()
            chunk_id = 0
        except Exception as ex:
            print(f"Error: can't read file '{file}' \n({ex})", file=sys.stderr)
            return []

    # print("---txt--parce  -- source len", len(source), "chunk_id:", chunk_id)
    # print("source:", source)

    if len(source) <= param.max_chunk_size:
        return [MinimalSource(file_path=file,
                              first_character_index=start_char + shift,
                              last_character_index=(shift + len(source)-1),
                              parent_id=parent_id,
                              chunk_id=chunk_id)]

    if len(txt_separatos) < 1:
        extension = Path(file).suffix.lower()
        txt_separatos = _TEXT_SEPARATORS.get(
            extension, _DEFAULT_SEPARATORS)

    end_char = len(source)-1
    chunk_id -= 1

    while start_char < end_char:
        chunk_id += 1
        if end_char - start_char < param.max_chunk_size:
            if end_char - start_char >= param.min_chunk_size:
                chunks.append(
                    MinimalSource(file_path=file,
                                  first_character_index=start_char + shift,
                                  last_character_index=end_char + shift,
                                  parent_id=parent_id,
                                  chunk_id=chunk_id))
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
                chunks.append(
                    MinimalSource(file_path=file,
                                  first_character_index=start_char + shift,
                                  last_character_index=end_char + shift,
                                  parent_id=parent_id,
                                  chunk_id=chunk_id))

            # print("-- start:", start_char, "end:",
            # end_char, "len:", 1 + end_char - start_char)
            #  print(source[start_char:end_char])

            return chunks

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
        chunks.append(
            MinimalSource(file_path=file,
                          first_character_index=start_char + shift,
                          last_character_index=index + shift,
                          parent_id=parent_id,
                          chunk_id=chunk_id))
        # print("-- start:", start_char, "index:",
        # index, "len:", index + 1 - start_char)
        # print(source[start_char:index])

        start_char = index

    return chunks


def r_chunk_py(file: str,
               param: argparse.Namespace,
               source: str = "", chunk_id: int = 1) -> list[MinimalSource]:
    """ Chunk Python files """

    start_char = 0
    end_char = 0
    start_line = 0
    end_line = 0
    chunks: list[MinimalSource] = []

    # print("--py-file:", file)

    if not source or not source.strip():
        try:
            f_path = Path(param.data_raw_path) / file
            with open(f_path) as f:
                source = f.read()
        except Exception as ex:
            print(f"Error: can't read file {file} \n({ex})", file=sys.stderr)
            return []
        chunk_id = 0

    try:
        tree = ast.parse(source)
    except Exception as ex:
        print(f"Warning: can't parse file {file} \n({ex}) as python code",
              file=sys.stderr)
        return r_chunk_txt(file, param, source=source)

    if len(source) <= param.max_chunk_size:
        return [MinimalSource(file_path=file,
                              first_character_index=start_char,
                              last_character_index=(len(source)-1),
                              chunk_id=chunk_id)]

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

    if len(top_level) < 1:
        # if AST can't find objects in file - parse as text
        return r_chunk_txt(file, param, source=source)

    top_level.sort(key=lambda node: node.lineno)
    curr_object = 0

    lines = source.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    # i = 0
    for line in lines:
        line_offsets.append(offset)
        # print("l-", i, "-", offset, ":", lines[i], end="")
        offset += len(line)
        # i += 1

    parent_id = 0
    while start_line < len(lines) and chunk_id < 5000:
        if len(source) - start_char < param.max_chunk_size:
            # end of file
            chunks.append(MinimalSource(
                file_path=file,
                first_character_index=start_char,
                last_character_index=(len(source)-1),
                chunk_id=chunk_id,
                parent_id=parent_id))
            break

        # find     curr_object
        if curr_object < len(top_level):
            end_line_numb = top_level[curr_object].end_lineno
            if end_line_numb is None:
                end_line_numb = len(lines)-1
            else:
                end_line_numb -= 1
            end_line_end_offset = (line_offsets[end_line_numb]
                                   + len(lines[end_line_numb]))

        while (curr_object < len(top_level)
               and start_char > end_line_end_offset):
            curr_object += 1
            if curr_object < len(top_level):
                end_line_numb = top_level[curr_object].end_lineno
                if end_line_numb is None:
                    end_line_numb = len(lines)-1
                else:
                    end_line_numb -= 1
                end_line_end_offset = (line_offsets[end_line_numb]
                                       + len(lines[end_line_numb]))

        if curr_object >= len(top_level):
            # end of object but not end of file
            # parce as text
            chunks.extend(r_chunk_txt(file, param,
                                      source=source[start_char:],
                                      shift=start_char,
                                      chunk_id=chunk_id))
            break

        top_line_numb = top_level[curr_object].lineno - 1
        top_line_offset = line_offsets[top_line_numb]
        top_line_len = len(lines[top_line_numb])
        end_line_numb = top_level[curr_object].end_lineno
        if end_line_numb is None:
            end_line_numb = len(lines)-1
        else:
            end_line_numb -= 1
        end_line_offset = line_offsets[end_line_numb]
        end_line_len = len(lines[end_line_numb])

        if ((start_char + param.max_chunk_size)
           < (top_line_offset + top_line_len + param.min_chunk_size)):
            # add text before object start
            chunks.extend(r_chunk_txt(file, param,
                                      source=source[
                                          start_char:
                                          top_line_offset],
                                      shift=start_char,
                                      chunk_id=chunk_id,
                                      txt_separatos=_DEFAULT_SEPARATORS))
            chunk_id = chunks[-1].chunk_id + 1
            start_line = top_line_numb
            start_char = top_line_offset
            # print("-- text_before_new_id", chunk_id, "next_char", start_char)

        if (end_line_offset + end_line_len
           - start_char) <= param.max_chunk_size:
            # Current object end in current chunk
            # Check if there could be another one that ends there
            end_ = end_line_offset + end_line_len
            for i in range(curr_object + 1, len(top_level)):
                end_line_numb_i = top_level[i].end_lineno
                if end_line_numb_i is None:
                    end_line_numb_i = len(lines)-1
                else:
                    end_line_numb_i -= 1
                end_line_end_offset_of_i = (
                    line_offsets[end_line_numb_i]
                    + len(lines[end_line_numb_i]))
                if (end_line_end_offset_of_i
                   - line_offsets[start_line]) <= param.max_chunk_size:
                    end_ = end_line_end_offset_of_i
                    end_line_numb = end_line_numb_i
                    curr_object = i + 1
                else:
                    break
            chunks.append(
                MinimalSource(file_path=file,
                              first_character_index=start_char,
                              last_character_index=end_,
                              chunk_id=chunk_id,
                              parent_id=parent_id)
                          )
            # print("---the end of object: chunk id = ", chunk_id,
            #       "start_char:", start_char,"end_line", end_line_numb)

            parent_id = 0
            chunk_id += 1

        # elif len(lines[start_line]) > param.max_chunk_size:
        #     chunks.extend(r_chunk_txt(file, param,
        #                               source=lines[start_line],
        #                               shift=line_offsets[start_line],
        #                               chunk_id=chunk_id))
        #     if end_line_numb > start_line:
        #         parent_id = chunk_id
        #     chunk_id = chunks[-1].chunk_id + 1
        #     start_char = line_offsets[start_line] + len(lines[start_line])
        #     start_line += 1
        #     continue

        else:
            # there should be parser for next level of objects
            # but now we parse it as text
            parent_id = chunk_id
            # print("big obj: id", chunk_id)
            chunks.extend(r_chunk_txt(file, param,
                                      source=source[
                                          start_char:
                                          end_line_offset + end_line_len],
                                      shift=line_offsets[start_line],
                                      chunk_id=chunk_id,
                                      parent_id=parent_id
                                      ))
            chunk_id = chunks[-1].chunk_id + 1
            parent_id = 0
            curr_object += 1

        start_char = line_offsets[end_line_numb] + len(lines[end_line_numb]) + 1
        start_line = end_line_numb + 1
        # print("end_line:", end_line_numb, "id:", chunk_id, "next_char:", start_char, "obj:", curr_object)

    # print("Top:", top_level)

    # for node in top_level:
    #     if hasattr(node, "name"):
    #         parent = getattr(node, "parent", None)
    #         print("t: l-b",
    #               node.lineno,
    #               "sh l-b",
    #               line_offsets[node.lineno-1],
    #               node.col_offset,
    #               "l-e",
    #               node.end_lineno,
    #               node.end_col_offset,
    #               node.name, type(parent).__name__ if parent else None)

    # print("-----code")
    # code = ast.get_source_segment(source, node)
    # print(code)

    return chunks


def r_index(param: argparse.Namespace) -> None:
    print("---chunk---")

    # i = 1
    chunks: list[MinimalSource] = []
    for f_ in get_all_file_paths(param.data_raw_path):
        extension = f_.suffix
        # size_in_bytes = f_.stat().st_size
        if extension == '.py':
            # print(f"{i:4} chunk py:", size_in_bytes, f_)
            # c_ = r_chunk_py(str(f_), param)
            # print(len(c_))
            # chunks.extend(c_)
            chunks.extend(r_chunk_py(str(f_), param))
        elif extension in _TEXT_SEPARATORS.keys():
            # print(f"{i:4} chunk txt:", size_in_bytes, f_)
            chunks.extend(r_chunk_txt(str(f_), param))
        elif should_index(str(Path(param.data_raw_path) / f_)):
            # print(f"{i:4} chunk as txt:", size_in_bytes, f_)
            chunks.extend(r_chunk_txt(str(f_), param))
        # else:
        #     print(f"{i:4} skip:", size_in_bytes, f_)
        # i += 1

    # print("-"*20)
    # r_chunk_py("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/examples/offline_inference/basic/chat.py", param)
    print("-"*20)
    # chunks = r_chunk_py("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/examples/others/tensorize_vllm_model.py", param)
    # print("-"*30, " txt")
    # # chunks = r_chunk_txt("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/LICENSE", param)
    # # print(chunks)
    # # print("-"*30, " txt")
    # # chunks = r_chunk_txt("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/format.sh", param)
    # # print(chunk)
    # chunks = [r_chunk_py("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/examples/others/tensorize_vllm_model.py", param)]
    # chunks = [r_chunk_py("/home/obachuri/avb/Python/RAG-against-the-machine/my-01/data/raw/vllm-0.10.1/examples/offline_inference/basic/chat.py", param)]

    # print(chunks)

    if chunks:
        # Write chunks to json file:  data/processed/chunks.json
        file_path = Path(str(param.data_processed_path)) / Path("chunks.json")

        try:
            # This creates the folders if they do not exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as ex:
            print("Error: can't create folder to store chunks.json! \n",
                  ex, file=sys.stderr)
            sys.exit(1)

        json_string = RootModel(chunks).model_dump_json(indent=2)

        try:
            # Writing the list of chunks to a JSON file
            with open(file_path, "w", encoding="utf-8") as chunk_file:
                chunk_file.write(json_string)
        except Exception as ex:
            print(f"Error: can't store chunks.json! ({file_path})\n",
                  ex, file=sys.stderr)
            sys.exit(1)

    print("---index---")
