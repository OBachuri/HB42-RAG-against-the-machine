from __future__ import annotations

import ast
# import argparse
import sys
from pathlib import Path
from pydantic import RootModel
from tqdm import tqdm

from markdown_it import MarkdownIt
from markdown_it.token import Token

from src.r_data_model import MinimalSource

from typing import TypedDict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.__main__ import RagCLI


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
    """ Check if file not need to be index """

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
    """ Get list of all files in folder (recursive) """

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
    """ Get char position for the end of chunk """

    rez = -1
    for i in range(start_line, len(line_offsets)):
        if line_offsets[i] > start + max_offset:
            rez = i - 1
            return rez
    if start_line < len(line_offsets):
        return len(line_offsets) - 1
    return rez


def r_chunk_txt(file: str,
                param: RagCLI,
                source: str = "",
                shift: int = 0,
                chunk_id: int = 1,
                txt_separatos: list[str] = [],
                parent_id: int = 0,
                symbol: str = ""
                ) -> list[MinimalSource]:
    """ Chunk plain text files """

    start_char = 0
    end_char = 0
    chunks: list[MinimalSource] = []

    # f_path = Path(param.data_raw_path).resolve() / file
    f_path = Path(param.data_raw_path) / file

    if not source:   # or not source.strip():
        try:
            with open(f_path) as f:
                source = f.read()
            chunk_id = 0
        except Exception as ex:
            print(f"Error: can't read file '{file}' \n({ex})", file=sys.stderr)
            return []

    # print("---txt--parce  -- source len", len(source), "chunk_id:", chunk_id)
    # print("source:", source)

    if len(source) <= param.max_chunk_size:
        return [MinimalSource(file_path=str(f_path),
                              first_character_index=start_char + shift,
                              last_character_index=(shift + len(source)-1),
                              parent_id=parent_id,
                              symbol=symbol,
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
                    MinimalSource(file_path=str(f_path),
                                  first_character_index=start_char + shift,
                                  last_character_index=end_char + shift,
                                  parent_id=parent_id,
                                  symbol=symbol,
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
                    MinimalSource(file_path=str(f_path),
                                  first_character_index=start_char + shift,
                                  last_character_index=end_char + shift,
                                  parent_id=parent_id,
                                  symbol=symbol,
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
            MinimalSource(file_path=str(f_path),
                          first_character_index=start_char + shift,
                          last_character_index=index + shift,
                          parent_id=parent_id,
                          symbol=symbol,
                          chunk_id=chunk_id))
        # print("-- start:", start_char, "index:",
        # index, "len:", index + 1 - start_char)
        # print(source[start_char:index])

        start_char = index

    return chunks


def r_chunk_md(file: str,
               param: RagCLI,
               source: str = "",
               chunk_id: int = 1,
               shift: int = 0) -> list[MinimalSource]:
    """ Chunk Markdown files """

    class Section(TypedDict):
        id: int
        p_id: int | None
        token: Token
        lines: tuple[int, int]
        chars: tuple[int, int]
        txt: str

    def get_symbol(current_section: int) -> str:
        if ((current_section < 1)
           or sections_[current_section]["p_id"] is None):
            return ""
        s = sections_[current_section]["p_id"]
        sym_ = ""
        while not (s is None):
            sym_ = sections_[s]["txt"] + "\n" + sym_
            s = sections_[s]["p_id"]
        return sym_

    def where_stop(start_sect: int, sections_l: list[Section]) -> int:
        """ Return next sections """

        cc_s = start_sect
        if (cc_s >= len(sections_l)-1):
            return cc_s
        end_c = int(sections_l[cc_s]["chars"][1])
        p1_id = sections_l[cc_s]["p_id"]
        while (cc_s < len(sections_l)
               and (end_c - start_char < param.max_chunk_size)):
            cc_s += 1
            while (cc_s < len(sections_l)
                   and (sections_l[cc_s]["p_id"] != p1_id)):
                cc_s += 1
            if cc_s < len(sections_l):
                end_c = int(sections_l[cc_s]["chars"][1])

        if (cc_s > start_sect) or (cc_s >= len(sections_l) - 1):
            return cc_s

        cc_s += 1
        cur_sec_tag = sections_l[start_sect]["token"].tag
        next_sec_tag = sections_l[cc_s]["token"].tag

        if cur_sec_tag < next_sec_tag:
            return where_stop(cc_s, sections_l)
        return cc_s

        # -------------

    chunks: list[MinimalSource] = []
    start_char = 0
    start_line = 0
    current_section = 0

    f_path = Path(param.data_raw_path) / file

    if not source or not source.strip():
        try:
            with open(f_path) as f:
                source = f.read()
        except Exception as ex:
            print(f"Error: can't read file {file} \n({ex})", file=sys.stderr)
            return []
        chunk_id = 0

    if len(source) <= param.max_chunk_size:
        return [MinimalSource(file_path=str(f_path),
                              first_character_index=start_char + shift,
                              last_character_index=shift+(len(source)-1),
                              chunk_id=chunk_id)]

    lines = source.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    # i = 0
    for line in lines:
        line_offsets.append(offset)
        # print("l-", i, "-", offset, ":", lines[i], end="")
        offset += len(line)
        # i += 1

    md = MarkdownIt()

    tokens = md.parse(source)

    # heads = []

    heads = [t for t in tokens if t.type == "heading_open"]
    sections_: list[Section] = []
    parent_id: list[int] = []

    for i, h in enumerate(heads):
        while parent_id and heads[parent_id[-1]].tag >= h.tag:
            del parent_id[-1]
        parent_id.append(i)

        if h.map is None:
            start_l = 0
            end_l = len(lines)
        else:
            start_l = h.map[0]
            end_l = h.map[1]
        txt = ("".join(lines[start_l:end_l])).strip()[:200]
        end_l = len(lines)
        for next_h in heads[i + 1:]:
            if next_h.tag <= h.tag:
                if not (next_h.map is None):
                    end_l = next_h.map[0]
                break
        start_c = line_offsets[start_l]
        end_c = line_offsets[end_l-1] + len(lines[end_l-1])
        if len(parent_id) > 1:
            p_id = parent_id[-2]
        else:
            p_id = None
        sections_.append({"id": i,
                          "p_id": p_id,
                          "token": h,
                          "lines": (start_l, end_l),
                          "chars": (start_c, end_c),
                          "txt": txt})
        # print(i, start_l, end_l, txt, h)

    if len(sections_) < 1:
        # can't find headers - parse as text
        return r_chunk_txt(file, param, source=source)

    # print("--- parse as Markdown: file", file)

    # Chek if there is something on the top of the first header
    # If there a lot - add it in chunks
    header_start = int(sections_[0]["chars"][0])
    header_line_end = header_start + len(lines[sections_[0]["lines"][0]])

    if (header_line_end >= max(param.min_chunk_size, param.max_chunk_size // 2)
       and (header_start >= param.min_chunk_size)):
        chunks.extend(r_chunk_txt(file, param,
                                  source=source[0:header_start],
                                  chunk_id=chunk_id,
                                  shift=shift
                                  ))
        start_char = header_start
        chunk_id += len(chunks)
        start_line = int(sections_[0]["lines"][0])

    # # for - test
    # for s in sections_:
    #     print(s["id"], s["p_id"], "l:", s["lines"], "c:", s["chars"],
    #           ",c_len:",
    #           s["chars"][1]-s["chars"][0],
    #           s["token"].tag,
    #           s["txt"])

    current_section = 0
    while (start_line < len(lines)
           and start_char < len(source)
           and chunk_id < 5000):

        # check for the end of the source or end of section
        if ((len(source) - start_char <= param.max_chunk_size)
           or current_section >= (len(sections_) - 1)):
            symbol = get_symbol(current_section)
            chunks.extend(r_chunk_txt(
                file, param,
                source=source[start_char:],
                chunk_id=chunk_id,
                shift=shift+start_char,
                symbol=symbol
                ))
            return chunks

        c_s = current_section
        p_id = sections_[c_s]["p_id"]
        cur_sec_tag = sections_[c_s]["token"].tag
        end_c = int(sections_[c_s]["chars"][1])

        while (c_s < len(sections_)
                and (end_c - start_char < param.max_chunk_size)):
            c_s += 1
            while (c_s < len(sections_)
                   and sections_[c_s]["p_id"] != p_id
                   and cur_sec_tag < sections_[c_s]["token"].tag):
                c_s += 1
            if (c_s < len(sections_)):
                end_c = int(sections_[c_s]["chars"][1])
        if c_s > current_section:
            symbol = get_symbol(current_section)
            if (c_s < len(sections_)):
                end_c = int(sections_[c_s]["chars"][0])
            else:
                end_c = len(source)-1
            chunks.append(MinimalSource(
                file_path=str(f_path),
                first_character_index=start_char + shift,
                last_character_index=(shift + end_c - 1),
                # parent_id=parent_id,
                symbol=symbol,
                chunk_id=chunk_id))
            chunk_id += 1
            if (c_s >= len(sections_)):
                return chunks
            current_section = c_s
            start_char = end_c
            start_line = int(sections_[c_s]["lines"][0])
            continue

        c_s += 1
        start_c = int(sections_[c_s]["chars"][0])
        start_l = int(sections_[c_s]["lines"][0])

        if (start_c - start_char + len(lines[start_l]) >= max(
           param.max_chunk_size // 2, param.min_chunk_size)):
            symbol = get_symbol(current_section)
            chunks.extend(r_chunk_txt(
                file, param,
                source=source[start_char:start_c],
                chunk_id=chunk_id,
                shift=shift+start_char,
                symbol=symbol
                ))
            start_char = start_c
            chunk_id = chunks[-1].chunk_id + 1
            start_line = int(sections_[c_s]["lines"][0])
            current_section = c_s
            continue

        c_s = where_stop(c_s, sections_)

        if c_s < len(sections_):
            start_c = int(sections_[c_s]["chars"][0])
            start_l = int(sections_[c_s]["lines"][0])

            symbol = get_symbol(current_section)
            chunks.extend(r_chunk_txt(
                file, param,
                source=source[start_char:start_c],
                chunk_id=chunk_id,
                shift=shift+start_char,
                symbol=symbol
                ))

            start_char = start_c
            chunk_id = chunks[-1].chunk_id + 1
            start_line = int(sections_[c_s]["lines"][0])
            current_section = c_s
        else:
            current_section = len(sections_) - 1

    #  # -- print(sections_)
    # for token in tokens:
    #     if token.type == "heading_open":
    #         level = int(token.tag[1])  # h1 -> 1, h2 -> 2, ...

    #         headings.append({
    #             "level": level,
    #             "map": token.map,
    #         })

    # for token in tokens:
    #     print(
    #         "type:", token.type,
    #         ", tag:", token.tag,
    #         ", map:", token.map,
    #         repr(token.content[:50]), "===="
    #     )
    #     print(token)
    #     print("--------")

    return chunks


def r_chunk_py(file: str,
               param: RagCLI,
               source: str = "",
               chunk_id: int = 1) -> list[MinimalSource]:
    """ Chunk Python files """

    start_char = 0
    start_line = 0
    # end_char = 0
    # end_line = 0
    chunks: list[MinimalSource] = []

    # print("--py-file:", file)

    f_path = Path(param.data_raw_path) / file

    if not source or not source.strip():
        try:
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
        return [MinimalSource(file_path=str(f_path),
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
                file_path=str(f_path),
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
                MinimalSource(file_path=str(f_path),
                              first_character_index=start_char,
                              last_character_index=end_,
                              chunk_id=chunk_id,
                              parent_id=parent_id)
                          )

            parent_id = 0
            chunk_id += 1

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

        start_char = (line_offsets[end_line_numb]
                      + len(lines[end_line_numb]) + 1)
        start_line = end_line_numb + 1

    return chunks


def r_chunking(param: RagCLI) -> list[MinimalSource]:
    print("Chunking:")

    f_count = 0
    chunks: list[MinimalSource] = []
    all_files = get_all_file_paths(param.data_raw_path)
    for f_ in tqdm(all_files, desc="Chunking files", unit="file"):
        extension = f_.suffix
        # size_in_bytes = f_.stat().st_size
        if extension == '.py':
            # print(f"{i:4} chunk py:", size_in_bytes, f_)
            # c_ = r_chunk_py(str(f_), param)
            # print(len(c_))
            # chunks.extend(c_)
            chunks.extend(r_chunk_py(str(f_), param))
        elif extension == '.md':
            # chunks.extend(r_chunk_txt(str(f_), param))
            chunks.extend(r_chunk_md(str(f_), param))
        elif extension in _TEXT_SEPARATORS.keys():
            # print(f"{i:4} chunk txt:", size_in_bytes, f_)
            chunks.extend(r_chunk_txt(str(f_), param))
        elif should_index(str(Path(param.data_raw_path) / f_)):
            # print(f"{i:4} chunk as txt:", size_in_bytes, f_)
            chunks.extend(r_chunk_txt(str(f_), param))
        else:
            continue

        f_count += 1

    print("  Files processed :", f_count)
    print("  Total chunks    :", len(chunks))

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
            print("  Saved in        :", file_path)
        except Exception as ex:
            print(f"Error: can't store chunks.json! ({file_path})\n",
                  ex, file=sys.stderr)
            sys.exit(1)
    return chunks
