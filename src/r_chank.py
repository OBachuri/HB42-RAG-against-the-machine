# import ast
import argparse
from pathlib import Path

def get_all_file_paths(directory_path: str | Path) -> list[Path]:
    
    base_dir = Path(directory_path).resolve()
    
    files = [file for file in base_dir.rglob("*") if file.is_file()]
    
    return files

def r_chank():
    print("---")


def r_index(param: argparse.Namespace) -> None:
    print("---index---")

    for f_ in get_all_file_paths(param.data_raw_path):
        extension = f_.suffix
        size_in_bytes = f_.stat().st_size
        if size_in_bytes <= param.max_chunk_size:
            print("add:", size_in_bytes, f_)
        elif extension == '.py':
            print("chank py:", size_in_bytes, f_)
        elif extension == '.md':
            print("chank md:", size_in_bytes, f_)
        else:
            print("skip:", size_in_bytes, f_)



    r_chank()
