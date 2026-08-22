""" Query result caching for search and answer """

from __future__ import annotations

import pathlib
import shutil
import hashlib
import sys
import json

from pydantic import RootModel

from src.r_data_model import MinimalAnswer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.__main__ import RagCLI

_CACHE_DIR = pathlib.Path("data/processed/cache")


def clear_cache() -> None:
    """ Delete cache folder"""

    if not _CACHE_DIR.exists():
        return
    shutil.rmtree(_CACHE_DIR)


def _cache_id(query: str, param: RagCLI) -> str:
    """ Return hash cache key for a (query + k). """

    value = f"{query.strip()}:{param.k}:{param.retrieve_mode}"

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cache_push(res: MinimalAnswer, param: RagCLI) -> None:
    """ Save search results to disk cache. """

    file_path = _CACHE_DIR / f"{_cache_id(res.question, param)}.json"

    json_string = RootModel(res).model_dump_json(indent=2)

    try:
        # This creates the folders if they do not exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as ex:
        print("Error: can't create folder to store"
              f" {file_path}! \n",
              ex, file=sys.stderr)
        sys.exit(1)

    try:
        # Writing to JSON file
        with open(file_path, "w", encoding="utf-8") as res_file:
            res_file.write(json_string)
    except Exception as ex:
        print("Error: can't save file!"
              f"({file_path})\n",
              ex, file=sys.stderr)
        sys.exit(1)


def cache_get(query: str, param: RagCLI) -> MinimalAnswer | None:
    """ Return cached results or None if not cached. """

    path = _CACHE_DIR / f"{_cache_id(query, param)}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return MinimalAnswer(**data)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
