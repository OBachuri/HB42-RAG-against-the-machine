import argparse
from pathlib import Path
import sys
from pydantic import TypeAdapter
import json
import re
import bm25s
from tqdm import tqdm

from src.r_data_model import MinimalSource

_CAMEL_1 = re.compile(r'([A-Z]+)([A-Z][a-z])')
_CAMEL_2 = re.compile(r'([a-z0-9])([A-Z])')
_NUMBER_AFTER = re.compile(r'([A-Za-z])([0-9])')
_NUMBER_BEFORE = re.compile(r'([0-9])([A-Za-z])')
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _get_word_from_text(text: str) -> list[str]:
    """ Prepare text for BM25 indexing. """

    ind_words = _IDENTIFIER.findall(text)

    # Splitting camelCase Words
    # Example: "myCamelCaseText" becomes "my Camel Case Text".
    # text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = _CAMEL_1.sub(r'\1 \2', text)
    text = _CAMEL_2.sub(r'\1 \2', text)

    # Separate numbers
    text = _NUMBER_AFTER.sub(r'\1 \2', text)
    text = _NUMBER_BEFORE.sub(r'\1 \2', text)

    text = text.lower()

    # Removing Punctuation (.,!?) and "_" in beginning or end of word
    # text = re.sub(r"[^\w\s]|(?<!\w)_|_(?!\w)", " ", text)
    # Removing Punctuation (.,!?) and "_" (snake )
    text = re.sub(r"[^\w\s]|[_]", " ", text)

    words = text.split()

    for w in ind_words:
        s = w.lower().strip()
        if len(s) > 1 and s not in words:
            words.append(s)

    return [w for w in words if (len(w) > 1) and (len(w) < 50)]


def r_index_bm25(param: argparse.Namespace):
    print("Indexing (BM25):")

    file_path = Path(str(param.data_processed_path)) / Path("chunks.json")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks: list[MinimalSource] = TypeAdapter(
            list[MinimalSource]).validate_python(data)
    except Exception as ex:
        print(f"Error: can't read file {file_path} \n({ex})", file=sys.stderr)
        sys.exit(1)

    file = ""
    corpus = []
    for c_ in tqdm(chunks, desc="Read row data by chunks", unit="chunk"):
        if c_.file_path == file:
            continue
        file = c_.file_path
        try:
            f_path = Path(param.data_raw_path) / file
            with open(f_path) as f:
                source = f.read()
            corpus.extend(_get_word_from_text(file))
            corpus.extend(_get_word_from_text(source))
        except Exception as ex:
            print(f"Error: can't read file {file} \n({ex})", file=sys.stderr)

    print("  Corpus words:", len(corpus))
    # print(corpus)

    # Tokenize the corpus and only keep the ids (faster and saves memory)
    # corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
    tokenizer = bm25s.tokenization.Tokenizer()
    corpus_tokens = tokenizer.tokenize(corpus, return_as="tuple",
                                       show_progress=True)

    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens, show_progress=True)

    # print("-"*20)
    # print(corpus_tokens)

    # Save the arrays to a directory...
    file_path = Path(
        str(param.data_processed_path)) / Path("animal_index_bm25")
    # You can save the corpus along with the model
    # retriever.save(file_path, corpus=corpus)
    retriever.save(file_path)

    tokenizer.save_vocab(file_path)
    tokenizer.save_stopwords(file_path)
    print(f"  Saved the index to: {file_path}")

    # get memory usage
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"  Peak memory usage: {mem_use:.2f} GB")
