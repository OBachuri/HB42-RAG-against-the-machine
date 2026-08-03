import argparse
from pathlib import Path
import sys
from pydantic import TypeAdapter, RootModel
import json
import re
import bm25s
from tqdm import tqdm

from src.r_data_model import MinimalSource
from src.r_chunk import r_chunking


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


def r_index_bm25(param: argparse.Namespace
                 ) -> tuple[bm25s.BM25, list[MinimalSource]]:
    """ Create index BM25 and return Retriever object"""

    file_path = Path(str(
        param.data_processed_path)) / Path("chunks.json")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks: list[MinimalSource] = TypeAdapter(
            list[MinimalSource]).validate_python(data)
    except Exception:
        chunks = r_chunking(param)

    print("Indexing (BM25):")

    file = ""
    source: str = ""
    corpus = []
    valid_chunks: list[MinimalSource] = []
    i = 1
    for c_ in tqdm(chunks, desc="Read row data by chunks", unit="chunk"):
        words = []
        if c_.file_path == file:
            words = _get_word_from_text(
                source[c_.first_character_index:(c_.last_character_index + 1)])
        else:
            file = c_.file_path
            try:
                f_path = Path(param.data_raw_path) / file
                with open(f_path) as f:
                    source = f.read()
                words = _get_word_from_text(file)
                words.extend(_get_word_from_text(source[
                    c_.first_character_index:c_.last_character_index+1]))
            except Exception as ex:
                print(f"Error: can't read file {file} \n({ex})",
                      file=sys.stderr)
        if len(words) > 2:
            corpus.append(words)
            # c_.chunk_id = i
            valid_chunks.append(c_)
            i += 1

    # write valid chunks
    if valid_chunks:
        # Write chunks to json file:  data/processed/chunks.json
        file_path = Path(str(param.data_processed_path)) / "index_bm25" / Path("chunks.json")

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
    else:
        print("Error: no data to process - no words in chunks")
        sys.exit(1)

    # print("  Corpus words:", len(corpus))
    # print(corpus)

    # Tokenize the corpus and only keep the ids (faster and saves memory)
    # corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
    # tokenizer = bm25s.tokenization.Tokenizer()
    # corpus_tokens = tokenizer.tokenize(corpus, return_as="tuple",
    #                                    show_progress=True)

    corpus_tokens = corpus

    # Create the BM25 model and index the corpus
    # retriever = bm25s.BM25(corpus=corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)

    # print("-"*20)
    # print(corpus_tokens)

    # Save the arrays to a directory...
    file_path = Path(
        str(param.data_processed_path)) / Path("index_bm25")
    # You can save the corpus along with the model
    # retriever.save(file_path, corpus=corpus)
    retriever.save(file_path)

    # tokenizer.save_vocab(file_path)
    # tokenizer.save_stopwords(file_path)
    print(f"  Index saved the to: {file_path}")

    # get memory usage
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"  Peak memory usage: {mem_use:.2f} GB")

    # f_path = Path(param.data_raw_path) / "vllm-0.10.1/docs/serving/openai_compatible_server.md"
    # with open(f_path) as f:
    #     source = f.read()
    # words = _get_word_from_text(source)
    # print(len(words), words)
    # print("-"*20)

    return (retriever, valid_chunks)


def r_bm25_load(param: argparse.Namespace
                ) -> tuple[bm25s.BM25, list[MinimalSource]]:

    # Path where index data was stored
    index_folder_path = Path(
        str(param.data_processed_path)) / Path("index_bm25")

    if (index_folder_path.is_dir() and any(index_folder_path.iterdir())):
        print("Error: can't read index data for bm25 retriever "
              f"({index_folder_path})\n",
              "Folder not exist or empty.\n"
              'Run "index" first ',
              file=sys.stderr)
        sys.exit(1)

    try:
        retriever = bm25s.BM25.load("bm25s_very_big_index", mmap=True)
    except Exception as ex:
        print("Error: can't read index data for bm25 retriever "
              f"({index_folder_path})\n",
              ex,
              file=sys.stderr)
        sys.exit(1)

    # Read chunks list
    chunks: list[MinimalSource] = []

    file_path = index_folder_path / Path("chunks.json")
    if not (file_path.is_file() and file_path.stat().st_size > 0):
        print("Error: chunks list file not exist or empty "
              f"({file_path})\n",
              file=sys.stderr)
        sys.exit(1)

    try:
        # Read the raw text from the file
        with open(file_path, "r", encoding="utf-8") as chunk_file:
            json_string = chunk_file.read()
        # Parse and validate the JSON string back into Pydantic objects
        chunks = RootModel[
            list[MinimalSource]].model_validate_json(json_string).root

    except Exception as ex:
        print("Error: Can't read chunks list from file"
              f"({file_path})\n",
              ex,
              file=sys.stderr)
        sys.exit(1)

    return (retriever, chunks)


def r_bm25_retrive(param: argparse.Namespace,
                   retriever: bm25s.BM25,
                   chunks: list[MinimalSource],
                   query: str = "How to configure OpenAI server?"):
    # -------------------------------------
    # Query the corpus
    # query = "How to configure OpenAI server?"
    query_tokens = [_get_word_from_text(query)]
    # query_tokens = bm25s.tokenize(query)
    print("q:", query_tokens)

    # Get top-k results as a tuple of (doc ids, scores).
    # Both are arrays of shape (n_queries, k).
    # To return docs instead of IDs, set the `corpus=corpus` parameter.
    results, scores = retriever.retrieve(query_tokens, k=param.k)
    indices: list[int] = results[0].tolist()

    print("-"*20, "res")
    print(results)
    print("-"*20, "scores")
    print(scores)
    print("-"*20)
    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        print(f"Rank {i+1} (score: {score:.2f}): {doc}")
        print(chunks[doc])
    print("-"*20)
    print(indices)

    print("-"*20)
