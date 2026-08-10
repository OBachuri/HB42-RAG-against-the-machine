import sys
import os
import json
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from pydantic import TypeAdapter, RootModel

from src.r_data_model import MinimalSource
from src.r_chunk import r_chunking

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.__main__ import RagCLI


INDEX_DIR = "vector_index"
COLLECTION_NAME = "RAG_index"


class RSentenceTransformer():
    """ Semantic embeddings

        Tested with embedding modeles:
            1) BAAI/bge-small-en-v1.5
                - better result (512 tokens) but it take 20 min
                  for embedding on H42 PC (i7-13 wo GPU)
            2) all-MiniLM-L6-v2
                - moderate result (256 tokens) but 4 min on embedding
    """

    def __init__(self, param: RagCLI,  model_name: str = 'all-MiniLM-L6-v2'):

        self.model_name: str = model_name
        self._file_path: str = ""

        # Disable the missing token warning (hides the warning)
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_VERBOSITY"] = "error"
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        # Load embedding model
        self._model = SentenceTransformer(self.model_name)

        file_path = Path(str(param.data_processed_path)) / Path(INDEX_DIR)

        try:
            # This creates the folders if they do not exist
            file_path.mkdir(parents=True, exist_ok=True)
        except Exception as ex:
            print("Error: can't create folder to store index! \n",
                  ex, file=sys.stderr)
            sys.exit(1)

        # Create persistent Qdrant database
        self._client = QdrantClient(path=str(file_path))
        self._file_path = str(file_path)

        # Determine vector dimension
        test_vector = self._model.encode("test")
        vector_size = len(test_vector)

        # Create collection
        if not self._client.collection_exists(COLLECTION_NAME):

            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def index(self, param: RagCLI) -> None:
        """ Create vector index """

        # Read chunks list
        chunks: list[MinimalSource] = []

        file_path = Path(str(
            param.data_processed_path)) / Path("chunks.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunks = TypeAdapter(
                list[MinimalSource]).validate_python(data)
        except Exception:
            chunks = r_chunking(param)
            print("-"*30)
        print(f"Indexing (Embedding model:{self.model_name}):")

        file = ""
        source: str = ""
        i = 0
        words: list[str] = []  # text of chunks
        for c_ in tqdm(chunks, desc="Read row data by chunks", unit="chunk"):
            if c_.file_path == file:
                words.append(source[c_.first_character_index:(
                    c_.last_character_index + 1)])
            else:
                file = c_.file_path
                try:
                    # f_path = Path(param.data_raw_path) / file
                    f_path = Path(file)
                    with open(f_path) as f:
                        source = f.read()
                    words.append(file + " " + source[
                        c_.first_character_index:c_.last_character_index+1])
                except Exception as ex:
                    print(f"Error: can't read file {file} \n({ex})",
                          file=sys.stderr)
            i += 1
        # print(i, words)

        vectors = self._model.encode(
            words,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # Store vectors
        points = []

        for chunk, vector in tqdm(zip(chunks, vectors),
                                  desc="Prepare point to store",
                                  unit="point"):
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=vector.tolist(),
                    payload={
                        # "text": chunk["text"],
                        "file": chunk.file_path
                        # "symbol": chunk["symbol"],
                    },
                )
            )

        print("Saving vector index to:", self._file_path)

        BATCH_SIZE = 250
        for i in tqdm(range(0, len(points), BATCH_SIZE),
                      desc="Saving point to store",
                      unit=f"batch ({BATCH_SIZE} points)"):
            batch = points[i:i + BATCH_SIZE]

            self._client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
            )

        # self._client.upsert(
        #     collection_name=COLLECTION_NAME,
        #     points=points,
        # )

        print("Index created.")

    def retrive(self,
                param: RagCLI,
                chunks: list[MinimalSource] = [],
                query: str = "",
                print_chunks: bool = False) -> list[MinimalSource]:

        # Convert question to vector
        query_vector = self._model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        # Search
        results = self._client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=param.k,
            ).points

        print(results)

        return []

    def __del__(self):
        if hasattr(self, "_client"):
            self.close()

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
