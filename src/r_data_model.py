from pydantic import BaseModel, Field, model_validator
import uuid
import hashlib
from enum import Enum
from typing import Any


class RetrieveMode(Enum):
    HYBRID = 1
    BM25 = 2
    EMBEDDINGS = 3


class MinimalSource(BaseModel):
    """ Represents a single source of information
    Define one Chunk of text - a link
    to a specific part of the source file.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    id: str = ""
    chunk_id: int = 0
    parent_id: int = 0
    symbol: str = ""

    @staticmethod
    def get_chunk_id_SHA256(
         file: str,
         start_char: int,
         end_char: int,
         text: str = "") -> str:

        """ Generate "unique" id for Chunk.
        SHA-256 does not mathematically guarantee a unique ID.
        But there 2^256 ≈ 1.16 x 10^77 possible ID values
        and this is more then enough for this task
        """

        value = f"{file}:{start_char}:{end_char}:{text}"

        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def get_chunk_id(
            file: str,
            start_char: int,
            end_char: int,
            text: str = "") -> str:

        """ Generate "unique" id for Chunk.
        Use SHA-256 and then convert in UUID
        """

        value = f"{file}:{start_char}:{end_char}:{text}"

        digest = hashlib.sha256(value.encode("utf-8")).digest()

        return str(uuid.UUID(bytes=digest[:16]))

    @model_validator(mode='before')
    @classmethod
    def generate_id(cls, data: dict[Any, Any]) -> dict[Any, Any]:
        """Check ID and generate new id necessary """

        if (isinstance(data, dict)
           and (data.get("id", None) is None)
           or (data.get("id", "") == "")):
            data["id"] = cls.get_chunk_id(
                file=str(data.get("file_path")),
                start_char=int(data.get("first_character_index", 0)),
                end_char=int(data.get("last_character_index", 0))
            )
        return data


class RetrievedChunk(MinimalSource):
    """ Retrieved Chunk + score """
    score: float = 0
    metod: RetrieveMode | None = None


class UnansweredQuestion(BaseModel):
    """ Represent an unanswered question """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """ Represent an answered question """

    sources: list[MinimalSource]
    answer: str


class RagDatasetAnswered(BaseModel):
    """ List of answered question
        used by evaluation """

    rag_questions: list[AnsweredQuestion]


class RagDataset(BaseModel):
    """ The RagDataset model represents a dataset of RAG questions """
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """ Search results for a single question.
        form subject v.2.0"""

    question_id: str
    question: str
    question_str: str  # added for moulinette = question
    retrieved_sources: list[MinimalSource]

# class MinimalSearchResults(BaseModel):
#     """Search results for a single question.
#         for moulinette """
#     question_id: str
#     question_str: str # ---- in subject "question"
#     retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """ Represent the search results
    and an answer """

    answer: str


class StudentSearchResults(BaseModel):
    """ Represent search results """

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """ Represent search results  with answers """

    search_results: list[MinimalAnswer]
    k: int


class AskRequest(BaseModel):
    """ Used to validate API query"""
    question: str
