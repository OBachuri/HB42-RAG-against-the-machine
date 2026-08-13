from pydantic import BaseModel, Field, model_validator
import uuid
import hashlib
from enum import Enum


class RetrieveMode(Enum):
    HYBRID = 1
    BM25 = 2
    EMBEDDINGS = 3


class MinimalSource(BaseModel):
    """ Define one Chunk of text - a link
    to a specific part of the source file.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    id: str = ""
    chunk_id: int = 0
    parent_id: int = 0

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
    def generate_id(cls, data: dict) -> dict:
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
    score: float = 0
    metod: RetrieveMode | None = None


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search results for a single question.
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
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int


class AskRequest(BaseModel):
    question: str
