import fire
# import os
import sys
import time
from tqdm import tqdm
from pathlib import Path
from pydantic import RootModel

# path_to_file = os.path.dirname(__file__)
# sys.path.append(path_to_file)

from src.r_bm25 import r_index_bm25, r_bm25_retrive, r_bm25_load
from src.r_bm25 import r_bm25_retrive_dataset
from src.r_chunk import r_chunking
from src.r_llm import R_LLM

from src.r_data_model import MinimalSource, MinimalAnswer
from src.r_data_model import StudentSearchResultsAndAnswer
from src.r_data_model import StudentSearchResults, RagDataset


commands = {"chunk": "chunk",
            "index": "index",
            "search": "search <query>",
            "search_dataset": "search_dataset",
            "answer": "answer <query>",
            "answer_dataset": "answer_dataset",
            "evaluate": "evaluate"}

# Color constants
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"


def file_name_compare(file_1: str, file_2: str, dir_: str) -> bool:
    """ Compare two paths and return true if they are equal
    or differ only by the "dir_" prefix at the beginning. """

    file_1_p = Path(file_1)
    file_2_p = Path(file_2)
    if file_1_p == file_2_p:
        return True

    file_1_p_s = str(file_1_p)
    file_2_p_s = str(file_2_p)

    if len(file_1_p_s) < len(file_2_p_s):
        file_2_p_s = str(file_1_p)
        file_1_p_s = str(file_2_p)

    # file 1 longer
    s = file_1_p_s[len(file_1_p_s)-len(file_2_p_s):]

    if file_2_p_s == s:
        s = file_1_p_s[0:len(file_1_p_s)-len(file_2_p_s)]
        if s == dir_:
            return True
    return False


def iou(x1_from: int, x1_to: int, x2_from: int, x2_to: int) -> float:
    """ IoU - Intersection over Union

IoU = Area of intersection (overlap) / Areas of union (total combinated area)
    """
    overlap_ = max(0, (min(x1_to, x2_to) - max(x1_from, x2_from) + 1))
    union_ = (x1_to - x1_from + 1) + (x2_to - x2_from + 1) - overlap_
    if union_ == 0:
        return 0
    # print("o:", overlap_, ", u:", union_, ", IoU:", overlap_/union_)
    return overlap_/union_


class RagCLI():
    """
Retrieval-Augmented Generation CLI.

  Usage:
        uv run python -m src <command> [options/flags]

  Commands:
      chunk               Split documents into chunks.
      index               Build or update the index.
      search <query>      Search for relevant chunks.
      search_dataset      Run search over a whole dataset and write \
a StudentSearchResults JSON file.
      answer <query>      Generate an answer using RAG.
      answer_dataset      Generate answers for a dataset, producing \
a StudentSearchResultsAndAnswer JSON.
      evaluate            Report your recall@k against \
a ground-truth dataset, for testing.

"""

    def __init__(self,
                 k: int = 10,
                 max_chunk_size: int = 2000,
                 min_chunk_size: int = 200,
                 max_overlap: int = 15,
                 data_raw_path: str = "data/raw/",
                 dataset_path: str = (
            "data/datasets/UnansweredQuestions/dataset_docs_public.json"),
                 data_processed_path: str = "data/processed/",
                 student_search_results_path: str = (
            "data/output/search_results/UnansweredQuestions/"
            "dataset_docs_public.json"),
                 save_directory: str = (
            "data/output/search_results_and_answer/UnansweredQuestions"),
                 eval_IoU: float = 0.05,
                 print_debug: bool = False,
                 llm_model_name: str = "Qwen/Qwen3-0.6B"
                 ):
        self.k = k
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_overlap = max_overlap   # Max %% of overlap for chunks
        self.data_raw_path = data_raw_path
        self.dataset_path = dataset_path
        self.data_processed_path = data_processed_path
        self.student_search_results_path = student_search_results_path
        self.save_directory = save_directory
        self.eval_IoU = eval_IoU  # IoU - Intersection over Union
        self.print_debug = print_debug
        self.llm_model_name = llm_model_name

        self._retriver = None
        self._chunks: list[MinimalSource] = []
        self._c_llm: R_LLM | None = None

    def chunk(self):
        """
Splits large text documents into smaller chunks

  Usage:
        uv run python -m src chunk --max_chunk_size <int>

"""
        # print("Chunking:")
        self._chunks = r_chunking(self)

    def index(self):
        """
Ingest data/raw/ (--data_raw_path)
and build the index under data/processed/ (--data_processed_path).

  Usage:
        uv run python -m src index --max_chunk_size <int>
"""
        # print("index---")
        self._chunks = r_chunking(self)
        self._retriver, self._chunks = r_index_bm25(self)

    def search(self, query: str):
        """
Return the top-k sources for a single query.

  Usage:
        uv run python -m src search <query> --k <int>
"""

        print("Search:", query)
        if not query or not query.strip():
            print('Error: Parameter <query> not set or to short! \n'
                  'The <query> parameter '
                  'must be specified for the "search" commands. \n'
                  'uv run python -m src search <query> --k <int> \n'
                  'Example:\n'
                  'uv run python -m src search '
                  '"How to configure OpenAI server?" --k=5',
                  file=sys.stderr)
            sys.exit(1)

        if (self._retriver is None) or (not self._chunks):
            self._retriver, self._chunks = r_bm25_load(self)

        r_bm25_retrive(param=self,
                       retriever=self._retriver,
                       chunks=self._chunks,
                       query=query,
                       print_chunks=True)

    def search_dataset(self):
        """
Run search over a whole dataset and write a StudentSearchResults JSON file.

  Usage:
        uv run python -m src search_dataset --dataset_path <path> --k <int> \
--save_directory <dir>
"""
        print("Search dataset:")

        if (self._retriver is None) or (not self._chunks):
            self._retriver, self._chunks = r_bm25_load(self)

        r_bm25_retrive_dataset(
            param=self,
            retriever=self._retriver,
            chunks=self._chunks,
            write_file=True)

    def answer(self, query: str):
        """
Answer a single query using the retrieved context.

  Usage:
        uv run python -m src answer <query> --k <int>
"""
        print("Answer to query:", query)
        if not query or not query.strip():
            print('Error: Parameter <query> not set or to short! \n'
                  'The <query> parameter '
                  'must be specified for the "search" commands. \n'
                  'uv run python -m src answer <query> --k <int> \n'
                  'Example:\n'
                  'uv run python -m src answer '
                  '"How to configure OpenAI server?" --k=5',
                  file=sys.stderr)
            sys.exit(1)

        if (self._retriver is None) or (not self._chunks):
            self._retriver, self._chunks = r_bm25_load(self)

        if self._c_llm is None:
            self._c_llm = R_LLM(self.llm_model_name)

        print("Source (chunks):")
        chunks_for_RAG = r_bm25_retrive(
            param=self,
            retriever=self._retriver,
            chunks=self._chunks,
            query=query,
            print_chunks=True)

        print("------------")
        print("Answer:")

        # answer_query_txt =
        self._c_llm.query(
            question=query,
            chunks=chunks_for_RAG,
            param=self)

        print("\n------------")

    def answer_dataset(self):
        """
Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON.

  Usage:
        uv run python -m src answer_dataset --student_search_results_path \
<path> --save_directory <dir>  --k <int>
"""
        print("Answer dataset:")
        if (self._retriver is None) or (not self._chunks):
            self._retriver, self._chunks = r_bm25_load(self)

        if self._c_llm is None:
            self._c_llm = R_LLM(self.llm_model_name)

        ret_res = r_bm25_retrive_dataset(
            param=self,
            retriever=self._retriver,
            chunks=self._chunks,
            write_file=False)

        answer_query_res = []

        for q_ in tqdm(
             ret_res,
             desc="Answer on Question",
             unit="Question"):
            print("\n Question:", q_.question)
            answer_query_res.append(
                MinimalAnswer(
                 answer=str(self._c_llm.query(
                   question=q_.question,
                   chunks=q_.retrieved_sources,
                   param=self)),
                 question_id=q_.question_id,
                 question=q_.question,
                 retrieved_sources=q_.retrieved_sources))
            print()

        print("\n------------")

        st_result = StudentSearchResultsAndAnswer(
            k=self.k,
            search_results=answer_query_res)

        json_string = RootModel(st_result).model_dump_json(indent=2)
        file_path = Path(self.save_directory) / (
            "dataset_docs_public.json")

        try:
            # This creates the folders if they do not exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as ex:
            print("Error: can't create folder to store"
                  "dataset_docs_public.json! \n",
                  ex, file=sys.stderr)
            sys.exit(1)

        try:
            # Writing to JSON file
            with open(file_path, "w", encoding="utf-8") as res_file:
                res_file.write(json_string)
            print("  Saved student_search_results_and_answer to :", file_path)
        except Exception as ex:
            print("Error: can't store student_search_results_and_answer!"
                  f"({file_path})\n",
                  ex, file=sys.stderr)
            sys.exit(1)

    def evaluate(self):
        """
Report recall@k against a ground-truth dataset, for testing.

  Usage:
        uv run python -m src evaluate \
--student_search_results_path <path> --dataset_path <path>
"""
        print("Evaluation:")

        # Read file to evaluate
        file_path = Path(str(self.student_search_results_path))
        if not (file_path.is_file() and file_path.stat().st_size > 0):
            print("Error: Can't read file to evaluate. No file or empty file. "
                  f"({file_path})\n",
                  file=sys.stderr)
            sys.exit(1)

        to_eval = None

        try:
            # Read the raw text from the file
            with open(file_path, "r", encoding="utf-8") as q_file:
                json_string = q_file.read()
            # Parse and validate the JSON string back into Pydantic objects
            to_eval = RootModel[
                StudentSearchResults].model_validate_json(json_string).root
        except Exception as ex:
            print("Error: Can't read json from file"
                  f"({file_path})\n",
                  ex,
                  file=sys.stderr)
            sys.exit(1)

        # Read reference file
        file_path = Path(str(self.dataset_path))
        if not (file_path.is_file() and file_path.stat().st_size > 0):
            print("Error: Can't read file to evaluate. No file or empty file. "
                  f"({file_path})\n",
                  file=sys.stderr)
            sys.exit(1)

        to_reference = None

        try:
            # Read the raw text from the file
            with open(file_path, "r", encoding="utf-8") as q_file:
                json_string = q_file.read()
            # Parse and validate the JSON string back into Pydantic objects
            to_reference = RootModel[
                RagDataset].model_validate_json(json_string).root
        except Exception as ex:
            print("Error: Can't read json from file"
                  f"({file_path})\n",
                  ex,
                  file=sys.stderr)
            sys.exit(1)

        print(f"{BLUE}Data is valid.{RESET}")

        print("To eval:", len(to_eval.search_results),
              "queries, For reference:", len(to_reference.rag_questions),
              "queries.")

        pass_query = {}
        pass_query[to_eval.k] = 0

        for i, q_ref in enumerate(tqdm(to_reference.rag_questions,
                                  desc="Evaliate seach result for query",
                                  unit="query")):
            # find in data for evaluation
            q_eval = None
            if ((i < len(to_eval.search_results))
               and (to_eval.search_results[
                   i].question_id == q_ref.question_id)):
                q_eval = to_eval.search_results[i]
            if q_eval is None:
                for q_e in to_eval.search_results:
                    if q_ref.question_id == q_e.question_id:
                        q_eval = q_e
                        break
            if q_eval is None:
                continue

            ii = 0
            for ei in range(0, len(q_eval.retrieved_sources)):
                if file_name_compare(
                     q_eval.retrieved_sources[ei].file_path,
                     q_ref.sources[0].file_path, self.data_raw_path):
                    # file found - check for overlap
                    if iou(q_eval.retrieved_sources[ei].first_character_index,
                           q_eval.retrieved_sources[ei].last_character_index,
                           q_ref.sources[0].first_character_index,
                           q_ref.sources[0].last_character_index
                           ) >= self.eval_IoU:
                        pass_query[ei + 1] = pass_query.get(ei + 1, 0) + 1
                        ii = ei + 1
                        break
                # print("ok")
            if (ii == 0) and self.print_debug:
                print(f"\n Source not found (n: {i}"
                      f", id: {q_ref.question_id}):\n", q_ref.question)
            # print(q_eval.retrieved_sources[0].file_path)
            # print(q_ref.sources[0].file_path)
            # print(i, "---")

        print("Evaluation results:")
        print("="*30)
        print("Questions evaluated:", len(to_reference.rag_questions))
        from_starts = 0
        for i in range(1, max(pass_query.keys())+1):
            curr_ = pass_query.get(i, 0)
            from_starts += curr_
            curr_recal = from_starts / len(to_reference.rag_questions)
            print(f"{BLUE}"
                  f"recall@{i} {curr_recal:2.3f} ({(curr_recal*100):3.1f} %)",
                  f"{RESET} quantity:", curr_)


def main() -> None:
    """Start the Python Fire command-line interface."""

    # Record start time
    start_time = time.perf_counter()

    fire.Fire(RagCLI, name="RAG", command=None)

    end_time = time.perf_counter()
    # Calculate duration
    duration = int(end_time - start_time)
    print(f"Execution time: {duration // 60}:{duration % 60}"
          " (minutes:seconds)")
    print("-"*20)


if __name__ == "__main__":
    main()
