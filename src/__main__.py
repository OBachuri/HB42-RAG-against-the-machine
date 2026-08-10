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
from src.r_semantic import RSentenceTransformer

from src.r_data_model import MinimalSource, MinimalAnswer
from src.r_data_model import StudentSearchResultsAndAnswer
from src.r_data_model import StudentSearchResults, RagDataset
from src.r_data_model import RetrieveMode


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
                 llm_model_name: str = "Qwen/Qwen3-0.6B",
                 retrievemode: RetrieveMode = RetrieveMode.HYBRID
                 ):
        self.k = 10

        try:
            k_ = int(k)
            if (k_ < 1) or (k_ > 100):
                raise ValueError("Wrong value, must be (1-100),"
                                 " recomended (5-10)")
            self.k = k_
        except (ValueError, TypeError) as ex:
            print(f'{RED}Error:{RESET} Wrong value for parameter "--k={k}" '
                  '(must be an integer greater than 0 and less than 100)\n',
                  f"Used default value k={self.k}\n{ex}\n",
                  file=sys.stderr)

        self.max_chunk_size = 2000
        try:
            v_ = int(max_chunk_size)
            if (v_ < 200) or (v_ > 10000):
                raise ValueError("Wrong value, must be (200-2000)")
            self.max_chunk_size = v_
        except (ValueError, TypeError) as ex:
            print(f'{RED}Error:{RESET} Wrong value for'
                  f' parameter "max_chunk_size={max_chunk_size}" '
                  '(must be an integer greater than 200 '
                  'and equal or less than 2000)\n',
                  f"Used default value max_chunk_size={self.max_chunk_size}\n"
                  f"{ex}\n", file=sys.stderr)

        self.min_chunk_size = 200
        try:
            v_ = int(min_chunk_size)
            if (v_ < 0) or (v_ > 500):
                raise ValueError("Wrong value, must be (0-500)")
            self.min_chunk_size = v_
        except (ValueError, TypeError) as ex:
            print(f'{RED}Error:{RESET} Wrong value for'
                  f' parameter "min_chunk_size={min_chunk_size}" '
                  '(must be an integer greater than 0 '
                  'and equal or less than 500)\n',
                  f"Used default value min_chunk_size={self.min_chunk_size}\n"
                  f"{ex}\n",
                  file=sys.stderr)

        self.max_overlap = 15   # Max %% of overlap for chunks
        try:
            v_ = int(max_overlap)
            if (v_ < 0) or (v_ >= 100):
                raise ValueError("Wrong value, must be (0-99) %")
            self.max_overlap = v_
        except (ValueError, TypeError) as ex:
            print(f'{RED}Error:{RESET} Wrong value for'
                  f' parameter "max_overlap={max_overlap}" % '
                  '(must be an integer greater than 0 '
                  'and less than 100)\n(recommended: 10-20%)\n',
                  f"Used default value max_overlap={self.max_overlap} %\n",
                  f"{ex}\n",
                  file=sys.stderr)

        self.data_raw_path = data_raw_path
        self.dataset_path = dataset_path
        self.data_processed_path = data_processed_path
        self.student_search_results_path = student_search_results_path
        self.save_directory = save_directory

        # IoU - Intersection over Union
        self.eval_IoU: float = 0.05
        try:
            f_ = float(eval_IoU)
            if (f_ <= 0) or (f_ > 1):
                raise ValueError("Wrong value, must be (0.001-0.999)")
            self.eval_IoU = f_
        except (ValueError, TypeError) as ex:
            print(f'{RED}Error:{RESET} Wrong value for'
                  f' parameter "eval_IoU={eval_IoU}" % '
                  '(must be an float greater than 0 '
                  'and less than 1)\n',
                  f"Used default value eval_IoU={self.eval_IoU} \n",
                  f"{ex}\n",
                  file=sys.stderr)

        self.print_debug = bool(print_debug)
        self.llm_model_name = llm_model_name

        self.retrieve_mode: RetrieveMode = RetrieveMode.HYBRID

        try:
            if isinstance(retrievemode, RetrieveMode):
                self.retrieve_mode = retrievemode
            else:
                retrieve_mode = RetrieveMode[str(retrievemode).upper()]
                self.retrieve_mode = retrieve_mode
        except Exception as ex:
            possible_val = [m.name for m in RetrieveMode]
            print(f'{RED}Error:{RESET} Wrong value for'
                  f' parameter "retrievemode={retrievemode}"  '
                  '(must be value from the list: ', possible_val,
                  ')\n',
                  f"Used default value retrievemode={self.retrieve_mode} \n",
                  f"{ex}\n",
                  file=sys.stderr)

        self._retriver = None
        self._chunks: list[MinimalSource] = []
        self._c_llm: R_LLM | None = None

    def chunk(self):
        """
Splits large text documents into smaller chunks

  Usage:
        uv run python -m src chunk --max_chunk_size <int>

"""
        if self.print_debug:
            self._print()
        # print("Chunking:")
        self._chunks = r_chunking(self)

    def index(self):
        """
Ingest data/raw/ (--data_raw_path)
and build the index under data/processed/ (--data_processed_path).

  Usage:
        uv run python -m src index --max_chunk_size <int>
"""
        if self.print_debug:
            self._print()
        # print("index---")
        self._chunks = r_chunking(self)
        print("-"*30)
        if (self.retrieve_mode == RetrieveMode.BM25
           or self.retrieve_mode == RetrieveMode.HYBRID):
            self._retriver, self._chunks = r_index_bm25(self)
            print("-"*30)
        if (self.retrieve_mode == RetrieveMode.EMBEDDINGS
           or self.retrieve_mode == RetrieveMode.HYBRID):
            with RSentenceTransformer(self) as rs:
                rs.index(self)
            print("-"*30)

    def search(self, query: str):
        """
Return the top-k sources for a single query.

  Usage:
        uv run python -m src search <query> --k <int>
"""

        print(f"Search(k={self.k}):", query)
        if not query or not query.strip() or (len(query) < 2):
            print(f'{RED}Error:{RESET} Parameter <query> not set'
                  ' or to short! \n'
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

        if (self.retrieve_mode == RetrieveMode.EMBEDDINGS
           or self.retrieve_mode == RetrieveMode.HYBRID):
            with RSentenceTransformer(self) as rs:
                rs.retrive(self,
                           chunks=self._chunks,
                           query=query,)
            print("-"*30)

    def search_dataset(self):
        """
Run search over a whole dataset and write a StudentSearchResults JSON file.

  Usage:
        uv run python -m src search_dataset --dataset_path <path> --k <int> \
--save_directory <dir>
"""
        if self.print_debug:
            self._print()

        print(f"Search dataset(k={self.k}):")

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
        if self.print_debug:
            self._print()

        print(f"Answer to query (k={self.k}):", query)
        if not query or not query.strip() or (len(query) < 2):
            print(f'{RED}Error:{RESET} Parameter <query> not'
                  ' set or to short!\n'
                  'The <query> parameter '
                  'must be specified for the "search" commands.\n'
                  'uv run python -m src answer <query> --k <int>\n'
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
        if self.print_debug:
            self._print()

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
        if self.print_debug:
            self._print()

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

    def _print(self) -> None:
        print("-"*30)
        print("Parameters:")
        var_ = vars(self)
        for p in var_:
            if str(p)[:1] != '_':
                print(f"{p:20}:{var_[p]}")
        print("-"*30)


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
