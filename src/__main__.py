import argparse
import os
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


commands = {"chunk": "chunk",
            "index": "index",
            "search": "search <query>",
            "search_dataset": "search_dataset",
            "answer": "answer <query>",
            "answer_dataset": "answer_dataset",
            "evaluate": "evaluate"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG against the machine - Retrieval-Augmented Generation"
        '\n (uv run python -m src <command> [options])',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'command',
        nargs='*',
        help="Command: " + str(tuple(commands.values()))
    )

    parser.add_argument(
        "--max_chunk_size",
        default=2000,
        help="Max size of a chunk",
        type=int,
    )

    parser.add_argument(
        "--min_chunk_size",
        default=200,
        help="Min size of a chunk",
        type=int,
    )

    parser.add_argument(
        "--max_overlap",
        default=15,
        help="Max %% of overlap for chunks ",
        type=int,
    )

    parser.add_argument(
        "--data_raw_path",
        default="data/raw/",
        help="Path to directory with data for index",
    )

    parser.add_argument(
        "--dataset_path",
        default="data/datasets/UnansweredQuestions/dataset_docs_public.json",
        help="Path to output directory",
    )

    parser.add_argument(
        "--data_processed_path",
        default="data/processed/",
        help="the directory with data produced by the index command",
        type=str,
    )

    parser.add_argument(
        "--student_search_results_path",
        default="data/output/search_results/UnansweredQuestions/"
        "dataset_docs_public.json",
        help="Path to search results file",
        type=str,
    )

    parser.add_argument(
        "--save_directory",
        default="data/output/search_results_and_answer/UnansweredQuestions",
        help="Path to output directory",
        type=str,
    )

    parser.add_argument(
        "--k",
        default=10,
        help="Quantity of chunks to retrieve",
        type=int,
    )

    param = parser.parse_args()

#    param._get_args()

    # if (str(param.the_task).strip().lower() not in commands.keys()):
    #     print(f"Command not found: '{param.the_task}'.",
    #           " use: ", list(commands.keys()),
    #           file=sys.stderr)
    #     sys.exit(1)

    if not os.path.exists(param.data_raw_path):
        print("Parameter error: "
              f"Directory with raw data not found: '{param.data_raw_path}'",
              file=sys.stderr)
        sys.exit(1)

    if (param.max_chunk_size < 200):
        print("Parameter error: "
              f"!!! max_chunk_size to small ({param.max_chunk_size}). "
              "Will be used max_chunk_size = 200")
        param.max_chunk_size = 200

    if (param.min_chunk_size < 100):
        print("Parameter error: "
              f"!!! min_chunk_size to small ({param.max_chunk_size}). "
              "Will be used min_chunk_size = 100")
        param.min_chunk_size = 100

    if (param.min_chunk_size >= param.max_chunk_size):
        print("Parameter error: "
              "!!! min_chunk_size must be smaller than max_chunk_size. "
              "Will be used min_chunk_size = 100")
        param.min_chunk_size = 100

    if param.k <= 0:
        print("Parameter error: "
              "--k must be greater than zero. Will be used k=5")
        param.k = 5

    # if not os.path.exists(param.functions_definition):
    #     print(f"File not found: '{param.functions_definition}'",
    #           file=sys.stderr)
    #     sys.exit(1)
    # if not os.path.exists(param.input):
    #     print(f"File not found: '{param.input}'",
    #           file=sys.stderr)
    #     sys.exit(1)

    print("-"*20)
    print("Parameters:")
    args_dict = vars(param)
    for p in args_dict:
        print(f"    {p}: {args_dict[p]}")
    print("-"*20)

    # Record start time
    start_time = time.perf_counter()
    duration_all = 0

    # r_chunking(param)

    # Record end time
    # end_time = time.perf_counter()
    # duration = int(end_time - start_time)
    # duration_all = duration
    # print(f"Execution time: {duration // 60}:{duration % 60}"
    #       " (minutes:seconds)")

    # start_time = end_time

    arg_commands = []
    search_query: list[str] = []
    answer_query: list[str] = []

    i = 0
    while i < len(param.command):
        if param.command[i] in commands.keys():
            if str(param.command[i]).lower() == "search":
                try:
                    query = param.command[i + 1]
                    # print(i, query)
                    if len(query) < 2:
                        raise IndexError
                    search_query.append(query)
                except Exception:
                    print('Error: Parameter <query> not set or to short! \n'
                          'The <query> parameter '
                          'must be specified for the "search" commands. \n'
                          'uv run python -m src search <query> --k <int> \n'
                          'Example:\n'
                          'uv run python -m src search '
                          '"How to configure OpenAI server?" --k=5',
                          file=sys.stderr)
                    sys.exit(1)
                i += 1
            elif str(param.command[i]).lower() == "answer":
                try:
                    query = param.command[i + 1]
                    if len(query) < 2:
                        raise IndexError
                    answer_query.append(query)
                except Exception:
                    print('Error: Parameter <query> not set or to short! \n'
                          'The <query> parameter '
                          'must be specified for the "answer" commands. \n'
                          'uv run python -m src answer <query> --k <int> \n'
                          'Example:\n'
                          'uv run python -m src answer '
                          '"How to configure OpenAI server?" --k=5',
                          file=sys.stderr)
                    sys.exit(1)
                i += 1
            else:
                arg_commands.append(param.command[i])
        else:
            print(f'Error: Command "{param.command[i]}" not allowed!',
                  file=sys.stderr)
            sys.exit(1)
        i += 1

    i = 0
    retriver = None
    chunks: list[MinimalSource] = []
    c_llm = None

    if "chunk" in arg_commands:
        chunks = r_chunking(param)
        i = 1
        # Record end time
        end_time = time.perf_counter()
        # Calculate duration
        duration = int(end_time - start_time)
        duration_all += duration
        print(f"Execution time: {duration // 60}:{duration % 60}"
              " (minutes:seconds)")
        print("-"*20)
        start_time = end_time
    if "index" in arg_commands:
        retriver, chunks = r_index_bm25(param)
        i += 1
        # Record end time
        end_time = time.perf_counter()
        # Calculate duration
        duration = int(end_time - start_time)
        duration_all += duration
        print(f"Execution time: {duration // 60}:{duration % 60}"
              " (minutes:seconds)")
        print("-"*20)
        start_time = end_time

    for q in search_query:
        print("Search:", q)
        if (retriver is None) or (not chunks):
            retriver, chunks = r_bm25_load(param)

        r_bm25_retrive(param=param,
                       retriever=retriver,
                       chunks=chunks,
                       query=q,
                       print_chunks=True)
        # ---------
        i += 1
        # Record end time
        end_time = time.perf_counter()
        duration = int(end_time - start_time)
        duration_all += duration
        print(f"Execution time: {duration // 60}:{duration % 60}"
              " (minutes:seconds)")
        print("-"*20)
        start_time = end_time

    for q in answer_query:
        print("Query to answer:", q)

        if (retriver is None) or (not chunks):
            retriver, chunks = r_bm25_load(param)

        if c_llm is None:
            c_llm = R_LLM()

        chunks_for_RAG = r_bm25_retrive(
            param=param,
            retriever=retriver,
            chunks=chunks,
            query=q,
            print_chunks=False)

        print("Answer:")

        # answer_query_txt =
        c_llm.query(
            question=q,
            chunks=chunks_for_RAG,
            param=param)

        # print("Answer:\n", answer_query_txt, "\n", "-------")

        print("\n-------")

        # ---------
        i += 1
        # Record end time
        end_time = time.perf_counter()
        duration = int(end_time - start_time)
        duration_all += duration
        print(f"Execution time: {duration // 60}:{duration % 60}"
              " (minutes:seconds)")
        print("-"*20)
        start_time = end_time

    if "search_dataset" in arg_commands:
        print("Search dataset:")
        if (retriver is None) or (not chunks):
            retriver, chunks = r_bm25_load(param)

        r_bm25_retrive_dataset(
            param=param,
            retriever=retriver,
            chunks=chunks,
            write_file=True)

        i += 1
        # Record end time
        end_time = time.perf_counter()
        # Calculate duration
        duration = int(end_time - start_time)
        duration_all += duration
        print(f"Execution time: {duration // 60}:{duration % 60}"
              " (minutes:seconds)")
        print("-"*20)
        start_time = end_time

    if "answer_dataset" in arg_commands:
        print("Answer dataset:")
        if (retriver is None) or (not chunks):
            retriver, chunks = r_bm25_load(param)

        ret_res = r_bm25_retrive_dataset(
            param=param,
            retriever=retriver,
            chunks=chunks,
            write_file=False)

        if c_llm is None:
            c_llm = R_LLM()

        answer_query_res: list[MinimalAnswer] = []

        for q_ in tqdm(
             ret_res,
             desc="Answer on Question",
             unit="Question"):
            print("\n Question:", q_.question)
            answer_query_res.append(
                MinimalAnswer(
                 answer=str(c_llm.query(
                   question=q_.question,
                   chunks=q_.retrieved_sources,
                   param=param)),
                 question_id=q_.question_id,
                 question=q_.question,
                 retrieved_sources=q_.retrieved_sources))
            print()
        st_result = StudentSearchResultsAndAnswer(
            k=param.k,
            search_results=answer_query_res)

        json_string = RootModel(st_result).model_dump_json(indent=2)
        file_path = Path(param.save_directory) / "dataset_docs_public.json"

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

        # print(answer_query_res)

        i += 1
        # Record end time
        end_time = time.perf_counter()
        # Calculate duration
        duration = int(end_time - start_time)
        duration_all += duration
        print(f"Execution time: {duration // 60}:{duration % 60}"
              " (minutes:seconds)")
        print("-"*20)
        start_time = end_time

    if (i == 0):
        help_text = parser.format_help()
        print("No commands to run.\n", help_text)
    elif i > 1:
        print(f"Total execution time: {duration_all // 60}:{duration_all % 60}"
              " (minutes:seconds)")


if __name__ == "__main__":
    main()
