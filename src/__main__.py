import argparse
import os
import sys
import time


# path_to_file = os.path.dirname(__file__)
# sys.path.append(path_to_file)

from src.r_index import r_index_bm25
from src.r_chunk import r_chunking

from src.r_data_model import MinimalSource


commands = {"chunk": "chunk",
            "index": "index",
            "search": "search <query>",
            "search_dataset": "search_dataset",
            "answer": "answer <query>",
            "answer_dataset": "answer_dataset",
            "evaluate": "evaluate"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG against the machine - Retrieval-Augmented Generation "
        '(uv run python -m src <command> [options])'
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
        print(f"Directory with raw data not found: '{param.data_raw_path}'",
              file=sys.stderr)
        sys.exit(1)

    if (param.max_chunk_size < 200):
        print(f"!!! max_chunk_size to small ({param.max_chunk_size}). "
              "Will be used max_chunk_size = 200")
        param.max_chunk_size = 200

    if (param.min_chunk_size < 100):
        print(f"!!! min_chunk_size to small ({param.max_chunk_size}). "
              "Will be used min_chunk_size = 100")
        param.min_chunk_size = 100

    if (param.min_chunk_size >= param.max_chunk_size):
        print("!!! min_chunk_size must be smaller than max_chunk_size. "
              "Will be used min_chunk_size = 100")
        param.min_chunk_size = 100

    # if not os.path.exists(param.functions_definition):
    #     print(f"File not found: '{param.functions_definition}'",
    #           file=sys.stderr)
    #     sys.exit(1)
    # if not os.path.exists(param.input):
    #     print(f"File not found: '{param.input}'",
    #           file=sys.stderr)
    #     sys.exit(1)

    print("-"*20)
    print(param)
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
                    print(i, query)
                    if len(query) < 2:
                        raise IndexError
                    search_query.append(query)
                except Exception:
                    print('Error: Parameter <query> not set or to short! \n'
                          'The <query> parameter '
                          'must be specified for the "search" commands.',
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
                          'must be specified for the "answer" commands.',
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
    chunks: list[MinimalSource] = []
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

    if (i == 0):
        help_text = parser.format_help()
        print("No commands to run.\n", help_text)
    elif i > 1:
        print(f"Total execution time: {duration_all // 60}:{duration_all % 60}"
              " (minutes:seconds)")


if __name__ == "__main__":
    main()
