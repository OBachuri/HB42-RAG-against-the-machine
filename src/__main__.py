import argparse
import os
import sys

# path_to_file = os.path.dirname(__file__)
# sys.path.append(path_to_file)

from src.r_chank import r_index


commands = {"index": "a_iii"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG against the machine - Retrieval-Augmented Generation"
        ""
    )

    parser.add_argument('the_task', help=str(set(commands.keys())))

    parser.add_argument(
        "--max_chunk_size",
        default=500,
        help="Max size of a chunk",
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
        help="KKK...",
        type=int,
    )

    param = parser.parse_args()

#    param._get_args()

    if (str(param.the_task).strip().lower() not in commands.keys()):
        print(f"Command not found: '{param.the_task}'.",
              " use: ", list(commands.keys()),
              file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(param.data_raw_path):
        print(f"Directory with raw data not found: '{param.data_raw_path}'",
              file=sys.stderr)
        sys.exit(1)

    if (param.max_chunk_size < 100):
        print(f"!!! max_chunk_size to small ({param.max_chunk_size}). "
              "Will be used max_chunk_size = 100")
        param.max_chunk_size = 100

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

    r_index(param)


if __name__ == "__main__":
    main()
