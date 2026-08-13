*This project has been created as part of the 42 curriculum by obachuri.*

---

# RAG against the machine - Retrieval-Augmented Generation

---

## Description

Retrieval-Augmented Generation 
Index -> Retrieve -> Generate answer

## Instructions

### Installation

**Python 3.10+** and **uv** must be installed in avance.

```bash
make install
```

### Execution

```bash
# Index the repository
uv run python -m src index --max_chunk_size=2000

# Search a single query
uv run python -m src search "How to configure OpenAI server?" --k=10

# Answer a single question
uv run python -m src answer "How to configure OpenAI server?" --k=10

# Search over a whole dataset and write a StudentSearchResults JSON file.
uv run python -m src search_dataset --dataset_path <path> --k <int> --save_directory <dir>

# Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file.
uv run python -m src answer_dataset --student_search_results_path <path> --save_directory <dir>

# Evaluate
uv run python -m src evaluate --student_search_results_path <path> --dataset_path <path>

# Start Http API servis (http://127.0.0.1:8000/docs)
uvicorn app:app --host 127.0.0.1 --port 8000 --reload

```

## Algorithm and implementation

### System architecture

```
Source files
       │
       ▼
  ┌──────────┐    ┌──────────────┐
  │ Ingestor │───▶│   Chunker    │ (Python AST / Markdown / Text)
  └──────────┘    └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        ┌─────────────┐      ┌──────────────┐
        │ BM25 Index  │      │ Vector Index │
        │  (bm25s)    │      │ (MiniLM-L6)  │
        └──────┬──────┘      └──────┬───────┘
               │        RRF Fusion  │
               └─────────┬──────────┘
                         ▼
                  ┌─────────────┐
                  │  Retriever  │
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │  Generator  │ (Qwen3-0.6B)
                  └─────────────┘
```


### Chunking strategy
- Python files: AST-aware splitting (top level)
- Markdown files: Heading-aware recursive splitting (#, ##, ### separators)
- Other as files - index only text file with file-type-specific separators 
- Maximum chunk size 2000 character (parameter: --max_chunk_size)

### Retrieval method


### Performance analysis

### Design decisions

### Challenges faced

-  Library **BM25S** can`t cteate index for big dataset ( RAM > size of dataset * 5 )
-  Recommended semantic embeddings model all-MiniLM-L6-v2 work only with 256 tokens (about 1000 characters)
   and with default chunk size (--max_chunk_size=2000) work realy bad.

### Example usage

```bash
# index (only BM25)
uv run python -m src index --retrievemode BM25

# Search by dataset (docs)
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions --retrievemode BM25
# Search by dataset (code)
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions --retrievemode BM25

# evaluate (docs)
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
# evaluate (code)
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json

# evaluate via moulinette (docs)
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
# evaluate via moulinette (code)
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 10 --max_context_length 2000

# Generate answers for a dataset
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions

```

## Resources

- [Prompt stucture and Control Tokens for Qwen3 LLM](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- [HuggingFace description for Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Pydantic Docs](https://docs.pydantic.dev/latest/)
- [BM25S Python library on github](https://github.com/xhluca/bm25s)


### AI Usage
Tools Used: ChatGPT (GPT-4)

AI was used to conceptual understanding and to structuring this README to meet subject requirements.

## License

Part of the 42 curriculum project.