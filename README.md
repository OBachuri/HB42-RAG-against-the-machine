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
make run
```

Usage:
```bash
# Index the repository
uv run python -m src index --max_chunk_size=2000

# Search a single query
uv run python -m src search "How to configure OpenAI server?" --k=10

# Answer a single question
uv run python -m src answer "How to configure OpenAI server?" --k=10
```


## Resources

- [Prompt stucture and Control Tokens for Qwen3 LLM](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- [HuggingFace description for Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Pydantic Docs](https://docs.pydantic.dev/latest/)
- [BM25S Python library on github](https://github.com/xhluca/bm25s)


## Algorithm and implementation

### System architecture

### Chunking strategy



### Retrieval method

### Performance analysis

### Design decisions

### Challenges faced

-  Library **BM25S** can`t cteate index for big dataset ( RAM > size of dataset * 5 )

### Example usage


### AI Usage
Tools Used: ChatGPT (GPT-4)

AI was used to conceptual understanding and to structuring this README to meet subject requirements.

## License

Part of the 42 curriculum project.