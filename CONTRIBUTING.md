# Contributing Guide

Thank you for your interest in contributing to the Hippocampus-Inspired Memory project!

## How to Contribute

### 1. Report Bugs

If you find a bug, please open an issue with:
- A clear title and description
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS

Example:
```
Title: Condition B crashes with unicode characters in documents

Steps to reproduce:
1. Add a document with unicode characters (e.g., "Café")
2. Run experiment.py
3. Observe: TypeError in similarity computation

Expected: Handle unicode gracefully
Actual: Crashes with TypeError
```

### 2. Suggest Enhancements

Open an issue for feature requests:
- Clear description of the enhancement
- Motivation: why is this useful?
- Possible implementation approaches (if you have ideas)

Example:
```
Title: Support for multi-hop reasoning in graph retrieval

This would allow deeper exploration of entity relationships
and improve retrieval quality for complex multi-step questions.
```

### 3. Submit Code

Before submitting a pull request:

1. **Fork the repository**
   ```bash
   git clone https://github.com/[your-fork]/hippocampus-inspired-memory-nn.git
   cd hippocampus-inspired-memory-nn
   ```

2. **Create a branch for your feature**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow PEP 8 style guide
   - Add comments for complex logic
   - Keep functions under 50 lines when possible

4. **Test your changes**
   ```bash
   python experiment.py
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature X: description of changes"
   ```

6. **Push and open a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### Python

- **Style:** PEP 8
- **Docstrings:** NumPy-style docstrings
- **Line length:** 88 characters (Black formatter)

Example:
```python
def retrieve_chunks(query: str, collection, k: int = 5) -> List[str]:
    """
    Retrieve top-k chunks similar to a query.
    
    Parameters
    ----------
    query : str
        The query string to search for.
    collection : chromadb.Collection
        ChromaDB collection to retrieve from.
    k : int, optional
        Number of chunks to retrieve (default: 5).
    
    Returns
    -------
    List[str]
        Top-k retrieved chunks, sorted by relevance.
    """
    embeddings = embedder.embed([query])
    results = collection.query(query_embeddings=embeddings, n_results=k)
    return results['documents'][0] if results else []
```

## Project Structure

```
hippocampus-inspired-memory-nn/
├── experiment.py              # Main experiment (modifiable)
├── TECHNICAL_REPORT.md        # Report (modifiable but be careful)
├── README.md                  # Documentation
├── QUICK_START.md             # Quick start guide
├── data/
│   └── corpus.md              # Test dataset description
├── results/
│   └── experiment_results.json # Sample results
└── requirements.txt           # Dependencies
```

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies including dev tools
pip install -r requirements.txt
pip install black flake8 mypy pytest

# Format code
black experiment.py

# Lint code
flake8 experiment.py

# Type check
mypy experiment.py

# Run tests (if you add any)
pytest
```

## Areas for Contribution

### High Priority

1. **Extend Test Dataset:** Add more biomedical QA pairs and documents
2. **Integrate Real LLMs:** Implement actual LLM calls (Ollama, OpenAI API)
3. **Automatic Entity Extraction:** NER and relation extraction for Condition C
4. **Human Evaluation:** Framework for human assessment of answer quality
5. **Benchmarking:** Test on larger, real-world knowledge graphs

### Medium Priority

1. **Visualization:** Plot results and comparisons
2. **Configuration:** YAML config files for experiment parameters
3. **Logging:** Detailed logging for debugging
4. **Documentation:** Extend TECHNICAL_REPORT with more examples
5. **Performance:** Optimize retrieval speed for large datasets

### Lower Priority

1. **Multimodal Embeddings:** Support for image + text embeddings
2. **Cross-Modal Binding:** Implement cross-modal memory consolidation
3. **Continual Learning:** Framework for learning new facts without forgetting
4. **Interactive Demo:** Web UI for exploring retrieval conditions

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows PEP 8
- [ ] All new functions have docstrings
- [ ] Experiment still runs without errors: `python experiment.py`
- [ ] Results are stable (run multiple times)
- [ ] Changes are clearly documented in commit messages
- [ ] No unnecessary dependencies added
- [ ] README or QUICK_START.md updated if needed

## Questions?

- Open an issue for questions
- Refer to `TECHNICAL_REPORT.md` for neuroscience background
- Check `README.md` for architecture details
- See `QUICK_START.md` for getting started

---

**Happy contributing!** We appreciate all contributions to advancing hippocampus-inspired memory research. 🧠
