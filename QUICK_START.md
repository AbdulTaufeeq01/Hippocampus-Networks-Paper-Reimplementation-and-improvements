# Quick Start Guide

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/hippocampus-inspired-memory-nn.git
cd hippocampus-inspired-memory-nn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Run the Experiment

```bash
python experiment.py
```

Expected output:
```
================================================================================
HIPPOCAMPUS-INSPIRED MEMORY EXPERIMENT
Comparing Three Retrieval Conditions
================================================================================

Test Dataset: 10 documents, 20 QA pairs
(10 similar-fact pairs, 10 control pairs)

Running Condition A: Standard RAG...
✓ Condition A complete.

Running Condition B: Pattern Separation RAG...
✓ Condition B complete.

Running Condition C: HippoRAG-Style Graph Retrieval...
✓ Condition C complete.

--------------------------------------------------------------------------------
RESULTS TABLE
--------------------------------------------------------------------------------
Condition                      Faithfulness    Precision@1     Confusion Rate  
--------------------------------------------------------------------------------
A: Standard RAG                0.75            0.65            0.40
B: Pattern Separation RAG      0.82            0.70            0.25
C: HippoRAG (Graph+Vector)    0.88            0.78            0.15
--------------------------------------------------------------------------------

ANALYSIS
--------------------------------------------------------------------------------
Pattern Separation vs. Standard: +9.3% faithfulness improvement
HippoRAG vs. Standard: +17.3% faithfulness improvement
Graph-based retrieval reduces confusion by 63%
--------------------------------------------------------------------------------

Results saved to results/experiment_results.json
```

## 3. Read the Report

Open `TECHNICAL_REPORT.md` in any text editor or markdown viewer:

```bash
# On Linux/Mac
open TECHNICAL_REPORT.md

# On Windows
start TECHNICAL_REPORT.md
```

## 4. Project Structure

```
hippocampus-inspired-memory-nn/
├── TECHNICAL_REPORT.md          ← Research paper (5-6 pages)
├── experiment.py                 ← Main experiment script
├── requirements.txt              ← Dependencies
├── README.md                      ← Full documentation
├── QUICK_START.md                ← This file
├── LICENSE                        ← MIT License
├── setup.py                       ← Package setup
├── data/                          ← Dataset directory
│   └── corpus.json               ← Biomedical test corpus
└── results/                       ← Output directory
    └── experiment_results.json    ← Results from experiment.py
```

## 5. Extending the Experiment

### Add More Test Data

Edit `experiment.py` and expand `DOCUMENTS` and `QA_PAIRS`:

```python
DOCUMENTS = [
    # Add more biomedical facts here...
]

QA_PAIRS = [
    # Add more question-answer pairs here...
]
```

### Customize the Knowledge Graph

In `condition_c_hipporag_graph_retrieval()`, modify entities and edges:

```python
entities = ["Entity1", "Entity2", ...]  # Add your entities
edges = [("Entity1", "Entity2"), ...]   # Add relations
```

### Use Different LLMs

Modify the embedder in each condition:

```python
from sentence_transformers import SentenceTransformer

# Use a different model
embedder = SimpleEmbedder(model_name="all-mpnet-base-v2")
```

## 6. Key Files Explained

| File | Purpose |
|------|---------|
| `TECHNICAL_REPORT.md` | 5-6 page academic paper covering neuroscience, AI architectures, methodology, results, and discussion |
| `experiment.py` | Runnable Python script (< 250 lines); implements all three conditions with test dataset |
| `README.md` | Comprehensive project overview, neuroscience background, and architecture descriptions |
| `requirements.txt` | Python dependencies (chromadb, sentence-transformers, networkx, etc.) |

## 7. Troubleshooting

**ChromaDB not found:**
```bash
pip install chromadb
```

**Sentence-transformers downloading models:**
The first run will download embedding models (~500 MB). Be patient.

**Out of memory:**
Reduce the number of documents or QA pairs in `experiment.py`.

## 8. Next Steps

- Extend the experiment with your own biomedical data
- Implement consolidation mechanisms (simulating sleep-based replay)
- Integrate with a live LLM (Ollama, OpenAI API)
- Run on larger knowledge graphs (DrugBank, KEGG)

---

**Happy researching!** 🧠🤖
