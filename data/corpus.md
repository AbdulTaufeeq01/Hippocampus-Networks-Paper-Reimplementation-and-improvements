# Biomedical Test Corpus

This file contains the test dataset used in the experiment: 10 biomedical documents and 20 question-answer pairs designed to test memory faithfulness and pattern separation.

## Documents

```json
{
  "documents": [
    {
      "id": 0,
      "text": "Drug A activates protein X, leading to pathway P1 activation and increased cell proliferation."
    },
    {
      "id": 1,
      "text": "Drug B activates protein X, but through a distinct mechanism involving pathway P2."
    },
    {
      "id": 2,
      "text": "Experiment 1 showed 50% efficacy of Drug A in cell viability assays over 48 hours."
    },
    {
      "id": 3,
      "text": "Experiment 2 showed 55% efficacy of Drug B in the same assay, outperforming Drug A by 5%."
    },
    {
      "id": 4,
      "text": "Disease Y is associated with dysregulation of pathway P1, commonly treated with Drug A."
    },
    {
      "id": 5,
      "text": "Disease Z is associated with dysregulation of pathway P1 but also involves pathway P2."
    },
    {
      "id": 6,
      "text": "Protein X is highly expressed in neurons and plays a role in synaptic plasticity."
    },
    {
      "id": 7,
      "text": "Protein Y is a binding partner of Protein X and is involved in signal transduction."
    },
    {
      "id": 8,
      "text": "The combination of Drug A and Drug B showed synergistic effects in preliminary studies."
    },
    {
      "id": 9,
      "text": "Long-term administration of Drug A resulted in 70% reduction in disease symptoms in a cohort of 50 patients."
    }
  ],
  "document_count": 10,
  "total_tokens": "~2000"
}
```

## Question-Answer Pairs

### Similar-Fact Pairs (Pattern Separation Challenge)

These 10 pairs are designed to distinguish between similar entities and relations:

| ID | Question | Correct Answer | Ground Truth Doc IDs |
|----|----------|---|---|
| 1 | What drug activates protein X through pathway P1? | Drug A | [0] |
| 2 | What drug activates protein X through pathway P2? | Drug B | [1] |
| 3 | What was the efficacy of Drug A in Experiment 1? | 50% | [2] |
| 4 | What was the efficacy of Drug B in Experiment 2? | 55% | [3] |
| 5 | Which disease is associated with pathway P1 and treated with Drug A? | Disease Y | [4] |
| 6 | Which disease is associated with pathway P2? | Disease Z | [5] |
| 7 | What pathways are involved in Disease Z? | P1 and P2 | [5] |
| 8 | Did Drug A or Drug B outperform in efficacy? | Drug B | [3] |
| 9 | What is the long-term efficacy of Drug A on disease symptoms? | 70% | [9] |
| 10 | How many patients were in the Drug A long-term study? | 50 | [9] |

### Control Pairs (Baseline, Unambiguous)

These 10 pairs have unique, non-overlapping correct answers:

| ID | Question | Correct Answer | Ground Truth Doc IDs |
|----|----------|---|---|
| 11 | What is Protein X's role? | synaptic plasticity | [6] |
| 12 | What is Protein Y's role? | signal transduction | [7] |
| 13 | Who is a binding partner of Protein X? | Protein Y | [7] |
| 14 | Did the combination of Drug A and Drug B show positive effects? | synergistic | [8] |
| 15 | How long was the Drug A long-term study? | long-term | [9] |
| 16 | What does Drug A activate? | protein X | [0] |
| 17 | What does Drug B activate? | protein X | [1] |
| 18 | Which pathway is involved in Disease Y treatment? | P1 | [4] |
| 19 | Is Protein X expressed in neurons? | yes | [6] |
| 20 | Describe the efficacy comparison between Drug A and Drug B. | Drug B 55%, Drug A 50% | [2, 3] |

## Metrics Explained

### Faithfulness (F)

Measures whether the LLM's answer is grounded in retrieved chunks and does not contradict them.

- Formula: `(# answers matching correct entity) / (# total QA pairs)`
- Expected range: 0.0–1.0
- Expected improvement: Pattern Separation should improve by ~7%, HippoRAG by ~6% further.

### Precision@1 (P@1)

Measures whether the top-ranked retrieved chunk is the ground-truth chunk for that QA pair.

- Formula: `(# top-1 chunks correct) / (# total QA pairs)`
- Expected range: 0.0–1.0

### Confusion Rate (C)

For the 10 distinguishing pairs, measures how often the LLM confuses two similar entities or relations.

- Formula: `(# confused answers) / (# distinguishing pairs)`
- Examples of confusion:
  - Answering "Drug B" when asked about Drug A's effects
  - Answering "55%" when asked about Drug A's efficacy (should be 50%)
  - Answering "Disease Z" when asked about Disease Y
- Expected range: 0.0–1.0
- Expected improvement: HippoRAG should reduce confusion by ~40%.

---

## Why This Corpus?

- **Small scale:** 10 documents, 20 pairs — fast to run, easy to debug.
- **Challenging:** 50% of QA pairs require distinguishing between similar facts (pattern separation task).
- **Domain:** Biomedical (drugs, proteins, diseases, pathways) — reflects real-world biomedical knowledge graph challenges.
- **Balanced:** Mix of simple direct facts and complex multi-hop reasoning.

---

## Extending the Corpus

To add more documents:

1. Add text to `DOCUMENTS` in `experiment.py`.
2. Add corresponding QA pairs to `QA_PAIRS`.
3. Update ground truth document indices.
4. If using Condition C (HippoRAG), update entities and edges in the knowledge graph.

Example:

```python
DOCUMENTS = [
    # Original documents...
    "New drug or fact here...",
]

QA_PAIRS = [
    # Original pairs...
    ("New question?", "new_answer", [10]),  # Document index 10
]
```

---

**Version:** 1.0 (May 2026)
