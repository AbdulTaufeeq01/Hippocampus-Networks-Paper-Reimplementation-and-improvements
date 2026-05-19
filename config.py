"""
Configuration and constants for the experiment.
Modify these to customize experiment behavior.
"""

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# LLM backend: "ollama", "openai", or "local"
LLM_BACKEND = "ollama"

# Ollama configuration
OLLAMA_MODEL = "llama2"
OLLAMA_HOST = "http://localhost:11434"

# OpenAI configuration
OPENAI_API_KEY = None  # Set your API key here or via env variable
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.7

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

# Embedding model (Hugging Face model name)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Embedding dimension (depends on model)
EMBEDDING_DIM = 384

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

# Top-k chunks to retrieve
TOP_K = 5

# Similarity threshold for pattern separation (Condition B)
PATTERN_SEP_THRESHOLD = 0.85

# Number of hops for graph traversal (Condition C)
GRAPH_HOPS = 2

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Test dataset
NUM_DOCUMENTS = 10
NUM_QA_PAIRS = 20
NUM_SIMILAR_FACT_PAIRS = 10
NUM_CONTROL_PAIRS = 10

# Output directory
RESULTS_DIR = "results"

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# LOGGING
# ============================================================================

# Log level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_LEVEL = "INFO"

# Log file (None = only console)
LOG_FILE = None

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================

# Enable timing profiling
PROFILE_TIMING = True

# Save intermediate retrieval results
SAVE_RETRIEVALS = True

# Number of experiment runs (for statistical analysis)
NUM_RUNS = 1

# Use GPU if available
USE_GPU = True

# Batch size for embedding
BATCH_SIZE = 32
