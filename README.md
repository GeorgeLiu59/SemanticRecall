# Semantically-Driven Cross-Temporal Memory Hierarchies for Long Video Understanding

## Research Overview

This project implements a novel approach to long video understanding that groups video segments semantically rather than temporally, inspired by how humans categorize visual memories. The system uses CLIP embeddings to identify semantically similar scenes across the entire video timeline and builds memory hierarchies organized by scene semantics.

## Key Contributions

1. **Cross-Temporal Scene Clustering**: First approach to group semantically similar scenes across entire video timeline using CLIP embeddings
2. **Scene-Semantic Memory Architecture**: Novel memory hierarchy organized by scene semantics rather than temporal proximity
3. **MOE Scene Specialization**: Expert networks specialized by semantic scene categories rather than temporal segments
4. **Semantic Scene Retrieval**: Efficient retrieval of relevant scene memories based on semantic similarity

## Architecture

```
Input Video → Scene Segmentation → CLIP Encoding → Semantic Clustering → Memory Hierarchy → MOE Experts → Output
```

### Components

- **Scene Encoder**: Extracts CLIP embeddings for video segments
- **Semantic Clustering**: Groups scenes by semantic similarity using K-means/spectral clustering
- **Cross-Temporal Memory**: Memory banks indexed by semantic scene clusters
- **MOE Routing**: Routes different scene types to specialized expert networks
- **Scene Retrieval**: Retrieves relevant scenes from memory during processing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from semantic_video import SemanticVideoProcessor

# Initialize processor
processor = SemanticVideoProcessor(
    video_path="path/to/video.mp4",
    num_clusters=10,
    memory_size=1000
)

# Process video and build semantic memory
memory_hierarchy = processor.build_semantic_memory()

# Query semantic memory
results = processor.query_semantic_memory("person walking")
```

### Advanced Usage

```python
# Custom clustering
processor = SemanticVideoProcessor(
    video_path="path/to/video.mp4",
    clustering_method="spectral",
    num_clusters=15,
    memory_size=2000
)

# Train MOE experts
processor.train_moe_experts()

# Cross-temporal reasoning
answer = processor.reason_across_time("What happened to the main character?")
```

## Project Structure

```
├── semantic_video/
│   ├── __init__.py
│   ├── scene_encoder.py      # CLIP-based scene encoding
│   ├── semantic_clustering.py # Scene clustering algorithms
│   ├── memory_hierarchy.py   # Cross-temporal memory system
│   ├── moe_experts.py        # Mixture of Experts implementation
│   ├── scene_retrieval.py    # Semantic scene retrieval
│   └── video_processor.py    # Main processing pipeline
├── experiments/
│   ├── baselines.py          # Baseline implementations
│   ├── evaluation.py         # Evaluation metrics
│   └── ablation_studies.py   # Ablation experiments
├── utils/
│   ├── video_utils.py        # Video processing utilities
│   ├── visualization.py      # Visualization tools
│   └── metrics.py           # Evaluation metrics
├── configs/
│   └── default_config.yaml   # Configuration files
└── examples/
    ├── basic_usage.py        # Basic usage examples
    └── advanced_usage.py     # Advanced usage examples
```
