"""
Semantically-Driven Cross-Temporal Memory Hierarchies for Long Video Understanding

This package implements a novel approach to long video understanding that groups
video segments semantically rather than temporally, using CLIP embeddings and
cross-temporal memory hierarchies.
"""

from .video_processor import SemanticVideoProcessor
from .scene_encoder import SceneEncoder
from .semantic_clustering import SemanticClustering
from .memory_hierarchy import MemoryHierarchy
from .moe_experts import MOEExperts
from .scene_retrieval import SceneRetrieval

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    "SemanticVideoProcessor",
    "SceneEncoder", 
    "SemanticClustering",
    "MemoryHierarchy",
    "MOEExperts",
    "SceneRetrieval"
] 