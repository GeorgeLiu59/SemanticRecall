#!/usr/bin/env python3
"""
Simple test script for the semantic video understanding system.
"""

import torch
from semantic_video import SemanticVideoProcessor


def test_basic_functionality():
    """Test basic system functionality."""
    print("Testing semantic video understanding system...")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize processor with minimal settings
    processor = SemanticVideoProcessor(
        video_path="Bee_Movie_Cropped.mp4",
        num_clusters=4,  # Small number for testing
        memory_size=100,
        clustering_method="kmeans",
        num_experts=4,
        device=device
    )
    
    # Test scene detection
    print("\n1. Testing scene detection...")
    boundaries = processor.detect_scene_boundaries()
    print(f"Detected {len(boundaries)} scenes")
    
    # Test embedding extraction
    print("\n2. Testing embedding extraction...")
    embeddings = processor.extract_scene_embeddings()
    print(f"Extracted {len(embeddings)} embeddings")
    
    # Test clustering
    print("\n3. Testing semantic clustering...")
    clusters = processor.perform_semantic_clustering()
    print(f"Created {len(set(clusters.values()))} clusters")
    
    # Test memory building
    print("\n4. Testing memory hierarchy...")
    memory = processor.build_semantic_memory()
    stats = memory.get_memory_statistics()
    print(f"Memory built: {stats['total_slots']} slots, {stats['total_scenes']} scenes")
    
    # Test querying
    print("\n5. Testing semantic querying...")
    results = processor.query_semantic_memory("person", top_k=2)
    print(f"Query 'person' returned {len(results)} results")
    
    print("\n✅ All basic tests passed!")


if __name__ == "__main__":
    test_basic_functionality() 