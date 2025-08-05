#!/usr/bin/env python3
"""
Main experiment runner for the semantic video understanding system.
"""

import torch
import yaml
import argparse
from semantic_video import SemanticVideoProcessor
from utils.video_utils import get_video_info


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run semantic video understanding experiment")
    parser.add_argument("--video", default="Bee_Movie_Cropped.mp4", help="Path to video file")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Config file path")
    parser.add_argument("--clusters", type=int, default=8, help="Number of clusters")
    parser.add_argument("--experts", type=int, default=8, help="Number of MOE experts")
    parser.add_argument("--no-train", action="store_true", help="Skip MOE training")
    parser.add_argument("--output", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get video info
    print(f"Video: {args.video}")
    video_info = get_video_info(args.video)
    print(f"Duration: {video_info['duration']:.1f}s, FPS: {video_info['fps']:.1f}")
    
    # Initialize processor
    processor = SemanticVideoProcessor(
        video_path=args.video,
        num_clusters=args.clusters,
        memory_size=config['memory']['memory_size'],
        clustering_method=config['clustering']['method'],
        num_experts=args.experts,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Process video
    print("\nProcessing video...")
    results = processor.process_video(
        train_experts=not args.no_train,
        save_results=True,
        output_dir=args.output
    )
    
    # Print results
    print(f"\nProcessing completed!")
    print(f"Scenes detected: {results['num_scenes']}")
    print(f"Clusters: {results['num_clusters']}")
    print(f"Total time: {results['total_processing_time']:.1f}s")
    
    # Test queries
    test_queries = ["person", "outdoor", "action", "dialogue"]
    print(f"\nTesting queries...")
    for query in test_queries:
        results = processor.query_semantic_memory(query, top_k=2)
        print(f"'{query}': {len(results)} results")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main() 