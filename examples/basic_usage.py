"""Basic usage example for the semantic video understanding system."""

from semantic_video import SemanticVideoProcessor
import torch


def main():
    # Initialize processor
    processor = SemanticVideoProcessor(
        video_path="Bee_Movie_Cropped.mp4",
        num_clusters=8,
        memory_size=500,
        clustering_method="kmeans",
        num_experts=8
    )
    
    # Process video (this will take some time)
    print("Processing video...")
    results = processor.process_video(
        train_experts=True,
        save_results=True,
        output_dir="results"
    )
    
    # Query semantic memory
    print("\nQuerying semantic memory...")
    queries = [
        "person talking",
        "outdoor scene", 
        "close-up shot",
        "action sequence"
    ]
    
    for query in queries:
        results = processor.query_semantic_memory(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for scene_id, similarity, metadata in results:
            print(f"  Scene {scene_id}: similarity={similarity:.3f}, cluster={metadata['cluster_id']}")
    
    # Cross-temporal reasoning
    print("\nPerforming cross-temporal reasoning...")
    reasoning_result = processor.reason_across_time(
        "What are the main characters doing?",
        max_scenes=5
    )
    
    print(f"Reasoning result: {reasoning_result['num_scenes_considered']} scenes considered")
    print(f"Average similarity: {reasoning_result['avg_similarity']:.3f}")
    
    # Generate visualizations
    processor.visualize_results(save_dir="visualizations")


if __name__ == "__main__":
    main() 