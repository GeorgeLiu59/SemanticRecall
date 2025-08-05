"""
Video Processor: Main pipeline for semantically-driven cross-temporal video understanding
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import os
import pickle
from tqdm import tqdm
import time

from .scene_encoder import SceneEncoder
from .semantic_clustering import SemanticClustering
from .memory_hierarchy import MemoryHierarchy
from .moe_experts import MOEExperts
from .scene_retrieval import SceneRetrieval


class SemanticVideoProcessor:
    """
    Main processor for semantically-driven cross-temporal video understanding.
    
    This class orchestrates the entire pipeline from video input to semantic
    memory construction and cross-temporal reasoning.
    """
    
    def __init__(self, 
                 video_path: str,
                 num_clusters: int = 10,
                 memory_size: int = 1000,
                 clustering_method: str = "kmeans",
                 num_experts: int = 10,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 **kwargs):
        """
        Initialize the semantic video processor.
        
        Args:
            video_path: Path to the input video file
            num_clusters: Number of semantic clusters
            memory_size: Maximum size of memory hierarchy
            clustering_method: Clustering algorithm to use
            num_experts: Number of MOE experts
            device: Device to run on
            **kwargs: Additional configuration parameters
        """
        self.video_path = video_path
        self.num_clusters = num_clusters
        self.memory_size = memory_size
        self.clustering_method = clustering_method
        self.num_experts = num_experts
        self.device = device
        
        # Initialize components
        self.scene_encoder = SceneEncoder(device=device, **kwargs)
        self.semantic_clustering = SemanticClustering(
            method=clustering_method,
            n_clusters=num_clusters,
            **kwargs
        )
        self.memory_hierarchy = MemoryHierarchy(
            memory_size=memory_size,
            **kwargs
        )
        self.moe_experts = MOEExperts(
            num_experts=num_experts,
            device=device,
            **kwargs
        )
        
        # Initialize scene retrieval after memory hierarchy is built
        self.scene_retrieval = None
        
        # Processing state
        self.is_processed = False
        self.scene_boundaries = []
        self.scene_embeddings = {}
        self.scene_to_cluster = {}
        self.scene_metadata = {}
        
        # Performance tracking
        self.processing_times = {}
    
    def detect_scene_boundaries(self, 
                              threshold: float = 30.0,
                              min_scene_length: float = 1.0) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries in the video.
        
        Args:
            threshold: Threshold for scene change detection
            min_scene_length: Minimum scene length in seconds
            
        Returns:
            List of (start_time, end_time) tuples for each scene
        """
        print("Detecting scene boundaries...")
        start_time = time.time()
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        scene_boundaries = []
        prev_frame = None
        scene_start = 0.0
        
        for frame_idx in tqdm(range(0, total_frames, 30), desc="Processing frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            current_time = frame_idx / fps
            
            if prev_frame is not None:
                # Compute frame difference
                diff = cv2.absdiff(prev_frame, frame)
                mean_diff = np.mean(diff)
                
                # Check for scene change
                if mean_diff > threshold:
                    scene_end = current_time
                    
                    # Only add scene if it's long enough
                    if scene_end - scene_start >= min_scene_length:
                        scene_boundaries.append((scene_start, scene_end))
                    
                    scene_start = scene_end
            
            prev_frame = frame.copy()
        
        # Add final scene
        if duration - scene_start >= min_scene_length:
            scene_boundaries.append((scene_start, duration))
        
        cap.release()
        
        self.scene_boundaries = scene_boundaries
        self.processing_times['scene_detection'] = time.time() - start_time
        
        print(f"Detected {len(scene_boundaries)} scenes")
        return scene_boundaries
    
    def extract_scene_embeddings(self) -> Dict[int, np.ndarray]:
        """
        Extract CLIP embeddings for all detected scenes.
        
        Returns:
            Dictionary mapping scene_id to CLIP embedding
        """
        print("Extracting scene embeddings...")
        start_time = time.time()
        
        if not self.scene_boundaries:
            self.detect_scene_boundaries()
        
        scene_embeddings = self.scene_encoder.extract_scene_embeddings(
            self.video_path, self.scene_boundaries
        )
        
        self.scene_embeddings = scene_embeddings
        self.processing_times['embedding_extraction'] = time.time() - start_time
        
        print(f"Extracted embeddings for {len(scene_embeddings)} scenes")
        return scene_embeddings
    
    def perform_semantic_clustering(self) -> Dict[int, int]:
        """
        Perform semantic clustering of scenes.
        
        Returns:
            Dictionary mapping scene_id to cluster_id
        """
        print("Performing semantic clustering...")
        start_time = time.time()
        
        if not self.scene_embeddings:
            self.extract_scene_embeddings()
        
        scene_to_cluster = self.semantic_clustering.fit(self.scene_embeddings)
        
        # Evaluate clustering quality
        metrics = self.semantic_clustering.evaluate_clustering(self.scene_embeddings)
        print(f"Clustering metrics: {metrics}")
        
        self.scene_to_cluster = scene_to_cluster
        self.processing_times['clustering'] = time.time() - start_time
        
        return scene_to_cluster
    
    def build_semantic_memory(self) -> MemoryHierarchy:
        """
        Build the cross-temporal semantic memory hierarchy.
        
        Returns:
            Configured memory hierarchy
        """
        print("Building semantic memory hierarchy...")
        start_time = time.time()
        
        if not self.scene_to_cluster:
            self.perform_semantic_clustering()
        
        # Add scenes to memory hierarchy
        scene_to_slot = self.memory_hierarchy.add_scenes(
            self.scene_embeddings,
            self.scene_to_cluster,
            self.scene_metadata
        )
        
        # Initialize scene retrieval
        self.scene_retrieval = SceneRetrieval(self.memory_hierarchy)
        
        self.processing_times['memory_building'] = time.time() - start_time
        
        # Print memory statistics
        stats = self.memory_hierarchy.get_memory_statistics()
        print(f"Memory statistics: {stats}")
        
        return self.memory_hierarchy
    
    def train_moe_experts(self, 
                         num_epochs: int = 100,
                         batch_size: int = 32,
                         learning_rate: float = 1e-3) -> MOEExperts:
        """
        Train the MOE expert networks.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            
        Returns:
            Trained MOE experts
        """
        print("Training MOE experts...")
        start_time = time.time()
        
        if not self.scene_to_cluster:
            self.perform_semantic_clustering()
        
        self.moe_experts.train_experts(
            self.scene_embeddings,
            self.scene_to_cluster,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        self.processing_times['moe_training'] = time.time() - start_time
        
        return self.moe_experts
    
    def process_video(self, 
                     train_experts: bool = True,
                     save_results: bool = True,
                     output_dir: str = "results") -> Dict[str, Any]:
        """
        Process the entire video pipeline.
        
        Args:
            train_experts: Whether to train MOE experts
            save_results: Whether to save processing results
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing processing results and statistics
        """
        print(f"Processing video: {self.video_path}")
        total_start_time = time.time()
        
        # Run the complete pipeline
        self.detect_scene_boundaries()
        self.extract_scene_embeddings()
        self.perform_semantic_clustering()
        self.build_semantic_memory()
        
        if train_experts:
            self.train_moe_experts()
        
        self.is_processed = True
        total_time = time.time() - total_start_time
        
        # Compile results
        results = {
            'video_path': self.video_path,
            'num_scenes': len(self.scene_boundaries),
            'num_clusters': self.num_clusters,
            'scene_boundaries': self.scene_boundaries,
            'scene_embeddings': self.scene_embeddings,
            'scene_to_cluster': self.scene_to_cluster,
            'processing_times': self.processing_times,
            'total_processing_time': total_time,
            'memory_statistics': self.memory_hierarchy.get_memory_statistics(),
            'clustering_metrics': self.semantic_clustering.evaluate_clustering(self.scene_embeddings)
        }
        
        if train_experts:
            results['expert_specializations'] = self.moe_experts.get_expert_specializations()
        
        # Save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            self.save_results(results, output_dir)
        
        print(f"Video processing completed in {total_time:.2f} seconds")
        return results
    
    def query_semantic_memory(self, 
                            query: str,
                            top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """
        Query the semantic memory with a text description.
        
        Args:
            query: Text description to search for
            top_k: Number of results to return
            
        Returns:
            List of (scene_id, similarity_score, metadata) tuples
        """
        if not self.is_processed:
            raise ValueError("Video must be processed before querying")
        
        if self.scene_retrieval is None:
            raise ValueError("Scene retrieval not initialized")
        
        return self.scene_retrieval.search_by_description(
            query, self.scene_encoder, top_k
        )
    
    def reason_across_time(self, 
                          question: str,
                          max_scenes: int = 10) -> Dict[str, Any]:
        """
        Perform cross-temporal reasoning on the video.
        
        Args:
            question: Question about the video content
            max_scenes: Maximum number of relevant scenes to consider
            
        Returns:
            Dictionary containing reasoning results
        """
        if not self.is_processed:
            raise ValueError("Video must be processed before reasoning")
        
        print(f"Performing cross-temporal reasoning: {question}")
        
        # Encode the question
        question_embedding = self.scene_encoder.encode_text_query(question)
        
        # Retrieve relevant scenes
        relevant_scenes = self.scene_retrieval.retrieve_scenes(
            question_embedding, top_k=max_scenes
        )
        
        # Process scenes through MOE experts
        processed_scenes = []
        for scene_id, similarity, metadata in relevant_scenes:
            scene_embedding = self.scene_embeddings[scene_id]
            processed_embedding, routing_info = self.moe_experts.process_scene(scene_embedding)
            
            processed_scenes.append({
                'scene_id': scene_id,
                'similarity': similarity,
                'metadata': metadata,
                'routing_info': routing_info,
                'processed_embedding': processed_embedding
            })
        
        # Generate reasoning result
        reasoning_result = {
            'question': question,
            'relevant_scenes': processed_scenes,
            'num_scenes_considered': len(processed_scenes),
            'avg_similarity': np.mean([s['similarity'] for s in processed_scenes]),
            'expert_usage': self.moe_experts.get_expert_specializations()
        }
        
        return reasoning_result
    
    def get_cross_temporal_scenes(self, 
                                query_embedding: np.ndarray,
                                top_k: int = 10) -> List[Tuple[int, float, int]]:
        """
        Get scenes from across the video timeline that are semantically similar.
        
        Args:
            query_embedding: CLIP embedding to query with
            top_k: Number of scenes to return
            
        Returns:
            List of (scene_id, similarity_score, slot_id) tuples
        """
        if not self.is_processed:
            raise ValueError("Video must be processed before retrieval")
        
        return self.memory_hierarchy.get_cross_temporal_scenes(query_embedding, top_k)
    
    def visualize_results(self, 
                         save_dir: str = "visualizations",
                         show_plots: bool = True):
        """
        Generate visualizations of the processing results.
        
        Args:
            save_dir: Directory to save visualizations
            show_plots: Whether to display plots
        """
        if not self.is_processed:
            raise ValueError("Video must be processed before visualization")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Clustering visualization
        self.semantic_clustering.visualize_clusters(
            self.scene_embeddings,
            save_path=os.path.join(save_dir, "clustering.png") if save_dir else None
        )
        
        # Memory distribution visualization
        self.memory_hierarchy.visualize_memory_distribution(
            save_path=os.path.join(save_dir, "memory_distribution.png") if save_dir else None
        )
        
        # Expert usage visualization
        if self.moe_experts.is_trained:
            self.moe_experts.visualize_expert_usage(
                save_path=os.path.join(save_dir, "expert_usage.png") if save_dir else None
            )
        
        if not show_plots:
            import matplotlib.pyplot as plt
            plt.close('all')
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save processing results to disk."""
        # Save main results
        results_file = os.path.join(output_dir, "processing_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Save memory hierarchy
        memory_file = os.path.join(output_dir, "memory_hierarchy.pkl")
        self.memory_hierarchy.save_memory(memory_file)
        
        # Save MOE experts if trained
        if self.moe_experts.is_trained:
            experts_file = os.path.join(output_dir, "moe_experts.pth")
            self.moe_experts.save_experts(experts_file)
        
        # Save clustering model
        clustering_file = os.path.join(output_dir, "clustering_model.pkl")
        with open(clustering_file, 'wb') as f:
            pickle.dump(self.semantic_clustering, f)
        
        print(f"Results saved to {output_dir}")
    
    def load_results(self, output_dir: str):
        """Load processing results from disk."""
        # Load main results
        results_file = os.path.join(output_dir, "processing_results.pkl")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        # Restore state
        self.scene_boundaries = results['scene_boundaries']
        self.scene_embeddings = results['scene_embeddings']
        self.scene_to_cluster = results['scene_to_cluster']
        self.processing_times = results['processing_times']
        
        # Load memory hierarchy
        memory_file = os.path.join(output_dir, "memory_hierarchy.pkl")
        self.memory_hierarchy.load_memory(memory_file)
        
        # Load MOE experts if available
        experts_file = os.path.join(output_dir, "moe_experts.pth")
        if os.path.exists(experts_file):
            self.moe_experts.load_experts(experts_file)
        
        # Load clustering model
        clustering_file = os.path.join(output_dir, "clustering_model.pkl")
        with open(clustering_file, 'rb') as f:
            self.semantic_clustering = pickle.load(f)
        
        # Initialize scene retrieval
        self.scene_retrieval = SceneRetrieval(self.memory_hierarchy)
        
        self.is_processed = True
        print(f"Results loaded from {output_dir}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        if not self.is_processed:
            return {}
        
        stats = {
            'video_path': self.video_path,
            'num_scenes': len(self.scene_boundaries),
            'num_clusters': self.num_clusters,
            'processing_times': self.processing_times,
            'total_processing_time': sum(self.processing_times.values()),
            'memory_statistics': self.memory_hierarchy.get_memory_statistics(),
            'clustering_metrics': self.semantic_clustering.evaluate_clustering(self.scene_embeddings)
        }
        
        if self.scene_retrieval:
            stats['retrieval_statistics'] = self.scene_retrieval.get_retrieval_statistics()
        
        if self.moe_experts.is_trained:
            stats['expert_specializations'] = self.moe_experts.get_expert_specializations()
        
        return stats 