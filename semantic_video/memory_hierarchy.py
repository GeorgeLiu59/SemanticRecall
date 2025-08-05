"""
Memory Hierarchy: Cross-temporal memory system organized by semantic scene clusters
"""

import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
from collections import defaultdict
import torch
import torch.nn as nn


class MemorySlot:
    """Represents a single memory slot in the semantic memory hierarchy."""
    
    def __init__(self, 
                 cluster_id: int,
                 scene_ids: List[int],
                 embeddings: List[np.ndarray],
                 metadata: Dict[str, Any] = None):
        """
        Initialize a memory slot.
        
        Args:
            cluster_id: Semantic cluster identifier
            scene_ids: List of scene IDs in this cluster
            embeddings: List of CLIP embeddings for scenes
            metadata: Additional metadata (timestamps, descriptions, etc.)
        """
        self.cluster_id = cluster_id
        self.scene_ids = scene_ids
        self.embeddings = embeddings
        self.metadata = metadata or {}
        
        # Compute cluster center
        if embeddings:
            self.center = np.mean(embeddings, axis=0)
            self.center = self.center / np.linalg.norm(self.center)  # Normalize
        else:
            self.center = None
        
        # Compute cluster statistics
        self.size = len(scene_ids)
        self.last_accessed = 0
        self.access_count = 0
    
    def add_scene(self, scene_id: int, embedding: np.ndarray, metadata: Dict = None):
        """Add a new scene to this memory slot."""
        self.scene_ids.append(scene_id)
        self.embeddings.append(embedding)
        
        # Update center
        if self.center is not None:
            self.center = np.mean(self.embeddings, axis=0)
            self.center = self.center / np.linalg.norm(self.center)
        else:
            self.center = embedding / np.linalg.norm(embedding)
        
        self.size += 1
        
        # Update metadata
        if metadata:
            for key, value in metadata.items():
                if key not in self.metadata:
                    self.metadata[key] = []
                self.metadata[key].append(value)
    
    def get_similarity(self, query_embedding: np.ndarray) -> float:
        """Compute similarity between query and cluster center."""
        if self.center is None:
            return 0.0
        return np.dot(query_embedding, self.center)
    
    def access(self):
        """Mark this memory slot as accessed."""
        self.access_count += 1
        self.last_accessed = 0  # Reset counter


class MemoryHierarchy:
    """
    Cross-temporal memory hierarchy organized by semantic scene clusters.
    
    This component stores and manages scene memories based on semantic similarity
    rather than temporal proximity, enabling efficient cross-temporal retrieval.
    """
    
    def __init__(self, 
                 memory_size: int = 1000,
                 embedding_dim: int = 512,
                 similarity_threshold: float = 0.7,
                 use_faiss: bool = True):
        """
        Initialize the memory hierarchy.
        
        Args:
            memory_size: Maximum number of memory slots
            embedding_dim: Dimension of CLIP embeddings
            similarity_threshold: Threshold for considering memories similar
            use_faiss: Whether to use FAISS for fast similarity search
        """
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.use_faiss = use_faiss
        
        # Memory storage
        self.memory_slots: Dict[int, MemorySlot] = {}
        self.scene_to_slot: Dict[int, int] = {}  # scene_id -> slot_id
        
        # FAISS index for fast similarity search
        if use_faiss:
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
            self.slot_ids = []  # Maps FAISS index to slot_id
        else:
            self.faiss_index = None
            self.slot_ids = None
        
        # Statistics
        self.total_scenes = 0
        self.total_slots = 0
    
    def add_scenes(self, 
                  scene_embeddings: Dict[int, np.ndarray],
                  scene_to_cluster: Dict[int, int],
                  scene_metadata: Optional[Dict[int, Dict]] = None) -> Dict[int, int]:
        """
        Add scenes to the memory hierarchy based on their cluster assignments.
        
        Args:
            scene_embeddings: Dictionary mapping scene_id to CLIP embedding
            scene_to_cluster: Dictionary mapping scene_id to cluster_id
            scene_metadata: Optional metadata for each scene
            
        Returns:
            Dictionary mapping scene_id to memory slot_id
        """
        scene_to_slot = {}
        
        # Group scenes by cluster
        cluster_scenes = defaultdict(list)
        for scene_id, cluster_id in scene_to_cluster.items():
            cluster_scenes[cluster_id].append(scene_id)
        
        # Create or update memory slots
        for cluster_id, scene_ids in cluster_scenes.items():
            slot_id = self._get_or_create_slot(cluster_id)
            
            for scene_id in scene_ids:
                embedding = scene_embeddings[scene_id]
                metadata = scene_metadata.get(scene_id, {}) if scene_metadata else {}
                
                self.memory_slots[slot_id].add_scene(scene_id, embedding, metadata)
                scene_to_slot[scene_id] = slot_id
                self.scene_to_slot[scene_id] = slot_id
                self.total_scenes += 1
        
        # Update FAISS index
        if self.use_faiss:
            self._update_faiss_index()
        
        return scene_to_slot
    
    def _get_or_create_slot(self, cluster_id: int) -> int:
        """Get existing slot for cluster or create new one."""
        # Check if slot already exists for this cluster
        for slot_id, slot in self.memory_slots.items():
            if slot.cluster_id == cluster_id:
                return slot_id
        
        # Create new slot
        slot_id = self.total_slots
        self.memory_slots[slot_id] = MemorySlot(cluster_id, [], [])
        self.total_slots += 1
        
        return slot_id
    
    def _update_faiss_index(self):
        """Update FAISS index with current memory slot centers."""
        if not self.use_faiss:
            return
        
        # Clear existing index
        self.faiss_index.reset()
        self.slot_ids = []
        
        # Add slot centers to index
        centers = []
        for slot_id, slot in self.memory_slots.items():
            if slot.center is not None:
                centers.append(slot.center.reshape(1, -1))
                self.slot_ids.append(slot_id)
        
        if centers:
            centers_array = np.vstack(centers)
            self.faiss_index.add(centers_array.astype('float32'))
    
    def query_semantic_memory(self, 
                            query_embedding: np.ndarray,
                            top_k: int = 5,
                            similarity_threshold: float = None) -> List[Tuple[int, float, MemorySlot]]:
        """
        Query semantic memory for similar scenes.
        
        Args:
            query_embedding: CLIP embedding to query with
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (slot_id, similarity_score, memory_slot) tuples
        """
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        results = []
        
        if self.use_faiss and self.faiss_index.ntotal > 0:
            # Use FAISS for fast similarity search
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            similarities, indices = self.faiss_index.search(
                query_norm.reshape(1, -1).astype('float32'), 
                min(top_k, self.faiss_index.ntotal)
            )
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity >= similarity_threshold:
                    slot_id = self.slot_ids[idx]
                    slot = self.memory_slots[slot_id]
                    slot.access()  # Mark as accessed
                    results.append((slot_id, float(similarity), slot))
        else:
            # Fallback to brute force search
            for slot_id, slot in self.memory_slots.items():
                similarity = slot.get_similarity(query_embedding)
                if similarity >= similarity_threshold:
                    slot.access()  # Mark as accessed
                    results.append((slot_id, similarity, slot))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
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
        similar_slots = self.query_semantic_memory(query_embedding, top_k=top_k)
        
        cross_temporal_scenes = []
        for slot_id, slot_similarity, slot in similar_slots:
            for scene_id in slot.scene_ids:
                # Compute individual scene similarity
                scene_embedding = slot.embeddings[slot.scene_ids.index(scene_id)]
                scene_similarity = np.dot(query_embedding, scene_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(scene_embedding)
                )
                cross_temporal_scenes.append((scene_id, scene_similarity, slot_id))
        
        # Sort by similarity and return top_k
        cross_temporal_scenes.sort(key=lambda x: x[1], reverse=True)
        return cross_temporal_scenes[:top_k]
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory hierarchy."""
        stats = {
            'total_slots': self.total_slots,
            'total_scenes': self.total_scenes,
            'avg_scenes_per_slot': self.total_scenes / max(1, self.total_slots),
            'memory_utilization': len(self.memory_slots) / self.memory_size,
            'slot_sizes': [],
            'access_counts': [],
            'cluster_distribution': defaultdict(int)
        }
        
        for slot in self.memory_slots.values():
            stats['slot_sizes'].append(slot.size)
            stats['access_counts'].append(slot.access_count)
            stats['cluster_distribution'][slot.cluster_id] += 1
        
        if stats['slot_sizes']:
            stats['min_slot_size'] = min(stats['slot_sizes'])
            stats['max_slot_size'] = max(stats['slot_sizes'])
            stats['avg_slot_size'] = np.mean(stats['slot_sizes'])
        
        if stats['access_counts']:
            stats['total_accesses'] = sum(stats['access_counts'])
            stats['avg_access_count'] = np.mean(stats['access_counts'])
        
        return stats
    
    def save_memory(self, filepath: str):
        """Save memory hierarchy to disk."""
        memory_data = {
            'memory_slots': self.memory_slots,
            'scene_to_slot': self.scene_to_slot,
            'total_scenes': self.total_scenes,
            'total_slots': self.total_slots,
            'memory_size': self.memory_size,
            'embedding_dim': self.embedding_dim,
            'similarity_threshold': self.similarity_threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(memory_data, f)
        
        print(f"Memory hierarchy saved to {filepath}")
    
    def load_memory(self, filepath: str):
        """Load memory hierarchy from disk."""
        with open(filepath, 'rb') as f:
            memory_data = pickle.load(f)
        
        self.memory_slots = memory_data['memory_slots']
        self.scene_to_slot = memory_data['scene_to_slot']
        self.total_scenes = memory_data['total_scenes']
        self.total_slots = memory_data['total_slots']
        self.memory_size = memory_data['memory_size']
        self.embedding_dim = memory_data['embedding_dim']
        self.similarity_threshold = memory_data['similarity_threshold']
        
        # Rebuild FAISS index
        if self.use_faiss:
            self._update_faiss_index()
        
        print(f"Memory hierarchy loaded from {filepath}")
    
    def visualize_memory_distribution(self, save_path: Optional[str] = None):
        """Visualize the distribution of scenes across memory slots."""
        import matplotlib.pyplot as plt
        
        slot_sizes = [slot.size for slot in self.memory_slots.values()]
        cluster_ids = [slot.cluster_id for slot in self.memory_slots.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Slot size distribution
        ax1.hist(slot_sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Number of Scenes per Slot')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Scene Counts per Memory Slot')
        ax1.grid(True, alpha=0.3)
        
        # Cluster distribution
        cluster_counts = defaultdict(int)
        for cluster_id in cluster_ids:
            cluster_counts[cluster_id] += 1
        
        ax2.bar(cluster_counts.keys(), cluster_counts.values(), alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Slots')
        ax2.set_title('Distribution of Memory Slots by Cluster')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 