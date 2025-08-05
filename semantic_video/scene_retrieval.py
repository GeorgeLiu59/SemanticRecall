"""
Scene Retrieval: Efficient retrieval of relevant scenes from semantic memory
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import heapq
from collections import defaultdict
import time


class SceneRetrieval:
    """
    Efficient scene retrieval system based on semantic similarity.
    
    This component provides fast and accurate retrieval of relevant scenes
    from the cross-temporal memory hierarchy using various search strategies.
    """
    
    def __init__(self, 
                 memory_hierarchy,
                 retrieval_method: str = "semantic",
                 max_results: int = 10,
                 similarity_threshold: float = 0.7,
                 use_caching: bool = True):
        """
        Initialize the scene retrieval system.
        
        Args:
            memory_hierarchy: Memory hierarchy instance
            retrieval_method: Retrieval method ('semantic', 'temporal', 'hybrid')
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            use_caching: Whether to use query result caching
        """
        self.memory_hierarchy = memory_hierarchy
        self.retrieval_method = retrieval_method
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold
        self.use_caching = use_caching
        
        # Query cache
        self.query_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Retrieval statistics
        self.retrieval_times = []
        self.query_counts = defaultdict(int)
    
    def retrieve_scenes(self, 
                       query_embedding: np.ndarray,
                       query_type: str = "semantic",
                       top_k: int = None,
                       temporal_constraints: Optional[Dict] = None) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve relevant scenes based on the query.
        
        Args:
            query_embedding: CLIP embedding of the query
            query_type: Type of query ('semantic', 'temporal', 'hybrid')
            top_k: Number of results to return (overrides max_results)
            temporal_constraints: Optional temporal constraints
            
        Returns:
            List of (scene_id, similarity_score, metadata) tuples
        """
        if top_k is None:
            top_k = self.max_results
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(query_embedding, query_type, top_k, temporal_constraints)
        if self.use_caching and cache_key in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[cache_key]
        
        self.cache_misses += 1
        
        # Perform retrieval based on method
        if query_type == "semantic":
            results = self._semantic_retrieval(query_embedding, top_k)
        elif query_type == "temporal":
            results = self._temporal_retrieval(query_embedding, top_k, temporal_constraints)
        elif query_type == "hybrid":
            results = self._hybrid_retrieval(query_embedding, top_k, temporal_constraints)
        else:
            raise ValueError(f"Unknown query type: {query_type}")
        
        # Apply temporal constraints if specified
        if temporal_constraints:
            results = self._apply_temporal_constraints(results, temporal_constraints)
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Cache results
        if self.use_caching:
            self.query_cache[cache_key] = results
        
        # Update statistics
        retrieval_time = time.time() - start_time
        self.retrieval_times.append(retrieval_time)
        self.query_counts[query_type] += 1
        
        return results[:top_k]
    
    def _semantic_retrieval(self, 
                           query_embedding: np.ndarray,
                           top_k: int) -> List[Tuple[int, float, Dict]]:
        """Retrieve scenes based on semantic similarity."""
        # Get cross-temporal scenes from memory hierarchy
        cross_temporal_scenes = self.memory_hierarchy.get_cross_temporal_scenes(
            query_embedding, top_k=top_k
        )
        
        results = []
        for scene_id, similarity, slot_id in cross_temporal_scenes:
            if similarity >= self.similarity_threshold:
                # Get metadata from memory slot
                slot = self.memory_hierarchy.memory_slots[slot_id]
                scene_idx = slot.scene_ids.index(scene_id)
                metadata = {
                    'slot_id': slot_id,
                    'cluster_id': slot.cluster_id,
                    'slot_size': slot.size,
                    'access_count': slot.access_count
                }
                
                # Add scene-specific metadata if available
                if 'timestamps' in slot.metadata:
                    metadata['timestamp'] = slot.metadata['timestamps'][scene_idx]
                if 'descriptions' in slot.metadata:
                    metadata['description'] = slot.metadata['descriptions'][scene_idx]
                
                results.append((scene_id, similarity, metadata))
        
        return results
    
    def _temporal_retrieval(self, 
                           query_embedding: np.ndarray,
                           top_k: int,
                           temporal_constraints: Optional[Dict]) -> List[Tuple[int, float, Dict]]:
        """Retrieve scenes based on temporal proximity."""
        # This would implement temporal-based retrieval
        # For now, fall back to semantic retrieval
        return self._semantic_retrieval(query_embedding, top_k)
    
    def _hybrid_retrieval(self, 
                         query_embedding: np.ndarray,
                         top_k: int,
                         temporal_constraints: Optional[Dict]) -> List[Tuple[int, float, Dict]]:
        """Retrieve scenes using both semantic and temporal criteria."""
        # Get semantic results
        semantic_results = self._semantic_retrieval(query_embedding, top_k * 2)
        
        # Apply temporal weighting if constraints provided
        if temporal_constraints:
            weighted_results = []
            for scene_id, similarity, metadata in semantic_results:
                # Apply temporal weighting
                temporal_weight = self._compute_temporal_weight(metadata, temporal_constraints)
                weighted_similarity = similarity * temporal_weight
                weighted_results.append((scene_id, weighted_similarity, metadata))
            
            # Sort by weighted similarity
            weighted_results.sort(key=lambda x: x[1], reverse=True)
            return weighted_results[:top_k]
        
        return semantic_results[:top_k]
    
    def _compute_temporal_weight(self, metadata: Dict, constraints: Dict) -> float:
        """Compute temporal weighting factor based on constraints."""
        if 'timestamp' not in metadata:
            return 1.0
        
        timestamp = metadata['timestamp']
        weight = 1.0
        
        # Time range constraint
        if 'start_time' in constraints and 'end_time' in constraints:
            if timestamp < constraints['start_time'] or timestamp > constraints['end_time']:
                weight *= 0.5  # Penalize scenes outside time range
        
        # Recency bias
        if 'recency_bias' in constraints:
            current_time = constraints.get('current_time', timestamp)
            time_diff = abs(current_time - timestamp)
            recency_factor = np.exp(-time_diff * constraints['recency_bias'])
            weight *= recency_factor
        
        return weight
    
    def _apply_temporal_constraints(self, 
                                  results: List[Tuple[int, float, Dict]],
                                  constraints: Dict) -> List[Tuple[int, float, Dict]]:
        """Apply temporal constraints to results."""
        filtered_results = []
        
        for scene_id, similarity, metadata in results:
            if 'timestamp' not in metadata:
                filtered_results.append((scene_id, similarity, metadata))
                continue
            
            timestamp = metadata['timestamp']
            
            # Check time range
            if 'start_time' in constraints and timestamp < constraints['start_time']:
                continue
            if 'end_time' in constraints and timestamp > constraints['end_time']:
                continue
            
            filtered_results.append((scene_id, similarity, metadata))
        
        return filtered_results
    
    def _get_cache_key(self, 
                      query_embedding: np.ndarray,
                      query_type: str,
                      top_k: int,
                      temporal_constraints: Optional[Dict]) -> str:
        """Generate cache key for query."""
        # Quantize embedding for cache key
        quantized_embedding = np.round(query_embedding, decimals=3)
        
        key_parts = [
            str(quantized_embedding.tobytes()),
            query_type,
            str(top_k)
        ]
        
        if temporal_constraints:
            key_parts.append(str(sorted(temporal_constraints.items())))
        
        return hash(tuple(key_parts))
    
    def batch_retrieve(self, 
                      query_embeddings: List[np.ndarray],
                      query_types: Optional[List[str]] = None,
                      top_k: int = None) -> List[List[Tuple[int, float, Dict]]]:
        """
        Perform batch retrieval for multiple queries.
        
        Args:
            query_embeddings: List of query embeddings
            query_types: List of query types (optional)
            top_k: Number of results per query
            
        Returns:
            List of result lists
        """
        if query_types is None:
            query_types = [self.retrieval_method] * len(query_embeddings)
        
        results = []
        for query_embedding, query_type in zip(query_embeddings, query_types):
            result = self.retrieve_scenes(query_embedding, query_type, top_k)
            results.append(result)
        
        return results
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retrieval statistics."""
        stats = {
            'total_queries': sum(self.query_counts.values()),
            'query_type_distribution': dict(self.query_counts),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'avg_retrieval_time': np.mean(self.retrieval_times) if self.retrieval_times else 0,
            'max_retrieval_time': np.max(self.retrieval_times) if self.retrieval_times else 0,
            'min_retrieval_time': np.min(self.retrieval_times) if self.retrieval_times else 0,
            'cache_size': len(self.query_cache)
        }
        
        return stats
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        print("Query cache cleared")
    
    def get_similar_scenes(self, 
                          scene_id: int,
                          top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """
        Find scenes similar to a given scene.
        
        Args:
            scene_id: ID of the reference scene
            top_k: Number of similar scenes to return
            
        Returns:
            List of (scene_id, similarity_score, metadata) tuples
        """
        # Get the scene's embedding from memory hierarchy
        if scene_id not in self.memory_hierarchy.scene_to_slot:
            return []
        
        slot_id = self.memory_hierarchy.scene_to_slot[scene_id]
        slot = self.memory_hierarchy.memory_slots[slot_id]
        
        # Find the scene's embedding
        scene_idx = slot.scene_ids.index(scene_id)
        scene_embedding = slot.embeddings[scene_idx]
        
        # Retrieve similar scenes
        return self.retrieve_scenes(scene_embedding, top_k=top_k)
    
    def get_cluster_scenes(self, 
                          cluster_id: int,
                          top_k: int = 10) -> List[Tuple[int, float, Dict]]:
        """
        Get all scenes from a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            top_k: Maximum number of scenes to return
            
        Returns:
            List of (scene_id, similarity_score, metadata) tuples
        """
        results = []
        
        for slot_id, slot in self.memory_hierarchy.memory_slots.items():
            if slot.cluster_id == cluster_id:
                for i, scene_id in enumerate(slot.scene_ids):
                    metadata = {
                        'slot_id': slot_id,
                        'cluster_id': cluster_id,
                        'slot_size': slot.size,
                        'access_count': slot.access_count
                    }
                    
                    # Add scene-specific metadata
                    if 'timestamps' in slot.metadata:
                        metadata['timestamp'] = slot.metadata['timestamps'][i]
                    if 'descriptions' in slot.metadata:
                        metadata['description'] = slot.metadata['descriptions'][i]
                    
                    # Use cluster center similarity as score
                    similarity = slot.get_similarity(slot.embeddings[i])
                    results.append((scene_id, similarity, metadata))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_by_description(self, 
                            description: str,
                            scene_encoder,
                            top_k: int = 10) -> List[Tuple[int, float, Dict]]:
        """
        Search for scenes using a text description.
        
        Args:
            description: Text description to search for
            scene_encoder: Scene encoder instance for text encoding
            top_k: Number of results to return
            
        Returns:
            List of (scene_id, similarity_score, metadata) tuples
        """
        # Encode the text description
        query_embedding = scene_encoder.encode_text_query(description)
        
        # Retrieve similar scenes
        return self.retrieve_scenes(query_embedding, top_k=top_k) 