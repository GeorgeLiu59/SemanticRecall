"""
Semantic Clustering: Groups video scenes by semantic similarity using CLIP embeddings
"""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform


class SemanticClustering:
    """
    Clusters video scenes by semantic similarity using CLIP embeddings.
    
    This component enables cross-temporal scene grouping by identifying
    semantically similar scenes across the entire video timeline.
    """
    
    def __init__(self, 
                 method: str = "kmeans",
                 n_clusters: int = 10,
                 random_state: int = 42,
                 similarity_threshold: float = 0.7):
        """
        Initialize the semantic clustering.
        
        Args:
            method: Clustering method ('kmeans', 'spectral', 'dbscan')
            n_clusters: Number of clusters for K-means/Spectral
            random_state: Random seed for reproducibility
            similarity_threshold: Threshold for similarity-based clustering
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.similarity_threshold = similarity_threshold
        self.clusterer = None
        self.cluster_labels = None
        self.cluster_centers = None
        
    def fit(self, scene_embeddings: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Fit the clustering model to scene embeddings.
        
        Args:
            scene_embeddings: Dictionary mapping scene_id to CLIP embedding
            
        Returns:
            Dictionary mapping scene_id to cluster_id
        """
        # Convert to numpy array
        scene_ids = list(scene_embeddings.keys())
        embeddings = np.array([scene_embeddings[scene_id] for scene_id in scene_ids])
        
        # Perform clustering
        if self.method == "kmeans":
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            self.cluster_labels = self.clusterer.fit_predict(embeddings)
            self.cluster_centers = self.clusterer.cluster_centers_
            
        elif self.method == "spectral":
            # Compute similarity matrix
            similarity_matrix = self._compute_similarity_matrix(embeddings)
            
            self.clusterer = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity='precomputed',
                random_state=self.random_state
            )
            self.cluster_labels = self.clusterer.fit_predict(similarity_matrix)
            self.cluster_centers = self._compute_cluster_centers(embeddings, self.cluster_labels)
            
        elif self.method == "dbscan":
            # Compute similarity matrix
            similarity_matrix = self._compute_similarity_matrix(embeddings)
            
            self.clusterer = DBSCAN(
                eps=1 - self.similarity_threshold,  # Convert similarity to distance
                min_samples=2,
                metric='precomputed'
            )
            self.cluster_labels = self.clusterer.fit_predict(1 - similarity_matrix)
            self.cluster_centers = self._compute_cluster_centers(embeddings, self.cluster_labels)
            
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Create scene_id to cluster_id mapping
        scene_to_cluster = {scene_ids[i]: int(self.cluster_labels[i]) for i in range(len(scene_ids))}
        
        return scene_to_cluster
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between embeddings."""
        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        
        return similarity_matrix
    
    def _compute_cluster_centers(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute cluster centers as mean of embeddings in each cluster."""
        unique_labels = np.unique(labels)
        centers = []
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
            cluster_embeddings = embeddings[labels == label]
            center = np.mean(cluster_embeddings, axis=0)
            centers.append(center)
        
        return np.array(centers)
    
    def evaluate_clustering(self, scene_embeddings: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate clustering quality using various metrics.
        
        Args:
            scene_embeddings: Dictionary mapping scene_id to CLIP embedding
            
        Returns:
            Dictionary of evaluation metrics
        """
        scene_ids = list(scene_embeddings.keys())
        embeddings = np.array([scene_embeddings[scene_id] for scene_id in scene_ids])
        
        metrics = {}
        
        # Silhouette score (higher is better)
        if len(np.unique(self.cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(embeddings, self.cluster_labels)
        
        # Calinski-Harabasz score (higher is better)
        if len(np.unique(self.cluster_labels)) > 1:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, self.cluster_labels)
        
        # Number of clusters
        n_clusters = len(np.unique(self.cluster_labels[self.cluster_labels != -1]))
        metrics['n_clusters'] = n_clusters
        
        # Cluster sizes
        cluster_sizes = []
        for label in np.unique(self.cluster_labels):
            if label != -1:  # Skip noise points
                size = np.sum(self.cluster_labels == label)
                cluster_sizes.append(size)
        
        metrics['min_cluster_size'] = min(cluster_sizes) if cluster_sizes else 0
        metrics['max_cluster_size'] = max(cluster_sizes) if cluster_sizes else 0
        metrics['avg_cluster_size'] = np.mean(cluster_sizes) if cluster_sizes else 0
        
        return metrics
    
    def visualize_clusters(self, 
                          scene_embeddings: Dict[int, np.ndarray],
                          save_path: Optional[str] = None) -> None:
        """
        Visualize clustering results using t-SNE.
        
        Args:
            scene_embeddings: Dictionary mapping scene_id to CLIP embedding
            save_path: Path to save the visualization
        """
        scene_ids = list(scene_embeddings.keys())
        embeddings = np.array([scene_embeddings[scene_id] for scene_id in scene_ids])
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot clusters
        unique_labels = np.unique(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points
                mask = self.cluster_labels == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                mask = self.cluster_labels == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], s=50, alpha=0.7, label=f'Cluster {label}')
        
        plt.title(f'Semantic Scene Clustering ({self.method.upper()})')
        plt.xlabel('')
        plt.ylabel('')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_cluster_statistics(self, scene_embeddings: Dict[int, np.ndarray]) -> Dict:
        """
        Get detailed statistics about each cluster.
        
        Args:
            scene_embeddings: Dictionary mapping scene_id to CLIP embedding
            
        Returns:
            Dictionary with cluster statistics
        """
        scene_ids = list(scene_embeddings.keys())
        embeddings = np.array([scene_embeddings[scene_id] for scene_id in scene_ids])
        
        stats = {}
        unique_labels = np.unique(self.cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            mask = self.cluster_labels == label
            cluster_embeddings = embeddings[mask]
            cluster_scene_ids = [scene_ids[i] for i, is_in_cluster in enumerate(mask) if is_in_cluster]
            
            # Compute cluster statistics
            cluster_center = np.mean(cluster_embeddings, axis=0)
            cluster_std = np.std(cluster_embeddings, axis=0)
            
            # Compute intra-cluster similarity
            similarities = []
            for i in range(len(cluster_embeddings)):
                for j in range(i+1, len(cluster_embeddings)):
                    sim = np.dot(cluster_embeddings[i], cluster_embeddings[j]) / (
                        np.linalg.norm(cluster_embeddings[i]) * np.linalg.norm(cluster_embeddings[j])
                    )
                    similarities.append(sim)
            
            avg_intra_similarity = np.mean(similarities) if similarities else 0
            
            stats[label] = {
                'size': len(cluster_scene_ids),
                'scene_ids': cluster_scene_ids,
                'center': cluster_center,
                'std': cluster_std,
                'avg_intra_similarity': avg_intra_similarity
            }
        
        return stats
    
    def find_similar_clusters(self, 
                            query_embedding: np.ndarray,
                            top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Find clusters most similar to a query embedding.
        
        Args:
            query_embedding: CLIP embedding to compare against
            top_k: Number of top clusters to return
            
        Returns:
            List of (cluster_id, similarity_score) tuples
        """
        if self.cluster_centers is None:
            return []
        
        similarities = []
        for cluster_id, center in enumerate(self.cluster_centers):
            similarity = np.dot(query_embedding, center) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(center)
            )
            similarities.append((cluster_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k] 