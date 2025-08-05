#!/usr/bin/env python3
"""
Show scene clusters from the video with semantic descriptions using Gemini API.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from semantic_video import SemanticVideoProcessor
import cv2
import os
import base64
from io import BytesIO
from PIL import Image

# Gemini API imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not available. Install with: pip install google-generativeai")

def setup_gemini(api_key):
    """Setup Gemini API with the provided key."""
    if not GEMINI_AVAILABLE:
        return False
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Failed to setup Gemini: {e}")
        return False

def extract_sample_frames(video_path, scene_boundaries, scene_ids, cluster_embeddings, num_samples=1):
    """Extract sample video clips from scenes in the cluster, prioritizing the most representative."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if len(cluster_embeddings) > 0:
        # Find the most representative frame (closest to cluster center)
        cluster_center = cluster_embeddings.mean(axis=0)
        
        # Calculate distance from each scene to cluster center
        distances = []
        for i, scene_embedding in enumerate(cluster_embeddings):
            distance = np.linalg.norm(scene_embedding - cluster_center)
            distances.append(distance)
        
        # Find the scene closest to cluster center
        most_representative_idx = np.argmin(distances)
        most_representative_scene_id = scene_ids[most_representative_idx]
        
        # Extract video clip from the most representative scene
        start_time, end_time = scene_boundaries[most_representative_scene_id]
        duration = end_time - start_time
        
        # Limit clip duration to avoid rate limits (max 5 seconds)
        if duration > 5.0:
            middle_time = (start_time + end_time) / 2
            start_time = middle_time - 2.5
            end_time = middle_time + 2.5
        
        # Extract frames for video clip
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        clip_frames = []
        
        while cap.get(cv2.CAP_PROP_POS_MSEC) < end_time * 1000:
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clip_frames.append(frame_rgb)
            else:
                break
        
        if clip_frames:
            frames = clip_frames
            print(f"   Selected most representative scene {most_representative_scene_id} (distance to center: {distances[most_representative_idx]:.3f}) - extracted {len(clip_frames)} frames")
    
    cap.release()
    return frames

def encode_image_to_base64(image_array):
    """Convert numpy image array to base64 string."""
    pil_image = Image.fromarray(image_array)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def generate_gemini_description(api_key, video_path, scene_boundaries, scene_ids, cluster_id, cluster_embeddings):
    """Generate semantic description using Gemini API analyzing actual video frames."""
    
    if not GEMINI_AVAILABLE:
        return "Gemini API not available", 0.0
    
    try:
        # Setup Gemini
        if not setup_gemini(api_key):
            return "Failed to setup Gemini API", 0.0
        
        # Extract sample frames from the cluster
        sample_frames = extract_sample_frames(video_path, scene_boundaries, scene_ids, cluster_embeddings, num_samples=1)
        
        if not sample_frames:
            return "No frames could be extracted", 0.0
        
        # Create Gemini model - using models that support video input
        model_names = [
            'gemini-2.0-flash-lite',
            'gemini-2.0-flash-exp',
            'gemini-2.5-flash-lite',
            'gemini-1.5-flash-latest',
            'gemini-pro-vision'  # fallback
        ]
        
        model = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                print(f"   Using model: {model_name}")
                break
            except Exception as e:
                print(f"   Model {model_name} failed: {e}")
                continue
        
        if model is None:
            return "No compatible Gemini model found", 0.0
        
        # Prepare images/video for Gemini
        media = []
        for frame in sample_frames:
            img_base64 = encode_image_to_base64(frame)
            media.append({
                "mime_type": "image/jpeg",
                "data": img_base64
            })
        
        # Create prompt for Gemini
        prompt = f"""Describe this scene briefly in one short sentence."""
        
        # Generate description with Gemini
        if len(media) == 1:
            # Single frame - use image model
            response = model.generate_content([prompt] + media)
        else:
            # Multiple frames - use video model
            response = model.generate_content([prompt] + media)
        
        if response.text:
            # Add delay to avoid rate limits
            import time
            time.sleep(2)  # 2 second delay between calls
            return response.text.strip(), 0.9  # High confidence for Gemini
        else:
            return "No response from Gemini", 0.0
            
    except Exception as e:
        print(f"Gemini description failed: {e}")
        return f"Gemini analysis failed: {str(e)}", 0.0

def analyze_cluster_content(processor, cluster_embeddings, scene_ids, boundaries, cluster_id, video_path, gemini_api_key):
    """Analyze cluster content using Gemini API on actual video frames."""
    
    # Use Gemini to analyze actual video frames
    description, confidence = generate_gemini_description(
        gemini_api_key, video_path, boundaries, scene_ids, cluster_id, cluster_embeddings
    )
    
    return description, confidence

def improved_clustering(embeddings, n_clusters=8):
    """Improved clustering approach with better parameter selection."""
    
    # Try different clustering methods and parameters
    best_score = -1
    best_clusters = None
    best_method = ""
    
    # K-means with different k values
    for k in range(4, min(12, len(embeddings) // 3)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        if len(np.unique(cluster_labels)) > 1:
            score = silhouette_score(embeddings, cluster_labels)
            if score > best_score:
                best_score = score
                best_clusters = cluster_labels
                best_method = f"K-means (k={k})"
    
    # Spectral clustering
    try:
        for k in range(4, min(10, len(embeddings) // 4)):
            spectral = SpectralClustering(n_clusters=k, random_state=42, affinity='rbf')
            cluster_labels = spectral.fit_predict(embeddings)
            
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(embeddings, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_clusters = cluster_labels
                    best_method = f"Spectral (k={k})"
    except:
        pass
    
    # DBSCAN with different eps values
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
        try:
            dbscan = DBSCAN(eps=eps, min_samples=3)
            cluster_labels = dbscan.fit_predict(embeddings)
            
            if len(np.unique(cluster_labels)) > 1 and -1 not in cluster_labels:
                score = silhouette_score(embeddings, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_clusters = cluster_labels
                    best_method = f"DBSCAN (eps={eps})"
        except:
            pass
    
    return best_clusters, best_score, best_method

def show_clusters(gemini_api_key):
    print("Loading video and detecting scenes...")
    
    # Initialize processor
    processor = SemanticVideoProcessor(
        video_path="Bee_Movie_Cropped.mp4",
        num_clusters=8,
        memory_size=100,
        clustering_method="kmeans",
        num_experts=4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Detect scenes
    print("1. Detecting scene boundaries...")
    boundaries = processor.detect_scene_boundaries()
    print(f"   Found {len(boundaries)} scenes")
    
    # Extract embeddings
    print("2. Extracting CLIP embeddings...")
    embeddings = processor.extract_scene_embeddings()
    print(f"   Extracted {len(embeddings)} embeddings")
    
    # Convert embeddings to numpy array if they're in dictionary format
    if isinstance(embeddings, dict):
        # If embeddings is a dict, extract the actual embedding arrays
        embeddings_list = []
        for scene_id in sorted(embeddings.keys()):
            if isinstance(embeddings[scene_id], dict):
                # If each scene has a dict with 'embedding' key
                embeddings_list.append(embeddings[scene_id]['embedding'])
            else:
                # If each scene directly has the embedding
                embeddings_list.append(embeddings[scene_id])
        embeddings = np.array(embeddings_list)
    elif isinstance(embeddings, list):
        # If embeddings is a list, convert to numpy array
        embeddings = np.array(embeddings)
    
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Perform improved clustering
    print("3. Performing improved semantic clustering...")
    cluster_labels, silhouette_score, method = improved_clustering(embeddings)
    
    print(f"\nClustering Results:")
    print(f"   Method: {method}")
    print(f"   Number of clusters: {len(np.unique(cluster_labels))}")
    print(f"   Silhouette score: {silhouette_score:.3f}")
    
    # Analyze each cluster
    print(f"\nCluster Details:")
    unique_labels = np.unique(cluster_labels)
    
    # Store descriptions to avoid calling Gemini twice
    cluster_descriptions = {}
    
    for cluster_id in unique_labels:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        
        print(f"\nCluster {cluster_id}:")
        print(f"   Size: {len(cluster_indices)} scenes")
        print(f"   Scene IDs: {cluster_indices[:5].tolist()}{'...' if len(cluster_indices) > 5 else ''}")
        
        # Calculate intra-cluster similarity
        if len(cluster_embeddings) > 1:
            similarities = []
            for i in range(len(cluster_embeddings)):
                for j in range(i+1, len(cluster_embeddings)):
                    sim = processor.scene_encoder.compute_semantic_similarity(
                        cluster_embeddings[i], cluster_embeddings[j]
                    )
                    similarities.append(sim)
            avg_similarity = np.mean(similarities)
            print(f"   Average intra-cluster similarity: {avg_similarity:.3f}")
        
        # Generate semantic description using Gemini (only once per cluster)
        print(f"   Analyzing with Gemini API...")
        description, confidence = analyze_cluster_content(
            processor, cluster_embeddings, cluster_indices, boundaries, cluster_id, 
            "Bee_Movie_Cropped.mp4", gemini_api_key
        )
        print(f"   Semantic meaning: '{description}'")
        
        # Store description for summary
        cluster_descriptions[cluster_id] = (description, len(cluster_indices))
        
        # Show time ranges
        times = [boundaries[i] for i in cluster_indices]
        print(f"   Time ranges: {times[:3]}{'...' if len(times) > 3 else ''}")
    
    # Create visualization
    print(f"\nCreating visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, cluster_id in enumerate(unique_labels):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        plt.scatter(
            embeddings_2d[cluster_indices, 0], 
            embeddings_2d[cluster_indices, 1],
            c=[colors[i]], 
            label=f'Cluster {cluster_id}',
            alpha=0.7,
            s=50
        )
    
    plt.title(f'Semantic Scene Clustering ({method})')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("scene_clusters_gemini.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: scene_clusters_gemini.png")
    
    # Summary using stored descriptions (no additional API calls)
    print(f"\nSemantic Cluster Summary:")
    for cluster_id in unique_labels:
        description, num_scenes = cluster_descriptions[cluster_id]
        print(f"   Cluster {cluster_id}: {description[:80]}{'...' if len(description) > 80 else ''} ({num_scenes} scenes)")

if __name__ == "__main__":
    # Get Gemini API key from environment variable
    import os
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("GEMINI_API_KEY environment variable not found. Please set it with:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        exit(1)
    
    show_clusters(gemini_api_key) 