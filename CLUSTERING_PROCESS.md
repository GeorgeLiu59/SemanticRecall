# Semantic Video Clustering Process

## Overview
This document explains the step-by-step process of how video scenes are analyzed and clustered based on their semantic content using CLIP embeddings and Gemini AI.

## Step-by-Step Processing Pipeline

### 1. Video Scene Detection
- **Input**: Raw video file (e.g., `Bee_Movie_Cropped.mp4`)
- **Process**: Shot boundary detection using OpenCV
- **Output**: List of scene boundaries with start/end timestamps
- **Example**: `[(0.0, 1.25), (1.25, 2.5), (2.5, 3.75), ...]`

### 2. Frame Extraction & CLIP Encoding
- **Input**: Scene boundaries from Step 1
- **Process**: 
  - Extract frames uniformly distributed across each scene (20 frames max)
  - Encode each frame using CLIP (Contrastive Language-Image Pre-training)
  - Convert visual content to high-dimensional embeddings (512-dimensional vectors)
- **Output**: Array of CLIP embeddings, one per scene
- **Purpose**: CLIP embeddings capture semantic meaning of visual content
- **Improvement**: Uniform temporal sampling ensures better coverage of entire scene duration

### 3. Semantic Clustering
- **Input**: CLIP embeddings from Step 2
- **Process**: 
  - Try multiple clustering algorithms (K-means, Spectral, DBSCAN)
  - Test different parameters (k values, eps values)
  - Select best clustering based on silhouette score
- **Output**: Cluster labels for each scene
- **Example**: `[0, 0, 1, 1, 2, 0, 3, 4, 4, 2, ...]`

### 4. Representative Frame Selection
- **Input**: Cluster assignments and embeddings
- **Process**:
  - Calculate cluster center (mean of all embeddings in cluster)
  - Find scene closest to cluster center using Euclidean distance
  - Select this as the most representative scene
- **Output**: Most representative scene ID for each cluster
- **Purpose**: Ensures Gemini analyzes the most typical scene from each cluster

### 5. Video Clip Extraction
- **Input**: Representative scene boundaries
- **Process**:
  - Extract video clip from representative scene (up to 5 seconds)
  - Convert to sequence of RGB frames
  - Limit duration to avoid API rate limits
- **Output**: Sequence of frames for each cluster
- **Example**: 15 frames from 2.5-second clip

### 6. Gemini AI Analysis
- **Input**: Video frames from Step 5
- **Process**:
  - Send frames to Gemini 1.5 Flash/Pro model
  - Model analyzes visual content, movement, and temporal context
  - Generate concise semantic description
- **Output**: Natural language description of cluster content
- **Example**: "Characters engage in animated dialogue with expressive gestures"

## Key Technical Components

### CLIP Embeddings
- **Model**: OpenAI CLIP (ViT-B/32 variant)
- **Output**: 512-dimensional vectors
- **Properties**: 
  - Captures semantic similarity between images and text
  - Enables zero-shot classification
  - Preserves visual-semantic relationships

### Clustering Algorithms
1. **K-means**: Groups scenes into k clusters based on embedding similarity
2. **Spectral Clustering**: Uses graph-based approach for non-linear clusters
3. **DBSCAN**: Density-based clustering for irregular cluster shapes

### Evaluation Metrics
- **Silhouette Score**: Measures how well-separated clusters are (-1 to 1, higher is better)
- **Intra-cluster Similarity**: Average similarity between scenes within same cluster
- **Cluster Size Distribution**: Balance of cluster sizes

## Example Output

```
Cluster 0:
   Size: 12 scenes
   Scene IDs: [4, 5, 32, 43, 54]...
   Average intra-cluster similarity: 0.835
   Analyzing with Gemini API...
   Selected most representative scene 85 (distance to center: 0.301) - extracted 31 frames
   Using model: gemini-2.0-flash-lite
   Semantic meaning: 'Barry B. Benson clings to a rope while hanging onto a moving truck.'
   Time ranges: [(5.0, 6.25), (6.25, 7.5), (81.25, 83.75)]...

Cluster 1:
   Size: 7 scenes
   Scene IDs: [2, 23, 26, 29, 75]...
   Average intra-cluster similarity: 0.783
   Analyzing with Gemini API...
   Selected most representative scene 29 (distance to center: 0.350) - extracted 31 frames
   Using model: gemini-2.0-flash-lite
   Semantic meaning: 'A man holds a jar of honey, examining it with a look of curiosity.'
   Time ranges: [(2.5, 3.75), (48.75, 63.75), (67.5, 68.75)]...
```