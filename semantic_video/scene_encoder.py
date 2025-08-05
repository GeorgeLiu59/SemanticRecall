"""
Scene Encoder: CLIP-based semantic encoding for video scenes
"""

import torch
import open_clip
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm


class SceneEncoder:
    """
    Encodes video scenes using CLIP embeddings to capture semantic content.
    
    This is the core component that extracts semantic representations from
    video segments, enabling cross-temporal scene similarity analysis.
    """
    
    def __init__(self, 
                 clip_model_name: str = "ViT-B/32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 frame_sample_rate: int = 1,
                 max_frames_per_scene: int = 10):
        """
        Initialize the scene encoder.
        
        Args:
            clip_model_name: CLIP model variant to use
            device: Device to run CLIP on
            frame_sample_rate: Sample every Nth frame for efficiency
            max_frames_per_scene: Maximum frames to process per scene
        """
        self.device = device
        self.frame_sample_rate = frame_sample_rate
        self.max_frames_per_scene = max_frames_per_scene
        
        # Load CLIP model
        print(f"Loading CLIP model: {clip_model_name}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, device=device, pretrained='openai'
        )
        self.model.eval()
        
        # Text templates for scene description
        self.text_templates = [
            "a scene showing {}",
            "a video of {}",
            "footage of {}",
            "a clip featuring {}"
        ]
    
    def extract_scene_embeddings(self, 
                                video_path: str,
                                scene_boundaries: List[Tuple[float, float]]) -> Dict[int, np.ndarray]:
        """
        Extract CLIP embeddings for each scene in the video.
        
        Args:
            video_path: Path to the video file
            scene_boundaries: List of (start_time, end_time) tuples for each scene
            
        Returns:
            Dictionary mapping scene_id to CLIP embedding
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scene_embeddings = {}
        
        for scene_id, (start_time, end_time) in enumerate(tqdm(scene_boundaries, desc="Encoding scenes")):
            # Convert time to frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Extract frames from this scene
            frames = self._extract_scene_frames(cap, start_frame, end_frame)
            
            if frames:
                # Encode the scene
                embedding = self._encode_scene_frames(frames)
                scene_embeddings[scene_id] = embedding
        
        cap.release()
        return scene_embeddings
    
    def _extract_scene_frames(self, cap, start_frame: int, end_frame: int) -> List[Image.Image]:
        """Extract and preprocess frames from a scene."""
        frames = []
        frame_count = 0
        
        # Sample frames from the scene
        for frame_idx in range(start_frame, end_frame, self.frame_sample_rate):
            if frame_count >= self.max_frames_per_scene:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB and to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                frame_count += 1
        
        return frames
    
    def _encode_scene_frames(self, frames: List[Image.Image]) -> np.ndarray:
        """
        Encode a list of frames into a single scene embedding.
        
        Args:
            frames: List of PIL Images representing frames from the scene
            
        Returns:
            CLIP embedding for the scene (normalized)
        """
        if not frames:
            return np.zeros(512)  # CLIP embedding dimension
        
        # Preprocess frames
        processed_frames = []
        for frame in frames:
            try:
                processed = self.preprocess(frame).unsqueeze(0).to(self.device)
                processed_frames.append(processed)
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
        
        if not processed_frames:
            return np.zeros(512)
        
        # Stack frames and encode
        with torch.no_grad():
            frame_tensor = torch.cat(processed_frames, dim=0)
            image_features = self.model.encode_image(frame_tensor)
            
            # Average pooling across frames
            scene_embedding = torch.mean(image_features, dim=0)
            
            # Normalize
            scene_embedding = F.normalize(scene_embedding, p=2, dim=0)
            
            return scene_embedding.cpu().numpy()
    
    def encode_text_query(self, text: str) -> np.ndarray:
        """
        Encode a text query using CLIP text encoder.
        
        Args:
            text: Text query to encode
            
        Returns:
            CLIP text embedding (normalized)
        """
        # Try different text templates for better matching
        text_embeddings = []
        
        for template in self.text_templates:
            formatted_text = template.format(text)
            text_tokens = open_clip.tokenize([formatted_text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, p=2, dim=1)
                text_embeddings.append(text_features.cpu().numpy())
        
        # Average across templates
        return np.mean(text_embeddings, axis=0).squeeze()
    
    def compute_semantic_similarity(self, 
                                  embedding1: np.ndarray, 
                                  embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First CLIP embedding
            embedding2: Second CLIP embedding
            
        Returns:
            Cosine similarity score
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def get_scene_description(self, scene_embedding: np.ndarray, 
                            candidate_descriptions: List[str]) -> str:
        """
        Find the best matching description for a scene embedding.
        
        Args:
            scene_embedding: CLIP embedding of the scene
            candidate_descriptions: List of possible scene descriptions
            
        Returns:
            Best matching description
        """
        best_score = -1
        best_description = ""
        
        for desc in candidate_descriptions:
            text_embedding = self.encode_text_query(desc)
            similarity = self.compute_semantic_similarity(scene_embedding, text_embedding)
            
            if similarity > best_score:
                best_score = similarity
                best_description = desc
        
        return best_description 