"""
MOE Experts: Mixture of Experts specialized by semantic scene categories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pickle
import os


class SceneExpert(nn.Module):
    """
    Individual expert network specialized for a specific scene category.
    
    Each expert is trained to handle scenes from a particular semantic cluster,
    enabling specialized processing for different types of visual content.
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize a scene expert.
        
        Args:
            input_dim: Input embedding dimension (CLIP embedding size)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Expert statistics
        self.training_samples = 0
        self.inference_count = 0
        self.avg_confidence = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert network.
        
        Args:
            x: Input embeddings (batch_size, input_dim)
            
        Returns:
            Processed embeddings (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_expert_info(self) -> Dict[str, Any]:
        """Get information about this expert."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'training_samples': self.training_samples,
            'inference_count': self.inference_count,
            'avg_confidence': self.avg_confidence,
            'total_params': sum(p.numel() for p in self.parameters())
        }


class MOEGate(nn.Module):
    """
    Gating network that routes inputs to appropriate experts.
    
    The gate learns to assign scene embeddings to the most suitable expert
    based on semantic similarity and expert specialization.
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 num_experts: int = 10,
                 hidden_dim: int = 128,
                 temperature: float = 1.0):
        """
        Initialize the MOE gate.
        
        Args:
            input_dim: Input embedding dimension
            num_experts: Number of experts to route to
            hidden_dim: Hidden layer dimension
            temperature: Temperature for softmax (controls routing sharpness)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.temperature = temperature
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Expert embeddings (learnable representations of each expert's specialization)
        self.expert_embeddings = nn.Parameter(torch.randn(num_experts, input_dim))
        
        # Routing statistics
        self.routing_counts = defaultdict(int)
    
    def forward(self, x: torch.Tensor, top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the gate.
        
        Args:
            x: Input embeddings (batch_size, input_dim)
            top_k: Number of top experts to route to
            
        Returns:
            Tuple of (expert_weights, expert_indices)
        """
        batch_size = x.size(0)
        
        # Compute routing weights
        gate_weights = self.gate_network(x)  # (batch_size, num_experts)
        
        # Apply temperature
        gate_weights = gate_weights / self.temperature
        
        # Get top-k experts
        if top_k == 1:
            expert_indices = torch.argmax(gate_weights, dim=-1)  # (batch_size,)
            expert_weights = torch.ones_like(gate_weights).scatter_(
                1, expert_indices.unsqueeze(1), 1.0
            )
        else:
            expert_weights, expert_indices = torch.topk(gate_weights, top_k, dim=-1)
            # Normalize weights
            expert_weights = F.softmax(expert_weights, dim=-1)
        
        # Update routing statistics
        for i in range(batch_size):
            if top_k == 1:
                expert_idx = expert_indices[i].item()
                self.routing_counts[expert_idx] += 1
            else:
                for j in range(top_k):
                    expert_idx = expert_indices[i, j].item()
                    self.routing_counts[expert_idx] += expert_weights[i, j].item()
        
        return expert_weights, expert_indices
    
    def get_routing_statistics(self) -> Dict[int, int]:
        """Get routing statistics for each expert."""
        return dict(self.routing_counts)


class MOEExperts:
    """
    Mixture of Experts system with scene-specialized expert networks.
    
    This component routes different scene types to specialized expert networks,
    enabling efficient and accurate processing of diverse visual content.
    """
    
    def __init__(self, 
                 num_experts: int = 10,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 512,
                 expert_layers: int = 3,
                 gate_hidden_dim: int = 128,
                 temperature: float = 1.0,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the MOE experts system.
        
        Args:
            num_experts: Number of expert networks
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension for experts
            output_dim: Output embedding dimension
            expert_layers: Number of layers in each expert
            gate_hidden_dim: Hidden layer dimension for gate
            temperature: Temperature for gating softmax
            device: Device to run on
        """
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        # Create experts
        self.experts = nn.ModuleList([
            SceneExpert(input_dim, hidden_dim, output_dim, expert_layers)
            for _ in range(num_experts)
        ]).to(device)
        
        # Create gate
        self.gate = MOEGate(input_dim, num_experts, gate_hidden_dim, temperature).to(device)
        
        # Training state
        self.is_trained = False
        self.training_data = defaultdict(list)
        
        # Performance tracking
        self.expert_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    def forward(self, 
                x: torch.Tensor, 
                top_k: int = 1,
                return_routing: bool = False) -> torch.Tensor:
        """
        Forward pass through the MOE system.
        
        Args:
            x: Input embeddings (batch_size, input_dim)
            top_k: Number of top experts to use
            return_routing: Whether to return routing information
            
        Returns:
            Processed embeddings or tuple with routing info
        """
        batch_size = x.size(0)
        
        # Get routing weights and expert indices
        expert_weights, expert_indices = self.gate(x, top_k)
        
        # Initialize output
        output = torch.zeros(batch_size, self.output_dim, device=self.device)
        
        # Process through experts
        if top_k == 1:
            for i in range(batch_size):
                expert_idx = expert_indices[i]
                expert_output = self.experts[expert_idx](x[i:i+1])
                output[i] = expert_output.squeeze(0)
                
                # Update expert statistics
                self.experts[expert_idx].inference_count += 1
        else:
            for i in range(batch_size):
                for j in range(top_k):
                    expert_idx = expert_indices[i, j]
                    weight = expert_weights[i, j]
                    expert_output = self.experts[expert_idx](x[i:i+1])
                    output[i] += weight * expert_output.squeeze(0)
                    
                    # Update expert statistics
                    self.experts[expert_idx].inference_count += weight.item()
        
        if return_routing:
            return output, expert_weights, expert_indices
        else:
            return output
    
    def train_experts(self, 
                     scene_embeddings: Dict[int, np.ndarray],
                     scene_to_cluster: Dict[int, int],
                     num_epochs: int = 100,
                     batch_size: int = 32,
                     learning_rate: float = 1e-3,
                     cluster_to_expert: Optional[Dict[int, int]] = None):
        """
        Train the expert networks on scene embeddings.
        
        Args:
            scene_embeddings: Dictionary mapping scene_id to CLIP embedding
            scene_to_cluster: Dictionary mapping scene_id to cluster_id
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            cluster_to_expert: Optional mapping from cluster_id to expert_id
        """
        print("Training MOE experts...")
        
        # Create cluster to expert mapping if not provided
        if cluster_to_expert is None:
            unique_clusters = set(scene_to_cluster.values())
            cluster_to_expert = {}
            for i, cluster_id in enumerate(unique_clusters):
                expert_id = i % self.num_experts
                cluster_to_expert[cluster_id] = expert_id
        
        # Prepare training data
        expert_data = defaultdict(list)
        for scene_id, embedding in scene_embeddings.items():
            cluster_id = scene_to_cluster[scene_id]
            expert_id = cluster_to_expert[cluster_id]
            expert_data[expert_id].append(embedding)
        
        # Train each expert
        optimizers = [torch.optim.Adam(expert.parameters(), lr=learning_rate) 
                     for expert in self.experts]
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for expert_id in range(self.num_experts):
                if expert_id not in expert_data:
                    continue
                
                expert = self.experts[expert_id]
                optimizer = optimizers[expert_id]
                
                # Get data for this expert
                data = expert_data[expert_id]
                if len(data) == 0:
                    continue
                
                # Create batches
                for i in range(0, len(data), batch_size):
                    batch_data = data[i:i+batch_size]
                    batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    output = expert(batch_tensor)
                    
                    # Reconstruction loss (expert should preserve semantic information)
                    loss = criterion(output, batch_tensor)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Update expert statistics
                expert.training_samples = len(data)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
        
        self.is_trained = True
        print("MOE experts training completed!")
    
    def get_expert_specializations(self) -> Dict[int, Dict[str, Any]]:
        """Get information about each expert's specialization."""
        specializations = {}
        
        for expert_id, expert in enumerate(self.experts):
            info = expert.get_expert_info()
            info['routing_count'] = self.gate.routing_counts.get(expert_id, 0)
            specializations[expert_id] = info
        
        return specializations
    
    def route_scene(self, 
                   scene_embedding: np.ndarray,
                   top_k: int = 1) -> List[Tuple[int, float]]:
        """
        Route a scene embedding to appropriate experts.
        
        Args:
            scene_embedding: CLIP embedding of the scene
            top_k: Number of top experts to return
            
        Returns:
            List of (expert_id, confidence_score) tuples
        """
        with torch.no_grad():
            x = torch.tensor(scene_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
            expert_weights, expert_indices = self.gate(x, top_k)
            
            results = []
            if top_k == 1:
                expert_id = expert_indices[0].item()
                confidence = expert_weights[0, 0].item()
                results.append((expert_id, confidence))
            else:
                for j in range(top_k):
                    expert_id = expert_indices[0, j].item()
                    confidence = expert_weights[0, j].item()
                    results.append((expert_id, confidence))
            
            return results
    
    def process_scene(self, 
                     scene_embedding: np.ndarray,
                     top_k: int = 1) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
        """
        Process a scene through the MOE system.
        
        Args:
            scene_embedding: CLIP embedding of the scene
            top_k: Number of top experts to use
            
        Returns:
            Tuple of (processed_embedding, routing_info)
        """
        with torch.no_grad():
            x = torch.tensor(scene_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
            output, expert_weights, expert_indices = self.forward(x, top_k, return_routing=True)
            
            # Get routing information
            routing_info = []
            if top_k == 1:
                expert_id = expert_indices[0].item()
                confidence = expert_weights[0, 0].item()
                routing_info.append((expert_id, confidence))
            else:
                for j in range(top_k):
                    expert_id = expert_indices[0, j].item()
                    confidence = expert_weights[0, j].item()
                    routing_info.append((expert_id, confidence))
            
            return output.squeeze(0).cpu().numpy(), routing_info
    
    def save_experts(self, filepath: str):
        """Save the MOE experts to disk."""
        save_data = {
            'experts_state_dict': [expert.state_dict() for expert in self.experts],
            'gate_state_dict': self.gate.state_dict(),
            'num_experts': self.num_experts,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'is_trained': self.is_trained,
            'expert_performance': dict(self.expert_performance)
        }
        
        torch.save(save_data, filepath)
        print(f"MOE experts saved to {filepath}")
    
    def load_experts(self, filepath: str):
        """Load the MOE experts from disk."""
        save_data = torch.load(filepath, map_location=self.device)
        
        # Load expert states
        for i, state_dict in enumerate(save_data['experts_state_dict']):
            self.experts[i].load_state_dict(state_dict)
        
        # Load gate state
        self.gate.load_state_dict(save_data['gate_state_dict'])
        
        # Load other data
        self.is_trained = save_data['is_trained']
        self.expert_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.expert_performance.update(save_data['expert_performance'])
        
        print(f"MOE experts loaded from {filepath}")
    
    def visualize_expert_usage(self, save_path: Optional[str] = None):
        """Visualize expert usage patterns."""
        import matplotlib.pyplot as plt
        
        expert_ids = list(range(self.num_experts))
        routing_counts = [self.gate.routing_counts.get(expert_id, 0) for expert_id in expert_ids]
        inference_counts = [expert.inference_count for expert in self.experts]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Routing distribution
        ax1.bar(expert_ids, routing_counts, alpha=0.7, color='lightblue')
        ax1.set_xlabel('Expert ID')
        ax1.set_ylabel('Routing Count')
        ax1.set_title('Expert Routing Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Inference distribution
        ax2.bar(expert_ids, inference_counts, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Expert ID')
        ax2.set_ylabel('Inference Count')
        ax2.set_title('Expert Inference Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 