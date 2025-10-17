"""
Organizing Agent: Trajectory-aware reranker.
Integrates with MoR's existing trajectory features and adds graph-aware features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryFeatureExtractor:
    """
    Extract comprehensive trajectory features.
    Integrates MoR's existing features with new graph-aware features.
    """
    
    def __init__(
        self, 
        graph: nx.Graph, 
        text_encoder: SentenceTransformer,
        device: str = 'cuda'
    ):
        """
        Initialize feature extractor.
        
        Args:
            graph: NetworkX graph of the knowledge base
            text_encoder: SentenceTransformer for text similarity
            device: Device to use
        """
        self.graph = graph
        self.text_encoder = text_encoder
        self.device = device
        
        # Precompute graph metrics (if graph is not empty)
        if len(graph.nodes()) > 0:
            try:
                logger.info("Computing graph centrality metrics...")
                self.pagerank = nx.pagerank(graph, max_iter=100)
                self.betweenness = nx.betweenness_centrality(graph, k=min(100, len(graph.nodes())))
                logger.info("Graph metrics computed successfully")
            except Exception as e:
                logger.warning(f"Could not compute graph metrics: {e}")
                self.pagerank = {node: 0.0 for node in graph.nodes()}
                self.betweenness = {node: 0.0 for node in graph.nodes()}
        else:
            self.pagerank = {}
            self.betweenness = {}
    
    def extract_features(
        self, 
        candidate_node: Any, 
        trajectory: Dict, 
        query: str
    ) -> torch.Tensor:
        """
        Extract rich trajectory features for a candidate.
        
        Args:
            candidate_node: Node ID (can be int or string)
            trajectory: Trajectory dict from reasoning (contains MoR features + alpha)
            query: Query text
        
        Returns:
            features: Feature vector [feature_dim]
        """
        features = []
        
        # 1. Textual fingerprint (from MoR)
        if 'textual_fingerprint' in trajectory:
            tf = trajectory['textual_fingerprint']
            if isinstance(tf, list):
                textual_fp = torch.tensor(tf, dtype=torch.float32, device=self.device)
            else:
                textual_fp = torch.tensor([tf], dtype=torch.float32, device=self.device)
            features.append(textual_fp)
        else:
            # Default if not present
            features.append(torch.zeros(2, dtype=torch.float32, device=self.device))
        
        # 2. Structural fingerprint (from MoR)
        if 'structural_fingerprint' in trajectory:
            sf = trajectory['structural_fingerprint']
            if isinstance(sf, list):
                structural_fp = torch.tensor(sf, dtype=torch.float32, device=self.device)
            else:
                structural_fp = torch.tensor([sf], dtype=torch.float32, device=self.device)
            features.append(structural_fp)
        else:
            features.append(torch.zeros(2, dtype=torch.float32, device=self.device))
        
        # 3. Traversal identifier (from MoR)
        if 'traversal_identifier' in trajectory:
            ti = trajectory['traversal_identifier']
            if isinstance(ti, list):
                traversal_id = torch.tensor(ti, dtype=torch.float32, device=self.device)
            else:
                traversal_id = torch.tensor([ti], dtype=torch.float32, device=self.device)
            features.append(traversal_id)
        else:
            features.append(torch.zeros(2, dtype=torch.float32, device=self.device))
        
        # 4. Graph centrality (NEW)
        if candidate_node in self.pagerank:
            centrality = torch.tensor([
                self.pagerank.get(candidate_node, 0.0),
                self.betweenness.get(candidate_node, 0.0)
            ], dtype=torch.float32, device=self.device)
        else:
            centrality = torch.zeros(2, dtype=torch.float32, device=self.device)
        features.append(centrality)
        
        # 5. Path statistics (NEW)
        if 'all_paths' in trajectory:
            paths = trajectory['all_paths']
            if paths and len(paths) > 0:
                path_lengths = [len(p) for p in paths]
                path_stats = torch.tensor([
                    len(paths),  # Number of paths
                    min(path_lengths),  # Shortest path
                    max(path_lengths),  # Longest path
                    sum(path_lengths) / len(path_lengths)  # Average path length
                ], dtype=torch.float32, device=self.device)
            else:
                path_stats = torch.zeros(4, dtype=torch.float32, device=self.device)
        else:
            path_stats = torch.zeros(4, dtype=torch.float32, device=self.device)
        features.append(path_stats)
        
        # 6. Query-candidate similarity (NEW)
        try:
            with torch.no_grad():
                query_emb = self.text_encoder.encode(
                    query, 
                    convert_to_tensor=True,
                    device=self.device
                )
                
                # Get candidate text
                candidate_text = ""
                if candidate_node in self.graph.nodes():
                    node_data = self.graph.nodes[candidate_node]
                    candidate_text = (
                        node_data.get('text', '') or 
                        node_data.get('title', '') or 
                        node_data.get('name', '') or 
                        str(candidate_node)
                    )
                else:
                    candidate_text = str(candidate_node)
                
                candidate_emb = self.text_encoder.encode(
                    candidate_text, 
                    convert_to_tensor=True,
                    device=self.device
                )
                
                text_sim = F.cosine_similarity(
                    query_emb.unsqueeze(0), 
                    candidate_emb.unsqueeze(0), 
                    dim=1
                )
                features.append(text_sim)
        except Exception as e:
            logger.warning(f"Could not compute text similarity: {e}")
            features.append(torch.zeros(1, dtype=torch.float32, device=self.device))
        
        # 7. Alpha value used in retrieval (NEW)
        if 'alpha' in trajectory:
            alpha_feat = torch.tensor([trajectory['alpha']], dtype=torch.float32, device=self.device)
            features.append(alpha_feat)
        else:
            features.append(torch.zeros(1, dtype=torch.float32, device=self.device))
        
        # 8. Node degree (NEW)
        if candidate_node in self.graph:
            degree = self.graph.degree(candidate_node)
            degree_feat = torch.tensor([degree / 100.0], dtype=torch.float32, device=self.device)  # Normalize
        else:
            degree_feat = torch.zeros(1, dtype=torch.float32, device=self.device)
        features.append(degree_feat)
        
        # 9. Path quality metrics (NEW)
        if 'all_paths' in trajectory and len(trajectory['all_paths']) > 0:
            # Diversity of path types
            unique_path_lengths = len(set(len(p) for p in trajectory['all_paths']))
            path_diversity = unique_path_lengths / len(trajectory['all_paths'])
            
            # Average similarity along paths (if textual_fingerprint has multiple values)
            if 'textual_fingerprint' in trajectory and isinstance(trajectory['textual_fingerprint'], list):
                if len(trajectory['textual_fingerprint']) > 0:
                    avg_similarity = np.mean(trajectory['textual_fingerprint'])
                else:
                    avg_similarity = 0.0
            else:
                avg_similarity = 0.0
            
            path_quality = torch.tensor([
                path_diversity,
                avg_similarity
            ], dtype=torch.float32, device=self.device)
        else:
            path_quality = torch.zeros(2, dtype=torch.float32, device=self.device)
        features.append(path_quality)
        
        # Concatenate all features
        all_features = torch.cat(features)
        
        return all_features


class OrganizingAgent(nn.Module):
    """
    Organizing agent: Reranks candidates using trajectory features.
    Includes value head for PPO training.
    """
    
    def __init__(
        self, 
        feature_dim: int = 50,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Initialize Organizing Agent.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            device: Device to use
        """
        super().__init__()
        
        self.device = device
        self.feature_dim = feature_dim
        
        # Scoring network
        layers = []
        in_dim = feature_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout/2))
            in_dim = hidden_dim
        
        # Final scoring layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.scorer = nn.Sequential(*layers).to(device)
        
        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        ).to(device)
        
        logger.info(f"OrganizingAgent initialized with feature_dim={feature_dim}")
    
    def forward(
        self, 
        features: torch.Tensor,
        temperature: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Score candidates and compute value.
        
        Args:
            features: [num_candidates, feature_dim]
            temperature: Temperature for action probability
        
        Returns:
            scores: [num_candidates] raw scores
            log_prob: Log probability for action (for RL)
            value: Value estimate (scalar)
        """
        # Score candidates
        scores = self.scorer(features).squeeze(-1)
        
        # Convert to probabilities for action log prob
        probs = F.softmax(scores / temperature, dim=0)
        
        # Log probability (sum over all candidates)
        log_prob = torch.log(probs + 1e-10).sum()
        
        # Value estimate (use mean features as state representation)
        mean_features = features.mean(dim=0)
        value = self.value_head(mean_features).squeeze()
        
        return scores, log_prob, value
    
    def rerank(
        self, 
        candidates: List[Any], 
        features: torch.Tensor, 
        k: int = 20,
        temperature: float = 0.5
    ) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
        """
        Rerank candidates based on trajectory features.
        
        Args:
            candidates: List of candidate node IDs
            features: [num_candidates, feature_dim]
            k: Number of top candidates to return
            temperature: Temperature for scoring
        
        Returns:
            reranked_candidates: Top-k reranked candidates
            log_prob: Log probability for action
            value: Value estimate
        """
        # Forward pass
        scores, log_prob, value = self.forward(features, temperature)
        
        # Get top-k indices
        k_actual = min(k, len(candidates))
        top_k_scores, top_k_local_indices = torch.topk(scores, k_actual)
        
        # Map to original candidate IDs
        reranked_candidates = [candidates[i] for i in top_k_local_indices.cpu().tolist()]
        
        return reranked_candidates, log_prob, value
    
    def save_pretrained(self, save_path: str):
        """Save scorer and value head."""
        logger.info(f"Saving OrganizingAgent to {save_path}")
        torch.save({
            'scorer': self.scorer.state_dict(),
            'value_head': self.value_head.state_dict(),
            'feature_dim': self.feature_dim
        }, f"{save_path}/organizing_agent.pt")
        logger.info("OrganizingAgent saved successfully")
    
    def load_pretrained(self, load_path: str):
        """Load scorer and value head."""
        logger.info(f"Loading OrganizingAgent from {load_path}")
        checkpoint = torch.load(f"{load_path}/organizing_agent.pt", map_location=self.device)
        self.scorer.load_state_dict(checkpoint['scorer'])
        self.value_head.load_state_dict(checkpoint['value_head'])
        logger.info("OrganizingAgent loaded successfully")


# Utility function to compute trajectory features for a batch
def extract_batch_features(
    candidate_indices: List[Any],
    trajectories: List[Dict],
    query: str,
    feature_extractor: TrajectoryFeatureExtractor
) -> torch.Tensor:
    """
    Extract features for a batch of candidates.
    
    Args:
        candidate_indices: List of candidate node IDs
        trajectories: List of trajectory dicts (one per candidate)
        query: Query text
        feature_extractor: TrajectoryFeatureExtractor instance
    
    Returns:
        features: [num_candidates, feature_dim]
    """
    features_list = []
    
    for idx, traj in zip(candidate_indices, trajectories):
        feat = feature_extractor.extract_features(idx, traj, query)
        features_list.append(feat)
    
    # Stack into tensor
    if len(features_list) > 0:
        features = torch.stack(features_list)
    else:
        # Return empty tensor if no candidates
        features = torch.empty(0, feature_extractor.text_encoder.get_sentence_embedding_dimension())
    
    return features