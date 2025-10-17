"""
Reasoning Agent: Graph-aware retrieval with learnable fusion.
Combines GNN encoding with textual retrieval using adaptive weighting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphEncoder(nn.Module):
    """
    GNN encoder using GAT (Graph Attention Networks) for graph-aware retrieval.
    Captures structural information from planning graphs.
    """
    
    def __init__(
        self, 
        node_dim: int = 384,  # sentence-transformers/all-MiniLM-L6-v2 output
        hidden_dim: int = 256, 
        output_dim: int = 128, 
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize Graph Encoder.
        
        Args:
            node_dim: Input node feature dimension
            hidden_dim: Hidden dimension for GAT layers
            output_dim: Output embedding dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer: node_dim -> hidden_dim * num_heads
        self.convs.append(
            GATConv(
                node_dim, 
                hidden_dim, 
                heads=num_heads, 
                concat=True,
                dropout=dropout
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Middle layers: hidden_dim * num_heads -> hidden_dim * num_heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads, 
                    hidden_dim, 
                    heads=num_heads, 
                    concat=True,
                    dropout=dropout
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Final layer: hidden_dim * num_heads -> output_dim (single head)
        self.convs.append(
            GATConv(
                hidden_dim * num_heads, 
                output_dim, 
                heads=1, 
                concat=False,
                dropout=dropout
            )
        )
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"GraphEncoder initialized: {num_layers} layers, {num_heads} heads")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GAT layers.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Graph edges [2, num_edges]
            batch: Batch assignment [num_nodes] for batched graphs
        
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Apply GAT layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer (no batch norm, no activation)
        x = self.convs[-1](x, edge_index)
        
        return x


class ReasoningAgent(nn.Module):
    """
    Reasoning agent with graph-aware retrieval and adaptive fusion.
    Combines structural (GNN) and textual (embedding similarity) signals.
    """
    
    def __init__(
        self,
        text_encoder_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        graph_dim: int = 128,
        text_dim: int = 384,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Initialize Reasoning Agent.
        
        Args:
            text_encoder_name: SentenceTransformer model name
            graph_dim: Graph embedding dimension
            text_dim: Text embedding dimension
            hidden_dim: Hidden dimension for fusion network
            num_gnn_layers: Number of GNN layers
            num_heads: Number of attention heads in GAT
            dropout: Dropout probability
            device: Device to use
        """
        super().__init__()
        
        self.device = device
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        
        # Text encoder (frozen)
        logger.info(f"Loading text encoder: {text_encoder_name}")
        self.text_encoder = SentenceTransformer(text_encoder_name, device=device)
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Graph encoder (trainable)
        self.graph_encoder = GraphEncoder(
            node_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=graph_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout
        ).to(device)
        
        # Query projection (project query to graph space for structural comparison)
        self.query_proj = nn.Linear(text_dim, graph_dim).to(device)
        
        # Adaptive fusion network (trainable)
        # Input: [query_emb (text_dim), graph_global_emb (graph_dim), graph_stats (2)]
        fusion_input_dim = text_dim + graph_dim + 2
        
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # alpha in [0, 1]
        ).to(device)
        
        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(device)
        
        logger.info("ReasoningAgent initialized successfully")
    
    def encode_query(self, query_text: str) -> torch.Tensor:
        """
        Encode query to embedding.
        
        Args:
            query_text: Natural language query
            
        Returns:
            Query embedding [text_dim]
        """
        with torch.no_grad():
            query_emb = self.text_encoder.encode(
                query_text, 
                convert_to_tensor=True,
                device=self.device
            )
        return query_emb
    
    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        """
        Encode documents to embeddings.
        
        Args:
            documents: List of document texts
            
        Returns:
            Document embeddings [num_docs, text_dim]
        """
        with torch.no_grad():
            doc_embs = self.text_encoder.encode(
                documents, 
                convert_to_tensor=True,
                device=self.device,
                batch_size=32,  # Process in batches for efficiency
                show_progress_bar=False
            )
        return doc_embs
    
    def _prepare_graph_data(
        self, 
        planning_graph: Dict, 
        documents: List[str]
    ) -> Data:
        """
        Convert planning graph to PyTorch Geometric Data format.
        
        Args:
            planning_graph: Dict with 'nodes' and 'edges'
            documents: List of document texts (for fallback)
            
        Returns:
            PyG Data object
        """
        nodes = planning_graph.get('nodes', [])
        edges = planning_graph.get('edges', [])
        
        # Handle empty graph
        if len(nodes) == 0:
            # Create a dummy single-node graph
            dummy_emb = torch.zeros(1, self.text_dim, device=self.device)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            return Data(x=dummy_emb, edge_index=edge_index, num_nodes=1)
        
        # Encode node texts
        node_texts = []
        for node in nodes:
            if isinstance(node, tuple):
                node_data = node[1] if len(node) > 1 else {}
            else:
                node_data = node
            
            text = node_data.get('text', '') or node_data.get('title', '') or node_data.get('name', '') or ''
            node_texts.append(str(text))
        
        # Encode node features
        with torch.no_grad():
            node_features = self.text_encoder.encode(
                node_texts, 
                convert_to_tensor=True,
                device=self.device,
                batch_size=32,
                show_progress_bar=False
            )
        
        # Build edge index
        edge_list = []
        for edge in edges:
            if isinstance(edge, tuple) and len(edge) >= 2:
                source, target = edge[0], edge[1]
                edge_list.append([source, target])
            elif isinstance(edge, dict):
                source = edge.get('source', edge.get('from', -1))
                target = edge.get('target', edge.get('to', -1))
                if source >= 0 and target >= 0:
                    edge_list.append([source, target])
        
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
        else:
            # No edges - create self-loops for stability
            num_nodes = len(nodes)
            edge_index = torch.stack([
                torch.arange(num_nodes, device=self.device),
                torch.arange(num_nodes, device=self.device)
            ])
        
        data = Data(x=node_features, edge_index=edge_index, num_nodes=len(nodes))
        
        return data
    
    def forward(
        self, 
        query_text: str, 
        planning_graph: Dict, 
        documents: List[str], 
        k: int = 20,
        temperature: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Retrieve candidates using graph-aware hybrid retrieval.
        
        Args:
            query_text: Query string
            planning_graph: Dict with nodes, edges from planning
            documents: List of document texts
            k: Number of candidates to retrieve
            temperature: Temperature for Gumbel-Softmax sampling
        
        Returns:
            candidate_indices: [k] indices of retrieved documents
            log_prob: Log probability for action (for RL)
            value: Estimated value
            alpha: Learned fusion weight (for analysis)
        """
        # 1. Encode query
        query_emb = self.encode_query(query_text)
        
        # 2. Prepare graph data
        graph_data = self._prepare_graph_data(planning_graph, documents)
        
        # 3. Encode graph with GNN
        graph_node_embs = self.graph_encoder(
            graph_data.x.to(self.device), 
            graph_data.edge_index.to(self.device)
        )
        
        # 4. Encode documents (textual)
        doc_embs = self.encode_documents(documents)
        
        # 5. Compute graph statistics
        num_nodes = graph_data.num_nodes
        num_edges = graph_data.edge_index.shape[1]
        
        # Graph density (avoid division by zero)
        max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
        graph_density = num_edges / max_edges
        
        # Average degree
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        
        graph_stats = torch.tensor(
            [graph_density, avg_degree], 
            device=self.device, 
            dtype=torch.float32
        )
        
        # 6. Compute adaptive alpha
        # Aggregate graph embedding (global pooling)
        batch_tensor = torch.zeros(graph_node_embs.shape[0], dtype=torch.long, device=self.device)
        graph_global = global_mean_pool(graph_node_embs, batch_tensor)
        
        # Concatenate features for fusion network
        fusion_input = torch.cat([
            query_emb.unsqueeze(0), 
            graph_global, 
            graph_stats.unsqueeze(0)
        ], dim=-1)
        
        alpha = self.fusion_network(fusion_input).squeeze()
        
        # 7. Compute structural scores
        # Project query to graph space for structural comparison
        query_emb_projected = self.query_proj(query_emb)
        
        # Match graph nodes to documents (assume first N documents correspond to N nodes)
        num_nodes_to_score = min(len(documents), graph_node_embs.shape[0])
        
        if num_nodes_to_score > 0:
            structural_scores = F.cosine_similarity(
                query_emb_projected.unsqueeze(0).expand(num_nodes_to_score, -1),
                graph_node_embs[:num_nodes_to_score],
                dim=1
            )
            # Pad with zeros if we have more documents than nodes
            if len(documents) > num_nodes_to_score:
                padding = torch.zeros(
                    len(documents) - num_nodes_to_score, 
                    device=self.device
                )
                structural_scores = torch.cat([structural_scores, padding])
        else:
            structural_scores = torch.zeros(len(documents), device=self.device)
        
        # 8. Compute textual scores
        textual_scores = F.cosine_similarity(
            query_emb.unsqueeze(0).expand(len(documents), -1),
            doc_embs,
            dim=1
        )
        
        # 9. Combine with learned alpha
        combined_scores = alpha * structural_scores + (1 - alpha) * textual_scores
        
        # 10. Sample top-k using Gumbel-Softmax for differentiability
        # Add Gumbel noise for exploration
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(combined_scores) + 1e-20
        ) + 1e-20)
        
        perturbed_scores = (combined_scores + gumbel_noise) / temperature
        
        # Get top-k
        k_actual = min(k, len(documents))
        top_k_scores, top_k_indices = torch.topk(perturbed_scores, k_actual)
        
        # 11. Compute log probability for policy gradient
        # Use softmax over perturbed scores
        log_probs_all = F.log_softmax(perturbed_scores, dim=0)
        log_prob = log_probs_all[top_k_indices].sum()
        
        # 12. Compute value estimate
        value = self.value_head(fusion_input).squeeze()
        
        return top_k_indices, log_prob, value, alpha.item()
    
    def save_pretrained(self, save_path: str):
        """Save trainable components."""
        logger.info(f"Saving ReasoningAgent to {save_path}")
        torch.save({
            'graph_encoder': self.graph_encoder.state_dict(),
            'query_proj': self.query_proj.state_dict(),
            'fusion_network': self.fusion_network.state_dict(),
            'value_head': self.value_head.state_dict()
        }, f"{save_path}/reasoning_agent.pt")
        logger.info("ReasoningAgent saved successfully")
    
    def load_pretrained(self, load_path: str):
        """Load trainable components."""
        logger.info(f"Loading ReasoningAgent from {load_path}")
        checkpoint = torch.load(f"{load_path}/reasoning_agent.pt", map_location=self.device)
        self.graph_encoder.load_state_dict(checkpoint['graph_encoder'])
        self.query_proj.load_state_dict(checkpoint['query_proj'])
        self.fusion_network.load_state_dict(checkpoint['fusion_network'])
        self.value_head.load_state_dict(checkpoint['value_head'])
        logger.info("ReasoningAgent loaded successfully")