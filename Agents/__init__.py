"""
MoR Agents Module
Multi-Agent Reinforcement Learning for Text-Rich Graph Retrieval

Contains 3 trainable agents for joint optimization:
1. PlanningAgent - Generates Cypher queries with LoRA-tuned LLM
2. ReasoningAgent - Graph-aware retrieval with GNN + adaptive fusion  
3. OrganizingAgent - Trajectory-aware reranking with value estimation

Phase 1: Agent Modules (COMPLETE)
Phase 2: Training Infrastructure (reward_functions.py, ppo_trainer.py)
Phase 3: Data Preparation (prepare_joint_training_data.py)
Phase 4: Training Script (train_joint.py)
Phase 5: Evaluation (eval_joint.py)
"""

from .planning_agent import PlanningAgent
from .reasoning_agent import ReasoningAgent, GraphEncoder
from .organizing_agent import (
    OrganizingAgent, 
    TrajectoryFeatureExtractor,
    extract_batch_features
)

__all__ = [
    'PlanningAgent',
    'ReasoningAgent',
    'GraphEncoder',
    'OrganizingAgent',
    'TrajectoryFeatureExtractor',
    'extract_batch_features'
]

__version__ = '1.0.0'
__author__ = 'MoR Joint Training Team'
__phase__ = 'Phase 1: Agent Modules - COMPLETE'