import sys
from pathlib import Path
# Get the absolute path of the current script
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
# Add the project root to the system path
sys.path.append(str(project_root))

import random
import torch
import os
from stark_qa.evaluator import Evaluator
import torch.nn as nn
from typing import Any, Union, List, Dict


class ModelForSTaRKQA(nn.Module):
    
    def __init__(self, skb, query_emb_dir='.'):
        """
        Initializes the model with the given knowledge base.
        
        Args:
            skb: Knowledge base containing candidate information.
        """
        super(ModelForSTaRKQA, self).__init__()
        self.skb = skb

        self.candidate_ids = skb.candidate_ids
        self.evaluator = Evaluator(self.candidate_ids)
        
    def evaluate(self, 
                 pred_dict: Dict[int, float], 
                 answer_ids: Union[torch.LongTensor, List[Any]], 
                 metrics: List[str] = ['mrr', 'hit@3', 'recall@20'], 
                 **kwargs: Any) -> Dict[str, float]:
        """
        Evaluates the predictions using the specified metrics.
        
        Args:
            pred_dict (Dict[int, float]): Predicted answer ids or scores.
            answer_ids (torch.LongTensor): Ground truth answer ids.
            metrics (List[str]): A list of metrics to be evaluated, including 'mrr', 'hit@k', 'recall@k', 
                                 'precision@k', 'map@k', 'ndcg@k'.
                             
        Returns:
            Dict[str, float]: A dictionary of evaluation metrics.
        """
        return self.evaluator(pred_dict, answer_ids, metrics)
    
    def evaluate_batch(self, 
                pred_ids: List[int],
                pred: torch.Tensor, 
                answer_ids: Union[torch.LongTensor, List[Any]], 
                metrics: List[str] = ['mrr', 'hit@3', 'recall@20'], 
                **kwargs: Any) -> Dict[str, float]:
        return self.evaluator.evaluate_batch(pred_ids, pred, answer_ids, metrics)
    

def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def move_to_device(sample, device=None):
    """Move tensors to specified device (CPU or CUDA)"""
    if device is None:
        device = get_device()
    
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_device(value)
                for key, value in maybe_tensor.items()
            }
        else:
            return maybe_tensor

    return _move_to_device(sample)

# Keep old function name for backward compatibility
def move_to_cuda(sample):
    """Deprecated: use move_to_device instead"""
    return move_to_device(sample, device='cpu')  # Force CPU


if __name__ == "__main__":
    print("Testing Utils")
