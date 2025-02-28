
"""
input: query, query_id, candidates_ids
output: pred_dict: {node_id: similarity}
"""

import sys
from pathlib import Path
# Get the absolute path of the current script
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
# Add the project root to the system path
sys.path.append(str(project_root))
import torch
from Reasoning.text_retrievers.stark_model import ModelForSTaRKQA

class Ada(ModelForSTaRKQA):
    def __init__(self, skb, dataset_name, device):
        super(Ada, self).__init__(skb)
        self.emb_dir = f"{project_root}/Reasoning/data/emb/{dataset_name}/"
        self.query_emb_path = self.emb_dir + "text-embedding-ada-002/query/query_emb_dict.pt"
        self.query_emb_dict = torch.load(self.query_emb_path)
        # print(f"777, {self.query_emb_path}")

        self.candidate_emb_path = self.emb_dir + "text-embedding-ada-002/doc/candidate_emb_dict.pt"
        self.candidate_emb_dict = torch.load(self.candidate_emb_path)
        self.device = device

        assert len(self.candidate_emb_dict) == len(self.candidate_ids)

        candidate_embs = [self.candidate_emb_dict[idx].view(1, -1) for idx in self.candidate_ids]
        self.candidate_embs = torch.cat(candidate_embs, dim=0).to(device)

    def score(self, query, q_id, candidate_ids):
        """
            pred_dict[node_id] = similarity (tensor)

        """
        # Dimension of query_emb: torch.Size([1, emb_dim])
        query_emb = self.query_emb_dict[q_id].view(1, -1)
        # Dimension of candidates_embs: torch.Size([num_candidates, emb_dim])
        # # candidates_embs = self.candidate_embs[candidates_ids]
        candi_embs = [self.candidate_emb_dict[c_id].view(1, -1) for c_id in candidate_ids]
        candidates_embs = torch.cat(candi_embs, dim=0).to(self.device)
        # Dimension of similarity: torch.Size([num_candidates])
        similarity = torch.matmul(query_emb.to(self.device), candidates_embs.T).squeeze(dim=0).cpu()
        pred_dict = {}
        for i in range(len(candidate_ids)):
            pred_dict[candidate_ids[i]] = similarity[i].item()
        
        return pred_dict
    
    def retrieve(self, query, q_id, topk, node_type=None):
        # Dimension of query_emb: torch.Size([1, emb_dim])
        query_emb = self.query_emb_dict[q_id].view(1, -1)

        similarity = torch.matmul(query_emb.to(self.device), self.candidate_embs.T).cpu()
        if isinstance(query, str):
            pred_dict = dict(zip(self.candidate_ids, similarity.view(-1)))

        sorted_pred_ids = sorted(pred_dict, key=lambda x: pred_dict[x], reverse=True)
        selected_pred_ids = sorted_pred_ids[:topk]
        pred_dict = {id: pred_dict[id].item() for id in selected_pred_ids}
        print(f"sorted: {pred_dict}")

        return pred_dict
