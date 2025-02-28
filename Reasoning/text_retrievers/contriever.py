
"""
input: query, query_id, candidates_ids
output: pred_dict: {node_id: similarity}
"""
import heapq
import sys
from pathlib import Path
# Get the absolute path of the current script
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
# Add the project root to the system path
sys.path.append(str(project_root))
from stark_qa.tools.api_lib.openai_emb import get_contriever, get_contriever_embeddings
import torch
from Reasoning.text_retrievers.stark_model import ModelForSTaRKQA

class Contriever(ModelForSTaRKQA):
    def __init__(self, skb, dataset_name, device):
        super(Contriever, self).__init__(skb)
        self.emb_dir = f"{project_root}/Reasoning/data/emb/{dataset_name}/"

        self.query_emb_path = self.emb_dir + "contriever/query_no_rel_no_compact/query_emb_dict.pt"
        self.query_emb_dict = torch.load(self.query_emb_path)

        self.candidate_emb_path = self.emb_dir + "contriever/doc_no_rel_no_compact/candidate_emb_dict.pt"
        self.candidate_emb_dict = torch.load(self.candidate_emb_path)
        self.device = device

        assert len(self.candidate_emb_dict) == len(self.candidate_ids)

        candidate_embs = [self.candidate_emb_dict[idx].view(1, -1) for idx in self.candidate_ids]
        self.candidate_embs = torch.cat(candidate_embs, dim=0).to(device)
        
        # load contriever for query embeddings
        self.encoder, self.tokenizer = get_contriever(dataset_name=dataset_name)    
        self.encoder = self.encoder.to(device)    
        

    def score(self, query, q_id, candidate_ids):
        """
            pred_dict[node_id] = similarity (tensor)

        """
        
        
        # Dimension of query_emb: torch.Size([1, emb_dim])
        query_emb = self.query_emb_dict[q_id].view(1, -1)

        # Dimension of candidates_embs: torch.Size([num_candidates, emb_dim])
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
        query_emb = get_contriever_embeddings(query, encoder=self.encoder, tokenizer=self.tokenizer, device=self.device)
        
        similarity = torch.matmul(query_emb.to(self.device), self.candidate_embs.T).cpu()
        
        if isinstance(query, str):
            pred_dict = dict(zip(self.candidate_ids, similarity.view(-1)))
        
    
        selected_pred_ids = heapq.nlargest(topk, pred_dict, key=pred_dict.get)
        pred_dict = {id: pred_dict[id].item() for id in selected_pred_ids}


        return pred_dict

if __name__ == '__main__':
    print("Testing Contriever")