import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
from stark_qa import load_skb


from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

from Reranking.utils import move_to_cuda, seed_everything
from Reranking.rerankers.path import PathReranker
import torch.nn.functional as F
import argparse
import pickle as pkl




class TestDataset(Dataset):
    """        
        data format: {
            "query": query,
            "pred_dict": {node_id: score},
            'score_vector_dict': {node_id: [bm25, bm_25, bm25, ada]},
            "text_emb_dict": {node_id: text_emb},
            "ans_ids": [],
        }

"""

    def __init__(self, saved_data, args):
        
        print(f"Start processing test dataset...")
        self.text2emb_dict = saved_data['text2emb_dict']
        self.data = saved_data['data']  
        
        self.text_emb_matrix = list(self.text2emb_dict.values())
        self.text_emb_matrix = torch.concat(self.text_emb_matrix, dim=0)
        
        # make the mapping between the key of text2emb_dict and the index of text_emb_matrix
        self.text2idx_dict = {key: idx for idx, key in enumerate(self.text2emb_dict.keys())}
        
        self.args = args
        
             
        
      
    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        
        if self.args.dataset_name == 'amazon':
            # change from the str to index
            self.data[idx]['text_emb_dict'] = {key: self.text2idx_dict[value] for key, value in self.data[idx]['text_emb_dict'].items()}
        else:
            # sort the pred_dict by the score
            pred_dict = self.data[idx]['pred_dict']
            sorted_ids = sorted(pred_dict.keys(), key=lambda x: pred_dict[x], reverse=True)
            # get the top 50 candidates
            sorted_ids = sorted_ids[:50]
            # get the score vector
            self.data[idx]['score_vector_dict'] = {key: self.data[idx]['score_vector_dict'][key] for key in sorted_ids}
            # get the symb_enc_dict
            self.data[idx]['symb_enc_dict'] = {key: self.data[idx]['symb_enc_dict'][key] for key in sorted_ids}
            # change from the str to index
            self.data[idx]['text_emb_dict'] = {key: self.text2idx_dict[value] for key, value in self.data[idx]['text_emb_dict'].items()}
            self.data[idx]['text_emb_dict'] = {key: self.data[idx]['text_emb_dict'][key] for key in sorted_ids}
    
                

        return self.data[idx]
    
    
    def collate_batch(self, batch):
        
        # q
        batch_q = [batch[i]['query'] for i in range(len(batch))]
        q_text = batch_q
        
        # c
        batch_c = [list(batch[i]['score_vector_dict'].keys()) for i in range(len(batch))] # [batch, 100]
        batch_c = torch.tensor(batch_c)
        c_score_vector = [list(batch[i]['score_vector_dict'].values()) for i in range(len(batch))] # [batch, 100, 4]
        c_score_vector = torch.tensor(c_score_vector)
        c_score_vector = c_score_vector[:, :, :self.args.vector_dim]
        
        # c_symb_enc
        c_symb_enc = [list(batch[i]['symb_enc_dict'].values()) for i in range(len(batch))]
        c_symb_enc = torch.tensor(c_symb_enc) # [bs, 100, 3]
    
        # c_text_emb
        c_text_emb = [self.text_emb_matrix[list(batch[i]['text_emb_dict'].values())].unsqueeze(0) for i in range(len(batch))]
        c_text_emb = torch.concat(c_text_emb, dim=0) # [bs, 100, 768]
        
        
        # ans_ids
        ans_ids = [batch[i]['ans_ids'] for i in range(len(batch))] # list of ans_ids
        
        # pred_ids
        pred_ids = batch_c.tolist()
            
        
        # Create a dictionary for the batch
        feed_dict = {
            'query': q_text,
            'c_score_vector': c_score_vector,
            'c_text_emb': c_text_emb,
            'c_symb_enc': c_symb_enc,
            'ans_ids': ans_ids,
            'pred_ids': pred_ids

        }
        
        
        return feed_dict
    
    
#  ***** batch_evaluator *****
def batch_evaluator(skb, scores_cand, ans_ids, batch):

    results = {}
    
    # **** batch wise evaluation ****
    # evaluate
    candidates_ids = skb.candidate_ids
    id_to_idx = {candidate_id: idx for idx, candidate_id in enumerate(candidates_ids)}
    
    
    # initialize the pred_matrix
    pred_matrix = torch.zeros((scores_cand.shape[0],len(candidates_ids)))
    
    
    # get the index of each pred_ids
    # flatten the pred_ids
    flat_pred_ids = torch.tensor(batch['pred_ids']).flatten().tolist()
    
    
    # get the index of each pred_ids
    pred_idx = [id_to_idx[pred_id] for pred_id in flat_pred_ids]
    
    
    # reshape the pred_idx
    pred_idx = torch.tensor(pred_idx).reshape(scores_cand.shape[0], -1) # [bs, 100]
    
    # move pred_matrix to the device
    pred_matrix = pred_matrix.to(scores_cand.device)
    
    # advanced indexing
    pred_matrix[torch.arange(scores_cand.shape[0]).unsqueeze(1), pred_idx] = scores_cand.squeeze(-1) # [bs, num_candidates]
    
    
    # Create a mapping from candidate IDs to their indices for faster lookup
    

    # Flatten ans_ids to a single list and map them to indices
    flat_ans_idx = [id_to_idx[a_id] for sublist in ans_ids for a_id in sublist]

    # Create the row indices for ans_matrix corresponding to the answers
    row_indices = torch.repeat_interleave(torch.arange(len(ans_ids)), torch.tensor([len(sublist) for sublist in ans_ids]))

    # Create the answer matrix
    ans_matrix = torch.zeros((scores_cand.shape[0], len(candidates_ids)), device=scores_cand.device)
    ans_matrix[row_indices, torch.tensor(flat_ans_idx, device=scores_cand.device)] = 1


    
    # batch computing hit1
    # find the index of the max score
    max_score, max_idx = torch.max(pred_matrix, dim=1)
    # check the label of the max idx
    batch_hit1 = ans_matrix[torch.arange(scores_cand.shape[0]), max_idx]
    hit1_list = batch_hit1.tolist()
    
    
    # batch computing hit@5
    _, top5_idx = torch.topk(pred_matrix, 5, dim=1)
    batch_hit5 = ans_matrix[torch.arange(scores_cand.shape[0]).unsqueeze(1), top5_idx]
    
    # max with each row
    batch_hit5 = torch.max(batch_hit5, dim=1)[0]
    hit5_list = batch_hit5.tolist()
    
    
    
    # batch computing recall@20
    _, top20_idx = torch.topk(pred_matrix, 20, dim=1)
    batch_recall20 = ans_matrix[torch.arange(scores_cand.shape[0]).unsqueeze(1), top20_idx]
    # sum with each row
    batch_recall20 = torch.sum(batch_recall20, dim=1)
    # divide by the sum of the ans_matrix along the row
    batch_recall20 = batch_recall20 / torch.sum(ans_matrix, dim=1)
    recall20_list = batch_recall20.tolist()
    
    
    
    # batch computing mrr
    # find the highest rank of the answer
    _, rank_idx = torch.sort(pred_matrix, dim=1, descending=True)
    # query the answer matrix with the rank_idx
    batch_mrr = ans_matrix[torch.arange(scores_cand.shape[0]).unsqueeze(1), rank_idx]
    # find the first rank of the answer
    batch_mrr = torch.argmax(batch_mrr, dim=1)
    # add 1 to the rank
    batch_mrr += 1
    # divide by the rank
    batch_mrr = 1 / batch_mrr.float()
    mrr_list = batch_mrr.tolist()
    

    results['hit@1'] = hit1_list
    results['hit@5'] = hit5_list
    results['recall@20'] = recall20_list
    results['mrr'] = mrr_list
    

    return results
    


# ***** evaluate *****
@torch.no_grad()
def evaluate(router, test_loader, skb):

    
    router.eval()

    all_results = {
        "hit@1": [],
        "hit@5": [],
        "recall@20": [],
        "mrr": []
    }
    avg_results = {
        "hit@1": 0,
        "hit@5": 0,
        "recall@20": 0,
        "mrr": 0
    }
    
    
    # save the scores and ans_ids, and pred_ids
    pred_list = []
    scores_cand_list = []
    ans_ids_list = []
    print(f"Start evaluating...")
    # use tqdm to show the progress
    for idx, batch in enumerate(tqdm(test_loader, desc='Evaluating', position=0)):
        # print(f"idx: {idx}")
        batch = move_to_cuda(batch)
        
        # Check if the model is wrapped in DataParallel
        if isinstance(router, nn.DataParallel):
            scores_cand = router.module.eval_batch(batch) # q_emb: [bs, 100], c_emb: [bs*100, 100]
        else:
            scores_cand = router.eval_batch(batch)


        # ans_ids
        ans_ids = batch['ans_ids']
           
        results = batch_evaluator(skb, scores_cand, ans_ids, batch)
                
        
        for key in results.keys():
            all_results[key].extend(results[key])
            
        # save the scores and ans_ids, and pred_ids
        pred_list.extend(batch['pred_ids'])
        scores_cand_list.extend(scores_cand.cpu().tolist())
        ans_ids_list.extend(ans_ids)
        
    
    
    for key in avg_results.keys():
        avg_results[key] = np.mean(all_results[key])
    
    print(f"Results: {avg_results}")
    

    
    return avg_results


def parse_args():
    
    parser = argparse.ArgumentParser(description="Run PathRouter with dynamic combinations of embeddings.")
    
    # dataset_name
    parser.add_argument("--dataset_name", type=str, default="mag", help="Name of the dataset.")
    
    # Add arguments for model configurations
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (e.g., 'cuda' or 'cpu').")

    
    # add concat_num
    parser.add_argument("--concat_num", type=int, default=0, help="Number of concatenation of embeddings.")
    
    # checkpoint save path
    parser.add_argument("--checkpoint_path", type=str, default="./data/checkpoints", help="Path saves the checkpoints.")
    
    # similarity vector dim
    parser.add_argument("--vector_dim", type=int, default=4, help="Dimension of the similarity vector.")
    
    
    # Parse the base arguments
    args = parser.parse_args()
    return args


def get_concat_num(combo):
    """
    Determine the value of concat_num based on the combination of embeddings.
    - score_vec adds +1
    - text_emb adds +1
    - symb_enc adds +3
    """
    concat_num = 0
    if combo.get("score_vec", False):  # If score_vec is True
        concat_num += 1
    if combo.get("text_emb", False):  # If text_emb is True
        concat_num += 1
    if combo.get("symb_enc", False):  # If symb_enc is True
        concat_num += 3
        
        
    return concat_num


def run(test_data, skb, dataset_name):
    
    
    
    test_size = 64
    test_dataset = TestDataset(test_data, args=args)
    test_loader = DataLoader(test_dataset, batch_size=test_size, num_workers=32, collate_fn=test_dataset.collate_batch)
    
    # load the model
    print(f"Load the model...")
    args.checkpoint_path = args.checkpoint_path + f"/{dataset_name}/best.pth"
    router = PathReranker(socre_vector_input_dim=4, text_emb_input_dim=768, symb_enc_dim=3, args=args)
    checkpoint = torch.load(args.checkpoint_path)
    router.load_state_dict(checkpoint)
    router = router.to(args.device)
    
    # evalute
    test_results = evaluate(router, test_loader, skb)
    print(f"Test evaluation")
    print(test_results)
    
    return test_results

if __name__ == "__main__":
    
    combo = {
        "text_emb": True,
        "score_vec": True,
        "symb_enc": True
    }
    concat_num = get_concat_num(combo)
    
    base_args = parse_args()
    args = argparse.Namespace(**vars(base_args), **combo)
    args.concat_num = concat_num
    dataset_name = args.dataset_name
    
    test_data_path = f"../{dataset_name}_test.pkl"
    with open(test_data_path, 'rb') as f:
        test_data = pkl.load(f)
    skb = load_skb(dataset_name)
    results = run(test_data, skb, dataset_name)
    