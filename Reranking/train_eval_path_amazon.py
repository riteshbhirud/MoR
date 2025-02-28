import sys
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import wandb
import numpy as np
import time
from torch_scatter import segment_csr, scatter_mean
from itertools import product
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import random
from collections import defaultdict
import os

from Reranking.utils import move_to_cuda, seed_everything
from Reranking.rerankers.path import PathReranker
from utils import ModelForSTaRKQA
from stark_qa import load_qa, load_skb
import torch.nn.functional as F
import argparse
import json
import time


seed_everything(42)

# ***** Dataset *****
class TrainDataset(Dataset):
    """
    Custom Dataset for the training data.
    Each instance contains multiple positive and negative candidates.
    """
    def __init__(self, saved_data, max_neg_candidates=100):
        """
            10s for 1000 data 
        """
        print(f"start processing training dataset...")
        s_time = time.time()
        self.max_neg_candidates = max_neg_candidates        
        self.sorted_query2neg = defaultdict(list)
        

        self.text2emb_dict = saved_data['text2emb_dict']
        self.data = saved_data['data']  

        
        # separage neg and pos, and prepare query, pos pairs
        new_data = []
        
        for i in range(len(self.data)):
            neg_ids = []
            pos_ids = []
            item = self.data[i]
            
            
            candidates_dict = item['pred_dict']
            ans_ids = item['ans_ids']
            # pos_ids = ans_ids
            for ans_id in ans_ids:
                if ans_id in candidates_dict.keys():
                    pos_ids.append(ans_id)
            neg_ids = list(set(candidates_dict.keys()) - set(pos_ids))
            
            # load scores vector
            score_vector_dict = item['score_vector_dict']
            
            # load the text path, str format
            text_emb_dict = item['text_emb_dict']
            
            # load the symb_enc_dict
            symb_enc_dict = item['symb_enc_dict']

            
            self.data[i]['pos_ids'] = pos_ids
            self.data[i]['neg_ids'] = neg_ids
            
            query = item['query']
            for pos_id in pos_ids:
                new_data.append((query, score_vector_dict[pos_id], self.text2emb_dict[text_emb_dict[pos_id]], symb_enc_dict[pos_id]))
                
            
            # print(f"new_data: {new_data}")

            neg_dict = {neg_id: candidates_dict[neg_id] for neg_id in neg_ids}
            sorted_neg_ids = sorted(neg_dict.keys(), key=lambda x: neg_dict[x], reverse=True) # return list
            
                
            self.sorted_query2neg[query] = [(score_vector_dict[neg_id], self.text2emb_dict[text_emb_dict[neg_id]], symb_enc_dict[neg_id]) for neg_id in sorted_neg_ids]        
                
            
        self.data = new_data
        print(f"Complete data preparation")
        print(f"Time: {time.time() - s_time}")
        
        
        

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):

        return self.data[idx]
        
    def collate_batch(self, pairs):
        s_time = time.time()
        
        # q
        batch_q = [pair[0] for pair in pairs] # q is text
        q_text = batch_q
        # print(f"q111, {q_text}")
        
        
        # pos
        # get the score vector
        batch_p_score_vector = [pair[1] for pair in pairs] # p is score vector
        batch_p_score_vector = torch.tensor(batch_p_score_vector) # [bs, 4]
        batch_p_score_vector = batch_p_score_vector[:, :args.vector_dim]
        # get the text emb
        batch_p_text_emb = [pair[2] for pair in pairs] # p is text emb
        batch_p_text_emb = torch.concat(batch_p_text_emb, dim=0) # [bs, 768]
        # get the symb_enc
        batch_p_symb_enc = [pair[3] for pair in pairs] # p is symb_enc
        batch_p_symb_enc = torch.tensor(batch_p_symb_enc) # [bs, 3]
        
        
        # Negative samples
        batch_n = [random.choices(self.sorted_query2neg[query], k=self.max_neg_candidates) for query in batch_q] # allow duplicates
        
        
        # get the score vector
        batch_n_score_vector = [pair[0] for sublist in batch_n for pair in sublist]
        batch_n_score_vector = torch.tensor(batch_n_score_vector) # [bs*100, 4]
        # reshape to [bs, 100, 4]
        batch_n_score_vector = batch_n_score_vector.reshape(len(batch_q), self.max_neg_candidates, -1) # [bs, 100, 4]
        batch_n_score_vector = batch_n_score_vector[:, :, :args.vector_dim]
        
        # get the text emb
        batch_n_text_emb = [pair[1] for sublist in batch_n for pair in sublist] 
        batch_n_text_emb = torch.concat(batch_n_text_emb, dim=0) # [bs*100, 768]
        # reshape to [bs, 100, 768]
        batch_n_text_emb = batch_n_text_emb.reshape(len(batch_q), self.max_neg_candidates, -1) # [bs, 100, 768]
        
        # get the symb_enc
        batch_n_symb_enc = [pair[2] for sublist in batch_n for pair in sublist]
        batch_n_symb_enc = torch.tensor(batch_n_symb_enc) # [bs*100, 3]
        # reshape to [bs, 100, 3]
        batch_n_symb_enc = batch_n_symb_enc.reshape(len(batch_q), self.max_neg_candidates, -1) # [bs, 100, 3]
            
        
        
        
        
        # Create a dictionary for the batch
        feed_dict = {
            'query': q_text,
            'p_score_vector': batch_p_score_vector,
            'p_text_emb': batch_p_text_emb,
            'p_symb_enc': batch_p_symb_enc,
            'n_score_vector': batch_n_score_vector,
            'n_text_emb': batch_n_text_emb,
            'n_symb_enc': batch_n_symb_enc,
            
        }


        return feed_dict
    

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

    def __init__(self, saved_data):
        
        print(f"Start processing test dataset...")
        self.text2emb_dict = saved_data['text2emb_dict']
        self.data = saved_data['data']  
        
        self.text_emb_matrix = list(self.text2emb_dict.values())
        self.text_emb_matrix = torch.concat(self.text_emb_matrix, dim=0)
        
        # make the mapping between the key of text2emb_dict and the index of text_emb_matrix
        self.text2idx_dict = {key: idx for idx, key in enumerate(self.text2emb_dict.keys())}
        
        print(f"Complete data preparation: {len(self.data)}")
        
             
        
      
    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        # ***** amazon *****
        # change from the str to index
        self.data[idx]['text_emb_dict'] = {key: self.text2idx_dict[value] for key, value in self.data[idx]['text_emb_dict'].items()}
        
        return self.data[idx]
    
    def collate_batch(self, batch):
        
        # q
        batch_q = [batch[i]['query'] for i in range(len(batch))]
        q_text = batch_q
        
        # c
        batch_c = [list(batch[i]['score_vector_dict'].keys()) for i in range(len(batch))] # [batch, 100]
        batch_c = torch.tensor(batch_c)
        # print(f"111, {batch_c.shape}")
        c_score_vector = [list(batch[i]['score_vector_dict'].values()) for i in range(len(batch))] # [batch, 100, 4]
        c_score_vector = torch.tensor(c_score_vector)[:, :, :args.vector_dim] # [batch, 100, 4]
        
        
        # print(f"222, {c_vector.shape}")
        # c_text_emb
        # c_text_emb = [torch.concat(list(batch[i]['text_emb_dict'].values()), dim=0).unsqueeze(0) for i in range(len(batch))]
        c_text_emb = [self.text_emb_matrix[list(batch[i]['text_emb_dict'].values())].unsqueeze(0) for i in range(len(batch))]
        c_text_emb = torch.concat(c_text_emb, dim=0) # [bs, 100, 768]
        
        # c_symb_enc
        c_symb_enc = [list(batch[i]['symb_enc_dict'].values()) for i in range(len(batch))]
        c_symb_enc = torch.tensor(c_symb_enc) # [bs, 100, 3]
        
        
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
    

# ******* loss function ********
def loss_fn(scores_pos, scores_neg):
    
    loss_fct = CrossEntropyLoss(ignore_index=-1)

    # Combine scores
    scores = torch.cat([scores_pos, scores_neg.squeeze(-1)], dim=1)  # B x (1 + max_neg_candidates*B)
    # print(f"scores: {scores.shape}")
    
    # Create target
    target = torch.zeros(scores.size(0), dtype=torch.long).to(scores.device)

    # Compute loss
    loss = loss_fct(scores, target)
    
    return loss

# ***** pairwise loss *****
def pairwise_loss(scores_pos, scores_neg, margin=0.5):
    # scores_pos: [bs, 1]
    # scores_neg: [bs, 100, 1]

    # Compute loss
    differences = scores_pos.unsqueeze(1) - scores_neg - margin # [bs, 100, 1]
    differences = differences.view(-1)  # [bs*100]
    loss = F.relu(-differences).mean()  # Standard pairwise loss

    return loss


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
def evaluate(reranker, test_loader):

    
    reranker.eval()

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
    # use tqdm to show the progress
    for idx, batch in enumerate(tqdm(test_loader, desc='Evaluating', position=0)):
        batch = move_to_cuda(batch)
        
        # Check if the model is wrapped in DataParallel
        if isinstance(reranker, nn.DataParallel):
            scores_cand = reranker.module.eval_batch(batch) # q_emb: [bs, 100], c_emb: [bs*100, 100]
        else:
            scores_cand = reranker.eval_batch(batch)


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
    
    
# ***** train *****
def main(train_data, val_data, test_data, skb, dataset_name, args):


    epochs = args.epochs
    device = args.device

    train_size = args.train_batch_size
    test_size = 64

    train_dataset = TrainDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=train_size, num_workers=32, collate_fn=train_dataset.collate_batch, drop_last=True)

    test_dataset = TestDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=test_size, num_workers=32, collate_fn=test_dataset.collate_batch)
    
    val_dataset = TestDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=test_size, num_workers=32, collate_fn=val_dataset.collate_batch)
    

    # ***** Model *****
    reranker = PathReranker(socre_vector_input_dim=args.vector_dim, text_emb_input_dim=768, symb_enc_dim=3, args=args)
    save_dir = f"./data/checkpoints/{dataset_name}/path"
    os.makedirs(save_dir, exist_ok=True)

    reranker.to(device)
    # # parallel processing
    reranker = nn.DataParallel(reranker)


    optimizer = torch.optim.Adam(reranker.parameters(), lr=args.lr)
    best_val_hit1 = float('-inf')

    
    val_results = evaluate(reranker, val_loader)
    print(f"Val evaluation")
    print(val_results)
    
    
    test_results = evaluate(reranker, test_loader)
    print(f"Test evaluation")
    print(test_results)
    
    # log both val and test results
    wandb.log({'val_mrr': val_results['mrr'], 'val_hit1': val_results['hit@1'], 'val_hit5': val_results['hit@5'], 'val_recall@20': val_results['recall@20'],
               'test_mrr': test_results['mrr'], 'test_hit1': test_results['hit@1'], 'test_hit5': test_results['hit@5'], 'test_recall@20': test_results['recall@20']})
    
    best_test_results = {}
    for epoch in tqdm(range(epochs), desc='Training Epochs', position=0):
        total_loss = 0.0
        reranker.train()
        count = 0
        total_instances = 0

        for batch in tqdm(train_loader):
            # print(batch)
            batch = move_to_cuda(batch)
            # print(batch)
            
            scores_pos, scores_neg = reranker(batch)
            
            # batch_loss = pairwise_loss(scores_pos, scores_neg)
            batch_loss = loss_fn(scores_pos, scores_neg)
            
            # clear optimizer 
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            # total_loss += batch_loss.item()
            count += 1 
            # compute the average loss
            total_instances += scores_pos.shape[0]
            total_loss += batch_loss.item()
                        
            
        train_loss = total_loss / total_instances
        
        print(f"Epoch {epoch+1}/{epochs}, Average Train Loss: {train_loss}")
        
        
        
        val_results = evaluate(reranker, val_loader)
        print(f"Val evaluation")
        print(val_results)
        
        
        test_results = evaluate(reranker, test_loader)
        print(f"Test evaluation")
        print(test_results)
        
        # log both val and test results
        wandb.log({'val_mrr': val_results['mrr'], 'val_hit1': val_results['hit@1'], 'val_hit5': val_results['hit@5'], 'val_recall@20': val_results['recall@20'],
                'test_mrr': test_results['mrr'], 'test_hit1': test_results['hit@1'], 'test_hit5': test_results['hit@5'], 'test_recall@20': test_results['recall@20'],
                'train_loss': train_loss})
        
        
        # save the best model when val hit1 is the highest
        hit1 = val_results['hit@1']
        if best_val_hit1 < hit1:
            best_val_hit1 = hit1
            
            save_path = f"{save_dir}/best_{best_val_hit1}.pth"
            
            if isinstance(reranker, nn.DataParallel):
                torch.save(reranker.module.state_dict(), save_path)
            else:
                torch.save(reranker.state_dict(), save_path)
            print(f"Checkpoint saved at epoch {epoch+1} with test hits@1 {hit1}")
            
            args.checkpoint_path = save_path
            best_test_results = test_results
        
        
                
    # save last epoch checkopint
    save_path = f"{save_dir}/last_{hit1}.pth"
    if isinstance(reranker, nn.DataParallel):
        torch.save(reranker.module.state_dict(), save_path)
    else:
        torch.save(reranker.state_dict(), save_path)
    print(f"Final checkpoint saved at {save_path}")
    
    
    
    # ***** save the results *****
    results = []
    results.append(
        {
            "config": vars(args),
            "test_results": best_test_results
        }
    )
    # save the results to json
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"./data/outputs/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(best_test_results)
    
    
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
    
    
    
def parse_args():
    
    parser = argparse.ArgumentParser(description="Run Pathreranker with dynamic combinations of embeddings.")
    
    # Add arguments for model configurations
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training or evaluation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (e.g., 'cuda' or 'cpu').")
    
    # Add arguments for the dataset
    parser.add_argument("--dataset_name", type=str, default="amazon", help="Name of the dataset to use.")
    # paths
    parser.add_argument("--train_path", type=str, default=f"../amazon_train.pkl", help="Path to the training data.")
    parser.add_argument("--test_path", type=str, default=f"../amazon_test.pkl", help="Path to the test data.")
    parser.add_argument("--val_path", type=str, default=f"../amazon_val.pkl", help="Path to the validation data.")
    
    # add concat_num
    parser.add_argument("--concat_num", type=int, default=0, help="Number of concatenation of embeddings.")
    
    # checkpoint save path
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to save the checkpoints.")
    parser.add_argument("--vector_dim", type=int, default=4, help="Dimension of the similarity vector.")
    
    
    # Parse the base arguments
    args = parser.parse_args()
    return args
    
    
if __name__ == "__main__":
    
    base_args = parse_args()
    test_path = base_args.test_path
    train_path = base_args.train_path
    val_path = base_args.val_path
    dataset_name = base_args.dataset_name
    
    with open(test_path, "rb") as f:
        test_data = pkl.load(f)
        
    with open(train_path, "rb") as f:
        train_data = pkl.load(f)
        
    with open(val_path, "rb") as f:
        val_data = pkl.load(f)
    
    # load skb
    skb = load_skb(dataset_name)
    
    # set all
    combo = {
        "text_emb": True,
        "score_vec": True,
        "symb_enc": True
    }
    concat_num = get_concat_num(combo)
    
    wandb.init(project=f'reranking-{dataset_name}', name=f"path")
    args = argparse.Namespace(**vars(base_args), **combo)
    args.concat_num = concat_num
    
    
    main(train_data, val_data, test_data, skb, dataset_name, args)
        