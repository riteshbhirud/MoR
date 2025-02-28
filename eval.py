import argparse
import sys


from Reasoning.mor4path import MOR4Path
from Planning.model import Planner
from prepare_rerank import prepare_trajectories
from tqdm import tqdm
import os
import pickle as pkl
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from stark_qa import load_qa, load_skb
import torch.nn as nn
        


# make model_name a argument
parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="mag")
# text retriever name
parser.add_argument("--text_retriever_name", type=str, default="bm25")
parser.add_argument("--scorer_name", type=str, default="ada", help="contriever, ada") # contriever for prime, ada for amazon and mag
# mod
parser.add_argument("--mod", type=str, default="test", help="train, valid, test")
# device
parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (e.g., 'cuda' or 'cpu').")




if __name__ == "__main__":
    
    args = parser.parse_args()
    dataset_name = args.dataset_name
    scorer_name = args.scorer_name
    text_retriever_name = args.text_retriever_name
    skb = load_skb(dataset_name)
    qa = load_qa(dataset_name, human_generated_eval=False)
    
    eval_metrics = [
        "mrr",
        "map",
        "rprecision",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@50",
        "recall@100",
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "hit@20",
        "hit@50",
    ]
    
    mor_path = MOR4Path(dataset_name, text_retriever_name, scorer_name, skb)
    reasoner = Planner(dataset_name)
    outputs = []
    topk = 100
    split_idx = qa.get_idx_split(test_ratio=1.0)
    mod = args.mod
    all_indices = split_idx[mod].tolist()
    eval_csv = pd.DataFrame(columns=["idx", "query_id", "pred_rank"] + eval_metrics)
    
    count = 0
    
    # ***** planning *****
    # if the plan cache exists, load it
    plan_cache_path = f"./cache/{dataset_name}/path/{mod}_20250222.pkl"
    if os.path.exists(plan_cache_path):
        with open(plan_cache_path, 'rb') as f:
            plan_output_list = pkl.load(f)
    else:
        plan_output_list = []
        for idx, i in enumerate(tqdm(all_indices)):
            plan_output = {}
            query, q_id, ans_ids, _ = qa[i]
            rg = reasoner(query)  
    
            plan_output['query'] = query
            plan_output['q_id'] = q_id
            plan_output['ans_ids'] = ans_ids
            plan_output['rg'] = rg
            plan_output_list.append(plan_output)
        # save plan_output_list
        plan_cache_path = f"./cache/{dataset_name}/path/{mod}_20250222.pkl"
        os.makedirs(os.path.dirname(plan_cache_path), exist_ok=True)
        with open(plan_cache_path, 'wb') as f:
            pkl.dump(plan_output_list, f)
    
    
    # ***** Reasoning *****
    for idx, i in enumerate(tqdm(all_indices)):
        
        query = plan_output_list[idx]['query']
        q_id = plan_output_list[idx]['q_id']
        ans_ids = plan_output_list[idx]['ans_ids']
        rg = plan_output_list[idx]['rg']
        
        
        output = mor_path(query, q_id, ans_ids, rg, args)
        
        ans_ids = torch.LongTensor(ans_ids)
        
        pred_dict = output['pred_dict']
        result = mor_path.evaluate(pred_dict, ans_ids, metrics=eval_metrics)
        
        result["idx"], result["query_id"] = i, q_id
        result["pred_rank"] = torch.LongTensor(list(pred_dict.keys()))[
            torch.argsort(torch.tensor(list(pred_dict.values())), descending=True)[
                :topk
            ]
        ].tolist()

        eval_csv = pd.concat([eval_csv, pd.DataFrame([result])], ignore_index=True)
        
        output['q_id'] = q_id
        outputs.append(output)
        
        count += 1 
        
                
        # for metric in eval_metrics:
        #     print(
        #         f"{metric}: {np.mean(eval_csv[eval_csv['idx'].isin(all_indices)][metric])}"
        #     )
    
    
    print(f"MOR count: {mor_path.mor_count}")
    

    # prepare trajectories and save
    bm25 = mor_path.text_retriever
    test_data = prepare_trajectories(dataset_name, bm25, skb, outputs)
    save_path = f"{dataset_name}_{mod}.pkl"
    with open(save_path, 'wb') as f:
        pkl.dump(test_data, f)
    
    