"""
input: query, node_type, topk
output: pred_dict: {node_id: score}

"""
import sys
sys.path.append('/home/yongjia/dgl/Yongjia/MOE_20250222/')

import bm25s
from tqdm import tqdm
from Reasoning.text_retrievers.stark_model import ModelForSTaRKQA

target_type = {'amazon': 'product', 'prime': 'combine', 'mag': 'paper'}

class BM25(ModelForSTaRKQA):
    
    def __init__(self, skb, dataset_name):
        super(BM25, self).__init__(skb)
        self.retrievers = {}
        self.text_to_ids = {}
        type_names = skb.node_type_lst()
        self.nodeid_to_index = {}
        
        self.target_type = target_type[dataset_name]

        if self.target_type not in type_names:
             ids = skb.get_candidate_ids()
             
             
             corpus = [skb.get_doc_info(id) for id in tqdm(ids, desc=f"Gathering docs for combine")]     
             retriever = bm25s.BM25(corpus=corpus)
             retriever.index(bm25s.tokenize(corpus))
             # Build hash map from text to node_id
             text_to_id = {hash(text): id for text, id in zip(corpus, ids)}
             # Store the retriever and text_to_id by type_name
             self.retrievers[self.target_type] = retriever
             self.text_to_ids[self.target_type] = text_to_id
             
             self.nodeid_to_index[self.target_type] = {id: i for i, id in enumerate(ids)}

        # Initialize retrievers and text-to-index maps for each type_name
        for type_name in type_names:
            ids = skb.get_node_ids_by_type(type_name)
            
            # we manually replace '&' with '_and_' to avoid the error in BM25, because BM25 uses '&' as a special character and will not tokenize it
            corpus = [skb.get_doc_info(id).replace('&', '_and_').replace('P.O.R', 'P_dot_O_dot_R') for id in tqdm(ids, desc=f"Gathering docs for {type_name}")]
            # Create the BM25 model for the current type_name
            retriever = bm25s.BM25(corpus=corpus)
            retriever.index(bm25s.tokenize(corpus))
            
            # Build hash map from text to index
            text_to_id = {hash(text): id for text, id in zip(corpus, ids)}

            # Store the retriever and text_to_id by type_name
            self.retrievers[type_name] = retriever
            self.text_to_ids[type_name] = text_to_id
            
            # build map from node_id to index
            self.nodeid_to_index[type_name] = {id: i for i, id in enumerate(ids)}
    
    def score(self, query, q_id, candidate_ids):
        pred_dict = {}
        
        for c_id in candidate_ids:
            type_name = self.skb.get_node_type_by_id(c_id)
            score = self.retrievers[type_name].get_scores(list(bm25s.tokenize(query)[1].keys()))[self.nodeid_to_index[type_name][c_id]] # save the query tokens 
            pred_dict[c_id] = score
            
        # print(f"999, {pred_dict}")
        
        return pred_dict
        
    def retrieve(self, query, q_id, topk, node_type=None):
                
        """
        Forward pass to compute similarity scores for the given query.

        Args:
            query (str): Query string.

        Returns:
            pred_dict (dict): A dictionary of candidate ids and their corresponding similarity scores.
        """
        if '&' in query:
               query = query.replace('&', '_and_')
        if 'P.O.R' in query:
               query = query.replace('P.O.R', 'P_dot_O_dot_R')
        if isinstance(node_type, list):
            if len(node_type) > 1:
                node_type = 'combine'
            else:
                node_type = node_type[0]
        results, scores = self.retrievers[node_type].retrieve(bm25s.tokenize(query), k=topk)
        ids = [self.text_to_ids[node_type][hash(result.item())] for result in results[0]]
        scores = scores[0].tolist()
        pred_dict = dict(zip(ids, scores))
        # print(f"666, {pred_dict}")
        
        return pred_dict

if __name__ == '__main__':
    from stark_qa import load_qa, load_skb
    dataset_name = 'amazon'
    skb = load_skb(dataset_name)
    retriever = BM25(skb, dataset_name)
    query = 'What is the price of the product?'
    pred_dict = retriever(query, 'combine', 5)
    print(pred_dict)