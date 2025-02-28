"""
input:   rg
output (fixed 100 candidates, for path-based reranking): 
{
    "query": query,
    "pred_dict": {node_id: score},
    "ans_ids": [],
    'paths': {node_id: [node_ids_path]}
}

"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

from utils import combine_dicts, parse_metapath, get_scorer, get_text_retriever, fix_length
from models.model import ModelForSTaRKQA
import time



class Stru4Path(ModelForSTaRKQA):
    def __init__(self, dataset_name, text_retriever_name, scorer_name, skb, topk=100):
        super(Stru4Path, self).__init__(skb)
        self.dataset_name = dataset_name
        self.text_retriever = get_text_retriever(dataset_name, text_retriever_name, skb)
        self.scorer = get_scorer(dataset_name, scorer_name=scorer_name, skb=skb)
        # self.scorer = self.text_retriever
        self.topk = topk
        self.node_type_list = skb.node_type_lst()
        self.edge_type_list = skb.rel_type_lst()
        if self.dataset_name == "prime":
            self.tp_list = skb.get_tuples()
            self.target_type_list = skb.candidate_types
        else:
            self.tp_dict = {(tp[0], tp[-1]): tp[1] for tp in skb.get_tuples()}
            self.target_type_list = ['paper' if dataset_name == 'mag' else 'product']
            
        self.skb = skb
        self.ini_k = 5 # topk for initial retrieval
        self.stru_count = 0
    
        
        
        
    def rg2routes(self, rg):
        """
            input: rg: {"Metapath": "", "Restriction": {}}
            output: routes: [['paper', 'author', 'paper'], ['paper', 'paper']]
        """
        # parse rg
        metapath = rg["Metapath"]
        if isinstance(rg["Metapath"], list):
            routes = rg["Metapath"]
        elif isinstance(rg["Metapath"], str):
            routes = parse_metapath(metapath)
        else:
            return None
        
        return routes
    
    def check_valid(self, routes, rg):
        # check the length of routes
        if not routes:
            # raise ValueError(f"Empty routes: {routes}")
            return None
        
        if len(routes) == 1 and len(routes[0]) == 1: # single node, directly do text retrieval
            return 1
        
        # Step 1: Filter routes by target type
        target_type_valid_routes = [
            route for route in routes if route[-1] in self.target_type_list
        ]
        if not target_type_valid_routes:
            return None

        # Step 2: Filter routes by node and edge type
        type_valid_routes = [
            route
            for route in target_type_valid_routes
            if all(
                node in self.node_type_list or node in self.edge_type_list
                for node in route
            )
        ]
        if not type_valid_routes:
            return None

        # Step 3: Check existence of relations
        relation_valid_routes = []
        for route in type_valid_routes:
            if self.dataset_name == "prime":
                triplets = [
                    (route[i], route[i + 1], route[i + 2])
                    for i in range(0, len(route) - 2, 2)
                ]
                
                if all(tp in self.tp_list for tp in triplets):
                    relation_valid_routes.append(route)
            else:
                pairs = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
                if all(tp in self.tp_dict.keys() for tp in pairs):
                    relations = [self.tp_dict[tp] for tp in pairs]
                    
                    # make route with relations
                    new_route = []
                    for i in range(len(relations)):
                        new_route.append(pairs[i][0])
                        new_route.append(relations[i])
                    new_route.append(pairs[-1][-1])
                    # print(f"222, {new_route}")
                    
                    relation_valid_routes.append(new_route)             

        if not relation_valid_routes:
            return None

        return relation_valid_routes
    
    def get_candidates4route(self, query, q_id, route, restriction):
        # initialization
        
        ini_node_type = route[0]
        
        try:
            extra_restr = "".join(restriction[ini_node_type])
        except:
            extra_restr = ""
        ini_dict = self.text_retriever.retrieve(query + " " + extra_restr, q_id=q_id, topk=self.ini_k, node_type=ini_node_type)
        current_node_ids = list(ini_dict.keys())
        
        # initilization for paths
        paths = {}
        for c_id in current_node_ids:
            paths[c_id] = [c_id]
    
        # loop
        hops = len(route)
        # for hop/layer
        for hop in range(0, hops-2, 2):
            new_paths = {}
            
            cur_node_type = route[hop]
            next_node_type = route[hop+2]
            edge_type = route[hop+1]
            next_node_ids = []
            
            # for node
            for node_id in current_node_ids:
                neighbor_ids = self.skb.get_neighbor_nodes(idx=node_id, edge_type=edge_type)
                next_node_ids.extend(neighbor_ids)
                
                # **x*** update paths *****
                for neighbor_id in neighbor_ids:
                    new_paths[neighbor_id] = paths[node_id] + [neighbor_id]
        
                        
            paths = new_paths
            
            current_node_ids = list(set(next_node_ids))

        candidates = current_node_ids
        self.paths.append(paths)
        
        
        return candidates
    
    def merge_candidate_pools(self, non_empty_candidates_lists):
         
         
         # if only one non-empy candidates list left, return it as a set
         if len(non_empty_candidates_lists) == 1:
              return set(non_empty_candidates_lists[0])   
         # find the intersection candidates ids
         result = set(non_empty_candidates_lists[0])
         for lst in non_empty_candidates_lists[1:]:
              result.intersection_update(lst)
        
         # if the intersection is empty, return the union of all candidates
         if len(result) == 0:
              result = set()
              for lst in non_empty_candidates_lists:
                   result.update(lst)        
         
         
         
         return list(result)
    
    def get_mor_candidates(self, query, q_id, valid_routes, restriction):
        
        # Step 1: Get candidates for each route
        candidates_pool = []
        for route in valid_routes:
            if route[0] in restriction.keys() and len(restriction[route[0]]) > 0:
                candidates_pool.append(self.get_candidates4route(query, q_id, route, restriction)) # topk is the candidates retrieved from textual retriever    
        
        non_empty_candidates_lists = [lst for lst in candidates_pool if lst]
        if not non_empty_candidates_lists: # no candidates, return empty dict
            print(f"123, {non_empty_candidates_lists}")
            
            # raise ValueError("No candidates for any route")
            return {}
        
        
        # Step 2: Combine candidates from different routes, try intersection first, then union
        candidates = self.merge_candidate_pools(candidates_pool) # candidates is a list
        if not candidates:
            return {}
        
        
        # step 3: score the candidates, ini to -1
        pred_dict = dict(zip(candidates, [-1]*len(candidates)))
        # print(f"111, {pred_dict}")
        
        return pred_dict
    
        
    
    def forward(self, query, q_id, ans_ids, rg):
        
        self.paths = []
        # ***** Structural Retrieval *****
        
        # reasoning grpah to routes
        s_time = time.time()
        routes = self.rg2routes(rg)
        # print(f"444, {time.time()-s_time}")
        
        # check valid
        s_time = time.time()
        valid_routes = self.check_valid(routes, rg) # add check for restriction
        # print(f"555, {time.time()-s_time}")
        
        if valid_routes is None:
            # return empty dict
            return {
                "query": query,
                "pred_dict": {},
                "ans_ids": ans_ids,
                'paths': {},
                'query_pattern': rg['Metapath']
            }
        elif valid_routes == 1: # TODO: empty string
            print(f"1234: {valid_routes}")
            # do text retrieval
            pred_dict = self.text_retriever.retrieve(query, q_id=q_id, topk=self.topk, node_type=f'{self.target_type_list[0]}')
            
        else:
            # do structural retrieval
            # truncate the valid_routes
            if self.dataset_name == "prime":
                pass
            else:
                valid_routes = [route[-5:] for route in valid_routes]
            
            restriction = rg["Restriction"]
            pred_dict = self.get_mor_candidates(query, q_id, valid_routes, restriction)
            self.stru_count += 1
        
        # **** combine paths ****
        if self.paths:
            self.paths = combine_dicts(self.paths, pred_dict=pred_dict) # return dict
            
        else:
            self.paths = {}
            for node_id in pred_dict.keys():
                self.paths[node_id] = [node_id]
        
        # if retrieved candidates is empty, return empty dict
        if not pred_dict:
            return {
                "query": query,
                "pred_dict": {},
                "ans_ids": ans_ids,
                'paths': {},
                'query_pattern': rg['Metapath']
            }
        
        # score the candidates
        pred_dict = self.scorer.score(query, q_id, list(pred_dict.keys()))
        
        # # **** length padding and truncate *****
        # self.paths = fix_length(self.paths)
                
        if len(self.paths) != len(pred_dict):
            print(f"paths: {self.paths}")
            print(f"pred_dict: {pred_dict}")
            raise ValueError(f"Length mismatch between paths and pred_dict: {len(self.paths)}, {len(pred_dict)}")

        output = {
            "query": query,
            "pred_dict": pred_dict,
            "ans_ids": ans_ids,
            'paths': self.paths,
            'query_pattern': rg['Metapath'],
            'rg': rg
        }
        
        
        return output
        

        

