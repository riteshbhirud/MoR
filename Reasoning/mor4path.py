import sys
from pathlib import Path
# Get the absolute path of the current script
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
# Add the project root to the system path
sys.path.append(str(project_root))

from .utils import combine_dicts, parse_metapath, get_scorer, get_text_retriever, fix_length
from models.model import ModelForSTaRKQA


class MOR4Path(ModelForSTaRKQA):
    def __init__(self, dataset_name, text_retriever_name, scorer_name, skb, topk=100):
        super(MOR4Path, self).__init__(skb)
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
        self.mor_k = 10 # topk for textual retrieval in MOR
        self.mor_count = 0
        self.num_negs = 200
        
        
        
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
            return None

        
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
                if len(route) < 3:
                    continue
                triplets = [
                    (route[i], route[i + 1], route[i + 2])
                    for i in range(0, len(route) - 2, 2)
                ]
                
                if all(tp in self.tp_list for tp in triplets) and all(len(tp) == 3 for tp in triplets): # and all length of triplets is 3
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
                    
                    relation_valid_routes.append(new_route)             

        if not relation_valid_routes:
            
            return None

        return relation_valid_routes
    
    def get_candidates4route(self, query, q_id, route, restriction):
        # initialization
        
        ini_node_type = route[0]
        
        try:
            type_restr = "".join(restriction[ini_node_type])
        except:
            type_restr = ""
            
        ini_dict = self.text_retriever.retrieve(query + " " + type_restr, q_id=q_id, topk=self.ini_k, node_type=ini_node_type)
        current_node_ids = list(ini_dict.keys())
        
        
        # initialize the bm_vector_dict
        bm_vector_dict = {key: [value] for key, value in ini_dict.items()}
        
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
            
            new_vector_dict = {}
            # for node
            for node_id in current_node_ids:
                neighbor_ids = self.skb.get_neighbor_nodes(idx=node_id, edge_type=edge_type)
                next_node_ids.extend(neighbor_ids)
                
                # ***** update paths and score_vector_dict *****
                for neighbor_id in neighbor_ids:
                    if neighbor_id not in new_paths.keys(): # only add new node
                        new_paths[neighbor_id] = paths[node_id] + [neighbor_id]
                        new_vector_dict[neighbor_id] = bm_vector_dict[node_id] + [-1] # -1 for padding
                    
    
            bm_vector_dict = new_vector_dict 
            
            
            
            # ***** layer text retrieval *****
            # if there is restriction for the next node, add text_retriever
            if next_node_type in restriction.keys() and len(restriction[next_node_type]) > 0 and restriction[next_node_type] != [""]:
                try:

                    retrieve_dict = self.text_retriever.retrieve(query+" "+"".join(restriction[next_node_type]), q_id=q_id, topk=self.mor_k, node_type=route[hop+2])

                    new_query = query+ " " + "".join(restriction[next_node_type])
                    
                    # take union
                    next_node_ids.extend(list(set(retrieve_dict.keys())))
                    
                    # ***** update paths and bm_vector_dict *****
                    for c_id in retrieve_dict.keys():
                        if c_id not in new_paths.keys():
                            new_paths[c_id] = [c_id]
                            bm_vector_dict[c_id] = [retrieve_dict[c_id]]
                            
                except:
                    pass
        
                
            paths = new_paths
            current_node_ids = list(set(next_node_ids))


        candidates = current_node_ids
        
        self.paths.append(paths)
        self.bm_vector_dict.append(bm_vector_dict)
        
        
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
        
        
        # remove empty lists from candidates
        non_empty_candidates_lists = [lst for lst in candidates_pool if lst]
        if len(non_empty_candidates_lists) == 0:

            return {}
        
        
        # Step 2: Combine candidates from different routes, try intersection first, then union
        candidates = self.merge_candidate_pools(non_empty_candidates_lists) # candidates is a list
        
        # step 3: score the candidates, ini to -1
        pred_dict = dict(zip(candidates, [-1]*len(candidates)))
        # print(f"111, {pred_dict}")
        
        return pred_dict
    
    def check_topk(self, query, q_id, pred_dict):
        
        missing = self.topk - len(set(pred_dict.keys()))
        if missing > 0:
            added_dict = self.text_retriever.retrieve(query, q_id, topk=self.topk+20, node_type=self.target_type_list) # +20 make it more safe 
            available_nodes = {key: value for key, value in added_dict.items() if key not in pred_dict.keys()}
            sorted_available_nodes = sorted(available_nodes.items(), key=lambda x: x[1], reverse=True)
            # Select only the required number of nodes to fill the missing slots
            selected_nodes = dict(sorted_available_nodes[:missing])
            
            # Update pred_dict with the selected nodes
            pred_dict.update(selected_nodes)
            
            # updata paths
            for node_id in selected_nodes.keys():
                self.paths[node_id] = [node_id]
                
            # update bm_vector_dict
            new_bm_vector_dict = {key: [value] for key, value in selected_nodes.items()}
            self.bm_vector_dict.update(new_bm_vector_dict)
            
        scored_dict = self.scorer.score(query, q_id=q_id, candidate_ids=list(pred_dict.keys()))
        
        if len(scored_dict) > self.topk:
            # initiliaze the new_paths
            new_paths = {}
            
            # Select the top-k nodes based on the scores
            sorted_scored_dict = sorted(scored_dict.items(), key=lambda x: x[1], reverse=True)
            scored_dict = dict(sorted_scored_dict[:self.topk])
            
            # update paths
            for node_id in scored_dict.keys():
                new_paths[node_id] = self.paths[node_id]
            
            self.paths = new_paths
            
            # update bm_vector_dict
            new_bm_vector_dict = {node_id: self.bm_vector_dict[node_id] for node_id in scored_dict.keys()}
            self.bm_vector_dict = new_bm_vector_dict
            
        
        return scored_dict
    
    
    # check fixed negtopk
    def check_negtopk(self, query, q_id, pred_dict, ans_ids):
        # check the positive nodes
        pos_ids = [node_id for node_id in ans_ids if node_id in pred_dict.keys()]
        pos_dict = {key: value for key, value in pred_dict.items() if key in pos_ids}
        neg_ids = pred_dict.keys() - set(pos_ids)
        neg_dict = {key: value for key, value in pred_dict.items() if key in neg_ids}
        
        # check the number of negative nodes
        missing = self.num_negs - len(neg_ids)
        
        if missing > 0:
            
            added_dict = self.text_retriever.retrieve(query, q_id, topk=self.num_negs+200, node_type=self.target_type_list) # +20 make it more safe 
            available_nodes = {key: value for key, value in added_dict.items() if key not in pred_dict.keys() and key not in ans_ids}
            sorted_available_nodes = sorted(available_nodes.items(), key=lambda x: x[1], reverse=True)
            # Select only the required number of nodes to fill the missing slots
            selected_nodes = dict(sorted_available_nodes[:missing])
            
            # Update pred_dict with the selected nodes
            neg_dict.update(selected_nodes)
            
            # updata paths
            for node_id in selected_nodes.keys():
                self.paths[node_id] = [node_id]
                
            # update bm_vector_dict
            new_bm_vector_dict = {key: [value] for key, value in selected_nodes.items()}
            self.bm_vector_dict.update(new_bm_vector_dict)
            
            
        scored_neg_dict = self.scorer.score(query, q_id=q_id, candidate_ids=list(neg_dict.keys()))
        if pos_dict:
            scored_pos_dict = self.scorer.score(query, q_id=q_id, candidate_ids=list(pos_dict.keys()))
        else:
            scored_pos_dict = {}
        
        if len(scored_neg_dict) > self.num_negs:
            # Select the top-k nodes based on the scores
            sorted_scored_neg_dict = sorted(scored_neg_dict.items(), key=lambda x: x[1], reverse=True)
            scored_neg_dict = dict(sorted_scored_neg_dict[:self.num_negs])
            
            
            
        scored_neg_dict.update(scored_pos_dict)
        scored_dict = scored_neg_dict
        print(len(scored_dict))
        
        # update paths
        new_paths = {}
        for node_id in scored_dict.keys():
            new_paths[node_id] = self.paths[node_id]
        self.paths = new_paths
        
        # update bm_vector_dict
        new_bm_vector_dict = {node_id: self.bm_vector_dict[node_id] for node_id in scored_dict.keys()}
        self.bm_vector_dict = new_bm_vector_dict
        
        
        return scored_dict
        
    
    def forward(self, query, q_id, ans_ids, rg, args):
        
        self.paths = []
        self.bm_vector_dict = []
        self.ada_score = {}
        # ***** Structural Retrieval *****
        
        # reasoning grpah to routes
        if self.dataset_name == "prime":
            routes = rg["Metapath"]
        else:
            routes = self.rg2routes(rg)
        
        # check valid
        valid_routes = self.check_valid(routes, rg) # add check for restriction
        
        if valid_routes is None:
            # do textual retrieval
            pred_dict = self.text_retriever.retrieve(query, q_id, topk=self.topk, node_type=self.target_type_list)  
            
            # update bm_vector_dict
            self.bm_vector_dict = {key: [value] for key, value in pred_dict.items()}
            
        else:
            # truncate the valid_routes
            if self.dataset_name == "prime":
                pass
            else:
                valid_routes = [route[-5:] for route in valid_routes]
                
            
            # do structural retrieval
            restriction = rg["Restriction"]
            pred_dict = self.get_mor_candidates(query, q_id, valid_routes, restriction)
            self.mor_count += 1
            
        
        # **** combine paths ****
        if self.paths:
            self.paths = combine_dicts(self.paths, pred_dict=pred_dict) # return dict
        else:
            self.paths = {}
            for node_id in pred_dict.keys():
                self.paths[node_id] = [node_id]
                
        # ***** combine bm_vector_dict *****
        if isinstance(self.bm_vector_dict, list):
            self.bm_vector_dict = combine_dicts(self.bm_vector_dict, pred_dict=pred_dict)
        
        # **** fix neg for training; fix candidates for testing ****
        if args.mod == "train":
            # check neg topk 
            pred_dict = self.check_negtopk(query, q_id, pred_dict, ans_ids)
        else:
            # check topk    
            pred_dict = self.check_topk(query, q_id, pred_dict)

        
        # **** length padding and truncate *****
        if self.dataset_name != "prime":
            self.paths = fix_length(self.paths)
                
        if len(self.paths) != len(pred_dict):
            print(f"paths: {self.paths}")
            print(f"pred_dict: {pred_dict}")
            raise ValueError(f"Length mismatch between paths and pred_dict: {len(self.paths)}, {len(pred_dict)}")

        output = {
            "query": query,
            "pred_dict": pred_dict,
            "ans_ids": ans_ids,
            'paths': self.paths,
            'bm_vector_dict': self.bm_vector_dict,
            'rg': rg
        }
        
        
        return output
    
if __name__ == "__main__":
    print(f"Test mor4path")
        

