from Reasoning.text_retrievers.bm25 import BM25
from Reasoning.text_retrievers.ada import Ada
from Reasoning.text_retrievers.contriever import Contriever


def combine_dicts(dicts_list, pred_dict):
    if len(dicts_list) == 1:
        return dicts_list[0]
    combined_dict = {}
    
    for d in dicts_list:
        for key, value in d.items():
            if key in combined_dict:
                # for route dict, the values are lists, keep the longest list
                if len(value) > len(combined_dict[key]):
                    combined_dict[key] = value
            else:
                combined_dict[key] = value

    # if the two reasoning paths have intersection, only keep the keys in pred_dict 
    combined_dict = {key: combined_dict[key] for key in pred_dict.keys()}
    
    
    return combined_dict
    
def fix_length(paths_dict):
    max_length = 3
    new_paths_dict = {}
    
    for key, value in paths_dict.items():
        if len(value) > max_length:
            value = value[-max_length:]
        if len(value) < max_length:
            # padding with -1 at the beginning
            value = [-1] * (max_length - len(value)) + value
        new_paths_dict[key] = value
        
    return new_paths_dict
            
    

def parse_metapath(metapath):
    """
        input: metapath: "paper -> author -> paper <- paper"
        output: routes: [['paper', 'author', 'paper'], ['paper', 'paper']]
    """
    
    def parse(remain_list, direction):
        """
            input: remain_list: ["paper", "->", "author", "->", "paper", "<-", "paper"]
                direction: "->"
            output: route: ["paper", "author", "paper"]
                    remain_list: ["paper", "<-", "paper"]
        """
        route = []
        i = 0
        while i < len(remain_list)-1 and remain_list[i+1] == direction:
            route.append(remain_list[i])
            i += 2
        route.append(remain_list[i])
        
        if direction == "<-":
            route.reverse()    
            
        remain_list = None if len(remain_list) == i+1 else remain_list[i:]
            
        return route, remain_list
    
    
    remain_list = metapath.split(' ')
    # print(f"111, {remain_list}")
    
    if len(remain_list) == 1: # single node
        return [remain_list]
    
    routes = []
    while remain_list is not None:
        if remain_list[1] == "<-":
            route, remain_list = parse(remain_list, "<-")
        
        elif remain_list[1] == "->":
            route, remain_list = parse(remain_list, "->")
            
        else: 
            # raise ValueError(f"Invalid metapath: {metapath}")
            return None
        
        routes.append(route)
            
    return routes 


def get_text_retriever(dataset_name, retriever_name, skb, **kwargs):
    if retriever_name == "bm25":
        return BM25(skb, dataset_name)
    elif retriever_name == "ada":
        return Ada(skb, dataset_name, kwargs.get("device", 'cuda'))
    elif retriever_name == "contriever":
        return Contriever(skb, dataset_name, kwargs.get("device", 'cuda'))
    else:
        raise ValueError(f"Invalid retriever name: {retriever_name}")


def get_scorer(dataset_name, scorer_name, skb, **kwargs):
    if scorer_name == "bm25":
        return BM25(skb, dataset_name)
    elif scorer_name == "ada":
        return Ada(skb, dataset_name, kwargs.get("device",'cuda'))
    elif scorer_name == "contriever":
        return Contriever(skb, dataset_name, kwargs.get("device", 'cuda'))
    else:
        raise ValueError(f"Invalid scorer name: {scorer_name}")
    

if __name__ == "__main__":
    print(f"Test utils")