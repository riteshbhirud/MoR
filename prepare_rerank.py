from Reasoning.text_retrievers.contriever import Contriever
from Reasoning.text_retrievers.ada import Ada
from stark_qa import load_qa, load_skb

import pickle as pkl
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

model_name = f"bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name)
encoder = BertModel.from_pretrained(model_name)


def get_bm25_scores(dataset_name, bm25, outputs):
    
    new_outputs = []
    # use tqdm to visualize the progress
    for i in range(len(outputs)):
        query, q_id, ans_ids = outputs[i]['query'], outputs[i]['q_id'], outputs[i]['ans_ids']
        paths= outputs[i]['paths']
        rg = outputs[i]['rg']
        
        if dataset_name == 'prime':
            new_path_dict = paths
        else:
            # make new path dict and remove the -1 from the path
            new_path_dict = {}
            for key in paths.keys():
                new_path = [x for x in paths[key] if x != -1]
                new_path_dict[key] = new_path
            
        # collect all values of the path without the first element
        candidates_ids = []
        for key in new_path_dict.keys():
            candidates_ids.extend(new_path_dict[key][1:])
            candidates_ids.extend(ans_ids)
        candidates_ids = list(set(candidates_ids))
        
        # get the bm25 score
        bm_score_dict = bm25.score(query, q_id, candidate_ids=candidates_ids)
        outputs[i]['bm_score_dict'] = bm_score_dict
        
        # replace -1 in the bm_vector_dict with the bm_score
        bm_vector_dict = outputs[i]['bm_vector_dict']
        for key in bm_vector_dict.keys():
            if -1 in bm_vector_dict[key]:
                path = new_path_dict[key]
                assert len(path) == len(bm_vector_dict[key])
    
                bm_vector_dict[key] = [bm_score_dict[path[j]] if x == -1 else x for j, x in enumerate(bm_vector_dict[key])]
                
        
        outputs[i]['bm_vector_dict'] = bm_vector_dict
        
        # fix length of paths in prime
        if dataset_name == 'prime':
            max_len = 3
            new_paths = {}
            for key in paths:
                new_path = paths[key]
                if len(paths[key]) < max_len:
                    new_path = [-1] * (max_len - len(paths[key])) + paths[key]
                elif len(paths[key]) > max_len:
                    new_path = paths[key][-max_len:]
                new_paths[key] = new_path
            
            # assign the new path to the paths
            outputs[i]['paths'] = new_paths
        
        new_outputs.append(outputs[i])
    
    return new_outputs


def prepare_score_vector_dict(raw_data):
    # make the score_vector_dict: [bm_score, bm_score, bm_score, ada_score/contriver_score]
    for i in range(len(raw_data)):
        # get the pred_dict
        pred_dict = raw_data[i]['pred_dict']
        # get the bm_vector_dict
        bm_vector_dict = raw_data[i]['bm_vector_dict']
        # initialize the score_vector_dict
        raw_data[i]['score_vector_dict'] = {}
        # add the value of pred_dict to the end of the bm_vector_dict
        for key in pred_dict:
            # get the bm_score, last element of the bm_vector_dict
            bm_vector = bm_vector_dict[key]
            # get the ranking score
            rk_score = pred_dict[key]
            # make the score_vector_dict
            score_vector = bm_vector + [rk_score]
            # check the length of the score_vector, if less than 4, pad with 0 at the beginning
            if len(score_vector) < 4:
                score_vector = [0] * (4 - len(score_vector)) + score_vector
            elif len(score_vector) > 4:
                score_vector = score_vector[-4:]
            # make the score_vector_dict
            raw_data[i]['score_vector_dict'][key] = score_vector
            
    return raw_data


def prepare_text_emb_symb_enc(raw_data, skb):
    # add the text_emb to the raw_data
    text2emb_list = []
    text2emb_dict = {}
    
    symbolic_encode_dict = {
    3: [0, 1, 1],
    2: [2, 0, 1],
    1: [2, 2, 0],
    }
    
    for i in range(len(raw_data)):
        # get the paths
        paths = raw_data[i]['paths']
        preds = raw_data[i]['pred_dict']
        assert len(paths) == len(preds)
            
        # initialize the text_emb_dict
        raw_data[i]['text_emb_dict'] = {}
        
        # initialize the symb_enc_dict
        raw_data[i]['symb_enc_dict'] = {}
        
        for key in paths:
            # get the path
            path = paths[key]
            # make uniquee text_emb_path and make dict
            text_path_li = [skb.get_node_type_by_id(node_id) if node_id != -1 else "padding" for node_id in path]
            text_path_str = " ".join(text_path_li)
            if text_path_str not in text2emb_list:
                
                text2emb_list.append(text_path_str)
                text2emb_dict[text_path_str] = -1
            
            # assgin thte text_path to the raw_data
            raw_data[i]['text_emb_dict'][key] = text_path_str
            
            # ***** make the symb_enc_dict *****
            # number of non -1 in the path
            num_non_1 = len([p for p in path if p != -1])
            # get the symbolic encoding
            symb_enc = symbolic_encode_dict[num_non_1]
            # make the symb_enc_dict
            raw_data[i]['symb_enc_dict'][key] = symb_enc
            
    # ***** get the text2emb_dict embeddings *****
    for key in text2emb_dict.keys():
        # get the tokens for the node type using th tokenizer
        text_enc = tokenizer(key, return_tensors='pt')['input_ids']
        outputs = encoder(text_enc)
        last_hidden_states = outputs.last_hidden_state.mean(dim=1)
        text2emb_dict[key] = last_hidden_states.detach()
    
            
    new_data = {'data': raw_data, 'text2emb_dict': text2emb_dict}
    
    return new_data


def prepare_trajectories(dataset_name, bm25, skb, outputs):
    # get the bm25 scores
    new_outputs = get_bm25_scores(dataset_name, bm25, outputs) # return list
    # prepare the score_vector_dict
    new_outputs = prepare_score_vector_dict(new_outputs) # return list
    # prepare the text_emb and symb_enc_dict
    new_data = prepare_text_emb_symb_enc(new_outputs, skb) # return dict
    
    return new_data

        
def get_contriever_scores(dataset_name, mod, skb, path):
    
    with open(path, 'rb') as f:
        data = pkl.load(f)

    raw_data = data['data']
    
        
    qa = load_qa(dataset_name, human_generated_eval=False)

    contriever = Contriever(skb, dataset_name, device='cuda')

    split_idx = qa.get_idx_split(test_ratio=1.0)

    all_indices = split_idx[mod].tolist()
    # use tqdm to visualize the progress
    for idx, i in enumerate(tqdm(all_indices)):
        query, q_id, ans_ids, _ = qa[i]
        assert query == raw_data[idx]['query']
        pred_ids = list(raw_data[idx]['pred_dict'].keys())
        candidates_ids = list(set(pred_ids))
        candidates_ids.extend(ans_ids)
        
        # get contriever score
        contriever_score_dict = contriever.score(query, q_id, candidate_ids=candidates_ids)

        raw_data[idx]['contriever_score_dict'] = contriever_score_dict
    
    
    data['data'] = raw_data
            
    with open(path, 'wb') as f:
        pkl.dump(data, f)
        
def get_ada_scores(dataset_name, mod, skb, path):
    
    with open(path, 'rb') as f:
        data = pkl.load(f)

    raw_data = data['data']
    
        
    qa = load_qa(dataset_name, human_generated_eval=False)

    ada = Ada(skb, dataset_name, device='cuda')

    split_idx = qa.get_idx_split(test_ratio=1.0)

    all_indices = split_idx[mod].tolist()
    # use tqdm to visualize the progress
    for idx, i in enumerate(tqdm(all_indices)):
        query, q_id, ans_ids, _ = qa[i]
        assert query == raw_data[idx]['query']
        pred_ids = list(raw_data[idx]['pred_dict'].keys())
        candidates_ids = list(set(pred_ids))
        candidates_ids.extend(ans_ids)
        
        # get ada score
        ada_score_dict = ada.score(query, q_id, candidate_ids=candidates_ids)

        raw_data[idx]['ada_score_dict'] = ada_score_dict
    
    
    data['data'] = raw_data
            
    with open(path, 'wb') as f:
        pkl.dump(data, f)

if __name__ == '__main__':
    print(f"Test prepare_rerank")
        
                