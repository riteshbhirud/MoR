import torch.nn as nn

    

# ***** Reranking Model *****
class Contriever(nn.Module):
    def __init__(self, encoder, emb_dim=768):
        super(Contriever, self).__init__()
        self.encoder = encoder
        self.emb_dim = emb_dim
    
    def mean_pooling(self, token_embs, mask):
        token_embs = token_embs.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embs.sum(dim=1) / (mask.sum(dim=1)[..., None].clamp(min=1e-9))
        return sentence_embeddings
    
    def encode_seq(self, input_ids, attention_mask, token_type_ids=None):
        # Combine inputs into a dictionary
        enc = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            enc['token_type_ids'] = token_type_ids

        outputs = self.encoder(**enc)
        # Mean pooling of last hidden states
        embedded = self.mean_pooling(outputs[0], attention_mask)
        # print(f"777, {embedded.shape}")
        return embedded

    def get_text_emb(self, input_ids, attention_mask, token_type_ids):
        emb = self.encode_seq(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        return emb
    
    def eval_batch(self, batch):
        # q_emb: [batch_size/num_gpus, token_dim]
        q_emb = self.get_text_emb(batch['q_enc_input_ids'], batch['q_enc_attention_mask'], batch['q_enc_token_type_ids'])
        
        # c_emb: [batch_size * num_candidates/num_gpus, token_dim]
        c_emb = self.get_text_emb(batch['c_enc_input_ids'], batch['c_enc_attention_mask'], batch['c_enc_token_type_ids'])
        

        
        return q_emb, c_emb
    
    def forward(self, batch):
        
        # q_emb: [batch_size/num_gpus, token_dim]
        q_emb = self.get_text_emb(batch['q_enc_input_ids'], batch['q_enc_attention_mask'], batch['q_enc_token_type_ids'])
        # p_emb: [batch_size*max_len/num_gpus, token_dim]
        p_emb = self.get_text_emb(batch['pos_enc_input_ids'], batch['pos_enc_attention_mask'], batch['pos_enc_token_type_ids'])
        # n_emb: [batch_size*max_len*num_sampled_negs/num_gpus, token_dim]
        n_emb = self.get_text_emb(batch['neg_enc_input_ids'], batch['neg_enc_attention_mask'], batch['neg_enc_token_type_ids'])
        

        
        return q_emb, p_emb, n_emb


# ***** Reranking Model *****
class NodeRouter(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, emb_dim=128):
        super(NodeRouter, self).__init__()
        self.fc1 = nn.Linear(input_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, output_dim)
        self.relu = nn.ReLU()
    
    def eval_batch(self, batch):
        scores_cand = self.fc1(batch['c_scores'])   
        scores_cand = self.relu(scores_cand)
        scores_cand = self.fc2(scores_cand)
        scores_cand = self.relu(scores_cand)
        
        return scores_cand
    
    def forward(self, batch):
        scores_pos = self.fc1(batch['p_scores'])
        scores_neg = self.fc1(batch['n_scores'])
        scores_pos = self.relu(scores_pos)
        scores_neg = self.relu(scores_neg)
        
        scores_pos = self.fc2(scores_pos)
        scores_neg = self.fc2(scores_neg)
        scores_pos = self.relu(scores_pos)
        scores_neg = self.relu(scores_neg)
        
        return scores_pos, scores_neg
        