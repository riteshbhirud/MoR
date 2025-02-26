import torch.nn as nn
import torch

# ***** Reranking Model *****
# two linear layers
class PathReranker(nn.Module):
    def __init__(self, socre_vector_input_dim=4, text_emb_input_dim=768, symb_enc_dim=3, output_dim=1, emb_dim=256, args=None):
        super(PathReranker, self).__init__()
        self.score_vec_enc = nn.Linear(socre_vector_input_dim, emb_dim)
        self.text_emb_enc = nn.Linear(text_emb_input_dim, emb_dim)
        self.symb_enc = nn.Embedding(symb_enc_dim, emb_dim)
        self.fc1 = nn.Linear(emb_dim*args.concat_num, output_dim)
        self.fc2 = nn.Linear(output_dim, 1)
        self.relu = nn.ReLU()
        self.args = args
        
        
    
    def eval_batch(self, batch):
        
        embeddings = []
        
        if self.args.text_emb:
            # Encode the text embedding and apply ReLU
            text_emb_c = self.relu(self.text_emb_enc(batch['c_text_emb'])) # [bs, 100, emb_dim]
            embeddings.append(text_emb_c)
            
        
        if self.args.score_vec:
            # Encode the score vector and apply ReLU
           score_vector_c = self.relu(self.score_vec_enc(batch['c_score_vector'])) # [bs, 100, emb_dim]
        
           embeddings.append(score_vector_c)
        
        
        if self.args.symb_enc:
            # encode the symbolic embedding and apply ReLU
            symb_enc_c = self.relu(self.symb_enc(batch['c_symb_enc']))
            # reshape the symbolic embedding
            symb_enc_c = torch.reshape(symb_enc_c, (symb_enc_c.shape[0], symb_enc_c.shape[1], -1))
            embeddings.append(symb_enc_c)
            
            
        if len(embeddings) > 1:
            emb_c = torch.cat(embeddings, dim=-1)
        else:
            emb_c = embeddings[0]
            
        
        # Feed the concatenated embeddings to the final layer
        emb_c = self.fc1(emb_c) # [bs, 100, emb_dim]
        scores_c = self.fc2(emb_c) # [bs, 100, 1]
        
        return scores_c

    
    def forward(self, batch):    
        
        embeddings_pos = []
        embeddings_neg = []
        
        if self.args.text_emb:
            # Encode the text embedding and apply ReLU
            text_emb_pos = self.relu(self.text_emb_enc(batch['p_text_emb']))
            text_emb_neg = self.relu(self.text_emb_enc(batch['n_text_emb']))
            embeddings_pos.append(text_emb_pos)
            embeddings_neg.append(text_emb_neg)
        
        if self.args.score_vec:
            # Encode the score vector and apply ReLU
            score_vector_pos = self.relu(self.score_vec_enc(batch['p_score_vector']))
            score_vector_neg = self.relu(self.score_vec_enc(batch['n_score_vector']))
            embeddings_pos.append(score_vector_pos)
            embeddings_neg.append(score_vector_neg)
        
        
        if self.args.symb_enc:
            # encode the symbolic embedding and apply ReLU
            symb_enc_pos = self.relu(self.symb_enc(batch['p_symb_enc']))
            # reshape the symbolic embedding
            symb_enc_pos = torch.reshape(symb_enc_pos, (symb_enc_pos.shape[0], -1))
            
            symb_enc_neg = self.relu(self.symb_enc(batch['n_symb_enc']))
            # reshape the symbolic embedding
            symb_enc_neg = torch.reshape(symb_enc_neg, (symb_enc_neg.shape[0], symb_enc_neg.shape[1], -1)) # [bs, neg_sp, path_len * emb_dim]
            
            embeddings_pos.append(symb_enc_pos)
            embeddings_neg.append(symb_enc_neg)
        
        
        if len(embeddings_pos) > 1:
            pos = torch.cat(embeddings_pos, dim=-1)
            neg = torch.cat(embeddings_neg, dim=-1)
        else:
            pos = embeddings_pos[0]
            neg = embeddings_neg[0]
                
        
        
        # Feed the concatenated embeddings to the final layer
        pos = self.fc1(pos)
        neg = self.fc1(neg)
        scores_pos = self.fc2(pos)
        scores_neg = self.fc2(neg)
        
        return scores_pos, scores_neg
    