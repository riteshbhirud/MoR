import sys
sys.path.append('/home/yongjia/dgl/Yongjia/MOE_20250222/')

import torch
import torch.nn as nn
import ast
from unsloth import FastLanguageModel
import json
from transformers import TextStreamer
import contractions
import re
from Planning.utils import remove_inner_single_quotes


class Planner(nn.Module):
    def __init__(self, dataset_name):
        super(Planner, self).__init__()
        self.dataset_name = dataset_name
        self.checkpoint_path = f"/home/yongjia/dgl/Yongjia/MOE_20250222/Planning/checkpoints/{dataset_name}/lora_model/"
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True
        
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = self.checkpoint_path,
        max_seq_length = self.max_seq_length,
        dtype = self.dtype,
        load_in_4bit = self.load_in_4bit
        )

        FastLanguageModel.for_inference(model)

        self.model = model
        self.tokenizer = tokenizer
        
    def forward(self, query):
        message = {'content': query, 'role': 'user'}
        inputs = self.tokenizer.apply_chat_template(
        [message],
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
        ).to("cuda")

        text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        outputs = self.model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128, # max_new_tokens is the maximum number of new tokens generated beyond the input
                        use_cache = True, temperature = 1.5, min_p = 0.1) # min_p is a cumulative probability, which makes the generation more diverse
        
        
        outputs = self.tokenizer.batch_decode(outputs)
        parts = outputs[0].split("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        if len(parts) > 1:
            results = parts[1].replace("<|eot_id|>", "")
        else:
            raise ValueError


        # ******* special processing for prime dataset
        if self.dataset_name == 'prime':
            try:
                    # Parse the string using ast.literal_eval
                    parsed_dict = ast.literal_eval(results)
                    
                    return parsed_dict
            except (SyntaxError, ValueError) as e:
                    print(f"Error parsing the string: {e}")
                    return {
                            "Metapath": "",
                            "Restriction": {}
                        }
        
        
        results = contractions.fix(results)

        try:
            results = ast.literal_eval(results)
        except:
            print(f"Fail")
            try:
                results = re.sub(r"\['(.*?)'", remove_inner_single_quotes, results) # TODO: need optimize
                results = ast.literal_eval(results)
            except:
                results = {
                "Metapath": "",
                "Restriction": {},
                
            }
        rg = results

            
        return rg