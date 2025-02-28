"""
Description: This script is used to post-process the data after getting it from the LLM.

input: data from llm
output: data for llama finetuning

"""
import json
import os
import copy

def save_data(data, dataset_name, model_name):
    file_dir = f"../data/finetune/{dataset_name}"
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, f"llama_ft_{model_name}.jsonl")
    
    with open(file_path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    
    print(f"Data saved to {file_path}")
    
def process(sample, dataset_name, model_name):
    """
    input: sample from llm
    output: sample for llama finetuning
    """
   
    output_format = {"conversations": [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
    ft_list = []
    for i in range(len(sample)):
        output = copy.deepcopy(output_format)
        output["conversations"][0]["content"] = sample[i]["query"]
        output["conversations"][1]["content"] = str(sample[i]["answer"])        
        ft_list.append(output)
    
    # save data
    save_data(ft_list, dataset_name, model_name)
        
    
# ***** Main *****
if __name__ == "__main__":
    # read data
    dataset_name_list = ["mag"]
    model_names = ["gpt-4o-mini-20240718", "o3-mini-2025-01-31", "gpt-o1-2024-12-17", "gpt-4o-2024-05-13", "gpt-4o-mini-20240718", "gpt35-1106"] # gpt-o1-2024-12-17, "gpt-4o-mini-20240718", "gpt35-1106", o3-mini-2025-01-31
    
    for dataset_name in dataset_name_list:
        for model_name in model_names:
            relative_path = f"finetune/{dataset_name}/1000.json" #  f"finetune/{dataset_name}/1000_{dataset_name}.json"
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Directory of the current script
            file_path = os.path.join(current_dir, relative_path)
            with open(f"{file_path}", "r") as f:
                sample = json.load(f)
            
            # process data
            process(sample, dataset_name, model_name)
            print(f"Processing {model_name} for {dataset_name} is done.")
            break