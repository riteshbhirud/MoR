"""
Desc: This file is used to get the training data from the LLM

"""
import sys
from pathlib import Path

# Get the absolute path of the current script
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]

# Add the project root to the system path
sys.path.append(str(project_root))

from stark_qa import load_qa

import argparse
import os
from openai import AzureOpenAI
import json
import openai
from prompts import prompts






"""

MAG:
sys_content: 478/query
output: 45/query
input: 25/query
1000 queries

total price:
    1. o1: $13.29
    2. o3mini: $0.97
    3. deepseek-chat: $0.24
    4. deepseek-reasoner: $0.49
    
Amazon:
sys_content: 478/query

"""

# get the prompt for different datasets
def get_sys_content(dataset_name):
    """
        input: 
            dataset_name: the name of the dataset
        output:
            sys_content: the sys_content for the dataset
    """
    sys_content = prompts(dataset_name)
    
    
    return sys_content

# get the response from the llm
def get_response(sys_content, user_content):
    
    messages = [{"role": "system", "content": sys_content},
               {"role": "user", "content": user_content}
               ]
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=parameters['azure']['model'], # parameters['azure']['model'], parameters['openai']['model']
        # temperature=0,
        seed=576879897,
    )
    response = chat_completion.choices[0].message.content
    
    # print(messages)
    # print(response)
    
    return response

# save the outputs to json file
def save_json(data, dataset_name):
    """
        input: 
            data: the data to be saved
            dataset_name: the name of the dataset
    """
    
    file_dir = f"/home/yongjia/dgl/Yongjia/MOE/Reasoner/data/finetune/{dataset_name}"
    os.makedirs(file_dir, exist_ok=True)
    file_path = f"{file_dir}/1000_{parameters['azure']['model']}.json"
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {file_path}")

# get the reasoning graphs for a dataset
def get_rg(dataset_name):
    """
        input: 
            dataset_name: the name of the dataset
        output:
            rg: the reasoning graph for the dataset
    """
    
    # get the prompt for the dataset
    sys_content = get_sys_content(dataset_name)
    
    # get qa dataset 
    qa = load_qa(dataset_name)
    train_qa = qa.get_subset('train')
    
    # we sample 1000 queries from the training set
    pair_list = []
    failure_count = 0
    for i in range(1500):
        query, q_id, ans_ids, _ = train_qa[i]
    
        # call the llm to get the reasoning graph
        response = get_response(sys_content, query)
        print(response)
        
        # process the response
    
        if dataset_name == 'prime':
            output = {
                "Triplets":[],
                "Restriction": [],
                "Target": ""
            }
            
            try:
                response = response.split('\n')
                triplets_raw = response[0].replace('Triplets:', '').strip()
                triplets = json.loads(triplets_raw)
                output['Triplets'] = triplets
                
                restriction_raw = response[1].replace('Restriction:', '').strip()
                restriction = json.loads(restriction_raw)
                output['Restriction'] = restriction
                
                target = response[2].replace('Target:', '').strip()
                output['Target'] = target
            except:
                failure_count += 1
                continue
            
        elif dataset_name == 'mag' or dataset_name == 'amazon':
            output = {
                "Metapath": "",
                "Restriction": [],
            }
            
            try:
                response = response.split('\n')
                metapath = response[0].replace('Metapath:', '').strip()
                output['Metapath'] = metapath
                
                restriction_raw = response[1].replace('Restriction:', '').strip()
                restriction = json.loads(restriction_raw)
                output['Restriction'] = restriction
            except:
                failure_count += 1
                continue
            
        else:
            raise ValueError('The dataset is not supported')
        
        pair = {'query': query, 'answer': output}

        pair_list.append(pair)
        
        if len(pair_list) == 1000:
            break
            
    # save the output to json file
    save_json(pair_list, dataset_name)
    print(f"Failure count: {failure_count}")
    

if __name__ == '__main__':    
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Load LLM parameters and initialize API clients.")
    
    # Dataset name
    parser.add_argument("--dataset_name", type=str, required=True, 
                        choices=["mag", "amazon", "prime"], 
                        help="Specify the dataset to use.")

    # Model selection
    parser.add_argument("--model", type=str, required=True, 
                        choices=["gpt-4o-mini-20240718", "gpt-4o-2024-05-13", 
                                "deepseek-reasoner", "gpt-o1-2024-12-17", 
                                "o3-mini-2025-01-31"],
                        help="Specify the model to use.")

    # Azure API parameters
    parser.add_argument("--azure_api_key", type=str, default=None, help="Azure API Key")
    parser.add_argument("--azure_endpoint", type=str, default=None, help="Azure API Endpoint")
    parser.add_argument("--azure_api_version", type=str, default=None, help="Azure API Version")

    # OpenAI API parameters
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API Key")
    parser.add_argument("--openai_endpoint", type=str, default=None, help="OpenAI API Endpoint")

    args = parser.parse_args()

    # Initialize parameters dictionary
    parameters = {
        "azure": {
            "api_key": args.azure_api_key,
            "azure_endpoint": args.azure_endpoint,
            "api_version": args.azure_api_version,
        },
        "openai": {
            "api_key": args.openai_api_key,
            "endpoint": args.openai_endpoint,
        }
    }


    # Determine which API client to use
    if parameters["openai"]["api_key"]:
        client = openai.OpenAI(
            base_url=parameters["openai"]["endpoint"],
            api_key=parameters["openai"]["api_key"],
        )
    else:
        client = AzureOpenAI(
            azure_endpoint=parameters["azure"]["azure_endpoint"],
            api_key=parameters["azure"]["api_key"],
            api_version=parameters["azure"]["api_version"],
        )
    
    get_rg(args.dataset_name)


