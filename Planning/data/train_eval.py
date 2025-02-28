"""
Description: This file is used to train and evaluate the llama model.

"""
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from datasets import load_dataset
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import get_chat_template
from sklearn.model_selection import train_test_split
from transformers import TextStreamer
import ast
import contractions
import re
from utils import remove_inner_single_quotes

# *****load model and tokenizer*****
max_seq_length = 2048
dtype = None    # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,    # specify the maximum length of input the model can accept
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# add LoRA adapters so we only need to update 1 to 10% of all parameters!
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",# Reference: https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=2eSvM9zX_2d3
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(data):
    convos = data['conversations']
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    
    return { "text" : texts, }

def train(train_dataset, val_dataset):
    # format the dataset
    texts_train = train_dataset.map(formatting_prompts_func, batched=True)
    print(texts_train)
    texts_val = val_dataset.map(formatting_prompts_func, batched=True)
    print(texts_val)
    
    # load trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = texts_train,
        eval_dataset= texts_val,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),  # A function or mechanism to batch and prepare data during training and evaluation
        dataset_num_proc = 2,   # number of processes to use for data preprocessing
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 4, 
            gradient_accumulation_steps = 8,
            warmup_steps = 5,
            num_train_epochs = 100, # Set this for 1 full training run. # TODO: increase the value
            max_steps = 1000, # TODO: increase the value
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            do_eval=True,
            report_to='wandb',
            evaluation_strategy="epoch",    # Specifies that evaluations will happen at the end of each epoch, using texts_val for metrics calculation. If we set the evaluation strategy without passing evaluation dataset, there is an error.
            save_strategy="epoch",      # Save checkpoints based on epoch.
            load_best_model_at_end=True,    # Load the best one from the disk to memory. Otherwise, the model is still the one trained after last epoch.
            metric_for_best_model="loss",       # Evaluation metric on evaluation set. Make sure the metric is an option included in TrainingArguments.
            greater_is_better=False,     # The metric is "greater_is_better".
            save_total_limit=1      # How many checkpoints will be saved.
        ),
    )
    
    # train only on outputs and ignore the loss of user's inputs
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer_stats = trainer.train()
    
    # save the model
    checkpoint_dir = f"./checkpoints/{dataset_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"lora_model")
    model.save_pretrained(checkpoint_path) # Local saving 
    tokenizer.save_pretrained(checkpoint_path)
    
    print("Training completed.")
    
    return checkpoint_path

def evaluate(test_dataset, checkpoint_path):
    
    # load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_path,
        max_seq_length = max_seq_length,    # specify the maximum length of input the model can accept
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    FastLanguageModel.for_inference(model)
    
    # evaluate
    acc = 0
    for idx, dp in enumerate(test_dataset): # TODO: batch
        message = dp['conversations'][0]
        label = dp['conversations'][1]
        assert label['role'] == "assistant"
        
        # format the input
        inputs = tokenizer.apply_chat_template([message], tokenize = True, add_generation_prompt = True, return_tensors = 'pt').to('cuda')
        print(f"222, {inputs}")
        
        text_streamer = TextStreamer(tokenizer, skip_prompt = True)
        outputs = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                        use_cache = True, temperature = 1.5, min_p = 0.1)
        outputs = tokenizer.batch_decode(outputs)
        parts = outputs[0].split("<|start_header_id|>assistant<|end_header_id|>\n\n")
        # print(f"111, {parts}")
        results = parts[1].strip("<|eot_id|>")
        results = contractions.fix(results)
        try:
            results = ast.literal_eval(results)
        except:
            try:
                results = re.sub(r"\['(.*?)'", remove_inner_single_quotes, results) 
                results = ast.literal_eval(results)
            except:
                results = {
                    "Metapath":"",
                    "Restriction":{},
                }

        
        
        pred_metapath = results['Metapath']
        
        ground_metapath = ast.literal_eval(label['content'])['Metapath']
        if pred_metapath == ground_metapath:
            acc += 1
        print(f"Prediction: {pred_metapath}")
        print(f"Ground truth: {ground_metapath}")
        
        
    
    print(f"Accuracy: {acc / len(test_dataset)}")
    

def main(dataset_name, model_name):
    # *****load dataset*****
    data_dir = f"./data/finetune"
    data_path = os.path.join(data_dir, f"{dataset_name}/llama_ft.jsonl")
    # data_path = os.path.join(data_dir, f"{dataset_name}/llama_ft_{model_name}.jsonl")
    dataset = load_dataset("json", data_files=data_path)
    dataset = dataset['train']

    # Add an index column to keep track of original indices
    dataset = dataset.add_column("index", list(range(len(dataset))))

    # Perform train-test split
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)

    # Extract the datasets
    train_dataset = train_test['train']
    val_dataset = val_test['train']
    test_dataset = val_test['test']

    # Remove the index column if not needed
    train_dataset = train_dataset.remove_columns(['index'])
    val_dataset = val_dataset.remove_columns(['index'])
    test_dataset = test_dataset.remove_columns(['index'])
    
    # *****train and evaluate*****
    checkpoint_path = train(train_dataset, val_dataset)
    
    
    checkpoint_path = f"./checkpoints/{dataset_name}/lora_model"
    # checkpoint_path = f"./checkpoints/{dataset_name}/lora_model_{model_name}"
    evaluate(test_dataset, checkpoint_path)

if __name__ == "__main__":
    dataset_name_list = ['mag', 'amazon', 'prime']
    model_names = ["4o"] 
    for dataset_name in dataset_name_list:
        for model_name in model_names:        
            main(dataset_name, model_name)
            
            
            
            