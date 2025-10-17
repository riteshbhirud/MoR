"""
Planning Agent: Generates Cypher queries using trainable LLM with LoRA.
Integrates with existing MoR Cypher execution logic.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging
from typing import Dict, List, Tuple, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanningAgent(nn.Module):
    """
    Trainable planning agent using LLM with LoRA fine-tuning.
    Generates Cypher queries for graph-based retrieval.
    """
    
    def __init__(
        self, 
        base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: str = 'cuda',
        use_4bit: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Planning Agent.
        
        Args:
            base_model_name: HuggingFace model identifier
            lora_r: LoRA rank (lower = more efficient)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout for LoRA layers
            device: Device to load model on
            use_4bit: Whether to use 4-bit quantization for memory efficiency
            cache_dir: Directory to cache model weights
        """
        super().__init__()
        self.device = device
        self.base_model_name = base_model_name
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer from {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure quantization if requested
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load base LLM
        logger.info(f"Loading base model from {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if not use_4bit else None,
            device_map="auto" if use_4bit else None,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training if using quantization
        if use_4bit:
            base_model = prepare_model_for_kbit_training(base_model)
        
        # Apply LoRA
        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        
        if not use_4bit:
            self.model = self.model.to(device)
        
        # Get hidden size from model config
        self.hidden_size = self.model.config.hidden_size
        
        # Value network for PPO advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(device)
        
        logger.info("Planning Agent initialized successfully")
    
    def _construct_cypher_prompt(
        self, 
        query: str, 
        schema: Dict[str, List[str]]
    ) -> str:
        """
        Construct prompt for Cypher generation following MoR conventions.
        
        Args:
            query: Natural language query
            schema: Graph schema with node_types, edge_types, properties
            
        Returns:
            Formatted prompt string
        """
        node_types_str = ", ".join(schema.get('node_types', []))
        edge_types_str = ", ".join(schema.get('edge_types', []))
        properties_str = ", ".join(schema.get('properties', []))
        
        prompt = f"""You are an expert at converting natural language questions into Cypher queries for graph databases.

Given the following graph schema:
- Node types: {node_types_str}
- Edge types: {edge_types_str}
- Properties: {properties_str}

Convert the following question into a Cypher query. Follow these rules:
1. Use MATCH to find patterns
2. Use WHERE for constraints
3. Use RETURN for the target variable
4. Keep the query simple and efficient
5. Use standard Cypher syntax

Question: {query}

Generate ONLY the Cypher query without any explanation:"""
        
        return prompt
    
    def generate_cypher(
        self, 
        query: str, 
        schema: Dict[str, List[str]], 
        temperature: float = 0.7,
        max_new_tokens: int = 200,
        do_sample: bool = True
    ) -> Tuple[str, torch.Tensor]:
        """
        Generate Cypher query from natural language query.
        
        Args:
            query: Natural language query
            schema: Graph schema (node types, edge types, properties)
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to sample or use greedy decoding
        
        Returns:
            cypher: Generated Cypher query
            log_prob: Log probability for training (scalar tensor)
        """
        # Construct prompt
        prompt = self._construct_cypher_prompt(query, schema)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract generated tokens (excluding prompt)
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        
        # Decode Cypher
        cypher = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up the Cypher query
        cypher = self._clean_cypher(cypher)
        
        # Compute log probabilities for policy gradient
        log_probs = []
        for i, score in enumerate(outputs.scores):
            if i < len(generated_ids):
                token_id = generated_ids[i]
                log_prob = torch.log_softmax(score[0], dim=-1)[token_id]
                log_probs.append(log_prob)
        
        # Sum log probabilities
        if len(log_probs) > 0:
            total_log_prob = torch.stack(log_probs).sum()
        else:
            total_log_prob = torch.tensor(0.0, device=self.device)
        
        return cypher, total_log_prob
    
    def _clean_cypher(self, cypher: str) -> str:
        """
        Clean and validate generated Cypher query.
        
        Args:
            cypher: Raw generated Cypher
            
        Returns:
            Cleaned Cypher query
        """
        # Remove markdown code blocks if present
        cypher = re.sub(r'```cypher\s*', '', cypher)
        cypher = re.sub(r'```\s*', '', cypher)
        
        # Remove extra whitespace
        cypher = ' '.join(cypher.split())
        
        # Ensure it starts with MATCH (most common pattern)
        if not cypher.strip().upper().startswith('MATCH'):
            # Try to find MATCH in the string
            match_idx = cypher.upper().find('MATCH')
            if match_idx != -1:
                cypher = cypher[match_idx:]
        
        return cypher.strip()
    
    def compute_value(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """
        Estimate value (expected future reward) for baseline in PPO.
        
        Args:
            query_embedding: Query representation [batch_size, hidden_size]
        
        Returns:
            value: Estimated value [batch_size]
        """
        value = self.value_head(query_embedding)
        return value.squeeze(-1)
    
    def get_query_embedding(self, query: str, schema: Dict[str, List[str]]) -> torch.Tensor:
        """
        Get query embedding for value estimation.
        
        Args:
            query: Natural language query
            schema: Graph schema
            
        Returns:
            query_embedding: [hidden_size]
        """
        prompt = self._construct_cypher_prompt(query, schema)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model.model(**inputs, output_hidden_states=True)
            
            # Use last hidden state of last token as query embedding
            query_embedding = outputs.hidden_states[-1][0, -1, :]
        
        return query_embedding
    
    def save_pretrained(self, save_path: str):
        """Save LoRA weights and value head."""
        logger.info(f"Saving Planning Agent to {save_path}")
        self.model.save_pretrained(save_path)
        torch.save(self.value_head.state_dict(), f"{save_path}/value_head.pt")
        logger.info("Planning Agent saved successfully")
    
    def load_pretrained(self, load_path: str):
        """Load LoRA weights and value head."""
        logger.info(f"Loading Planning Agent from {load_path}")
        # LoRA weights are loaded via from_pretrained in __init__
        value_head_path = f"{load_path}/value_head.pt"
        self.value_head.load_state_dict(torch.load(value_head_path, map_location=self.device))
        logger.info("Planning Agent loaded successfully")