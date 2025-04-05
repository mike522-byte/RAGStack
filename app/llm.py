from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import List, Dict
import logging
from pathlib import Path

class LocalLLM:
    def __init__(
        self,
        model_name: str,
        base_path: str = "models/local-llm",
        device: str = "auto",
        use_4bit: bool = True
    ):
        self.model_name = model_name
        self.model_path = Path(base_path) / model_name
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        try:
            # Check if model exists locally
            if not (self.model_path / "config.json").exists():
                self.logger.warning(f"No model found at {self.model_path}. Please run download_model.py first.")
                raise FileNotFoundError(f"No model found at {self.model_path}")
            
            # Load tokenizer
            self.logger.info(f"Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            # Configure quantization if using 4-bit
            if use_4bit and self.device == "cuda":
                self.logger.info("Configuring 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None
            
            # Load model
            self.logger.info(f"Loading model {model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.logger.info(f"Model {model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def _create_prompt(self, query: str, context: List[Dict]) -> str:
        """Create a prompt from the query and context."""
        context_str = "\n".join([chunk['content'] for chunk in context])
        
        prompt = f"""<s>[INST] <<SYS>>
You are a helpful AI assistant. Answer the question based on the provided context.
If you cannot answer based on the context, say "I cannot answer this question based on the provided context."
<</SYS>>

Context:
{context_str}

Question: {query} [/INST]"""
        
        return prompt

    def generate_response(self, query: str, relevant_chunks: List[Dict]) -> Dict:
        """Generate a response using the local LLM."""
        try:
            # Create the prompt
            prompt = self._create_prompt(query, relevant_chunks)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True
                )
            
    
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the answer part (everything after the last [/INST])
            answer = response.split("[/INST]")[-1].strip()
            
            return {
                "answer": answer,
                "sources": relevant_chunks
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error while generating the response.",
                "sources": []
            }

    def __call__(self, query: str, relevant_chunks: List[Dict]) -> Dict:
        """Make the class callable."""
        return self.generate_response(query, relevant_chunks)
    
if __name__ == "__main__":

    llm = LocalLLM(
        model_name="mistral-7b-instruct-v0.2",
        use_4bit=True  # Enable 4-bit quantization
    )
    print(llm("What is the capital of France?", []))
    

