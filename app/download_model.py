from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from pathlib import Path
import argparse

def download_model(
    model_id: str,
    base_path: str = "models/local-llm",
    trust_remote_code: bool = True
):
    """
    Download a model from Hugging Face and save it in a properly named folder.
    
    Args:
        model_id: The Hugging Face model ID
        base_path: Base directory where models will be stored
        trust_remote_code: Whether to trust remote code when loading models
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_name = model_id.split('/')[-1].lower()
    model_path = Path(base_path) / model_name
    model_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading model {model_id} to {model_path}")
    
    try:
        # Download and save tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
        tokenizer.save_pretrained(model_path)
        
        # Download and save model
        logger.info("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
        model.save_pretrained(model_path)
        
        logger.info(f"Model {model_id} successfully downloaded to {model_path}")
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face")
    parser.add_argument("model_id")
    parser.add_argument("--base-path", default="models/local-llm")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    
    args = parser.parse_args()
    
    download_model(
        model_id=args.model_id,
        base_path=args.base_path,
        trust_remote_code=not args.no_trust_remote_code
    ) 