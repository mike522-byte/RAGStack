import subprocess
import sys
import os
import time
import signal
import argparse

def run_api(port=8000):
    """Run the FastAPI server."""
    api_cmd = [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", str(port)]
    return subprocess.Popen(api_cmd)

def run_frontend(port=8501):
    """Run the Streamlit frontend."""
    frontend_cmd = [sys.executable, "-m", "streamlit", "run", "app/main.py", "--server.port", str(port)]
    return subprocess.Popen(frontend_cmd)

def main():
    parser = argparse.ArgumentParser(description="Run the RAG Prototype")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for the API server")
    parser.add_argument("--frontend-port", type=int, default=8501, help="Port for the Streamlit frontend")
    parser.add_argument("--model-name", type=str, default="mistral-7b-instruct-v0.2", help="Name of the model to use")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["USE_4BIT"] = str(args.use_4bit).lower()
    
    print(f"Starting RAG with model: {args.model_name}")
    print(f"4-bit quantization: {'Enabled' if args.use_4bit else 'Disabled'}")
    
    # Start the API server
    api_process = run_api(args.api_port)
    print(f"API server started on port {args.api_port}")
    
    # Wait for the API server to start
    time.sleep(2)
    
    # Start the Streamlit frontend
    frontend_process = run_frontend(args.frontend_port)
    print(f"Streamlit frontend started on port {args.frontend_port}")
    
    try:
        # Keep the script running
        api_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
        api_process.terminate()
        frontend_process.terminate()
        api_process.wait()
        frontend_process.wait()
        print("Shutdown complete")

if __name__ == "__main__":
    main() 