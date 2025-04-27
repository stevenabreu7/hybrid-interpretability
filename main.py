# ! git clone https://github.com/stevenabreu7/hybrid-interpretability
# ! cd hybrid-interpretability
# ! bash install.sh
import kagglehub
import os
import pathlib
import torch
import sentencepiece as spm
from recurrentgemma import torch as recurrentgemma
import subprocess
import shutil


def create_zip_archive(source_dir, output_filename):
    """
    Create a zip archive of the specified directory
    """
    print(f"Creating archive {output_filename} from {source_dir}")
    shutil.make_archive(
        output_filename.replace('.zip', ''),  # remove .zip extension for make_archive
        'zip',
        os.path.dirname(source_dir),
        os.path.basename(source_dir)
    )
    print(f"Archive created: {output_filename}")


def run_niah():
    """
    Run the NIAH (Needle In A Haystack) workflow
    """
    print("Running NIAH workflow...")
    
    # Change to project directory if needed
    # os.chdir('/path/to/your/project')
    
    # Install dependencies
    subprocess.run(["pip", "install", "flash-attn", "--no-build-isolation"], check=True)
    subprocess.run(["pip", "install", "python-dotenv", "tiktoken", "anthropic"], check=True)
    
    # Set environment variables
    # Replace with your own HF token or load from .env file
    os.environ['HF_TOKEN'] = 'your_huggingface_token'
    
    # Run NIAH scripts
    print("Running prompt.py...")
    subprocess.run(["python", "NIAH/Needle_test/prompt.py"], check=True)
    
    print("Running predictions...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(["python", "NIAH/Needle_test/pred.py"], check=True)
    
    print("Running evaluation...")
    # Create environment variables
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Set up Huggingface cache
    hf_token = os.environ.get('HF_TOKEN')
    custom_args = {"token": hf_token, "cache_dir": "./.cache"}  # speeds up loading
    
    # Run evaluation
    subprocess.run(["python", "NIAH/Needle_test/eval.py"], env=my_env, check=True)
    
    # Run visualization
    subprocess.run(["python", "LongAlign/Needle_test/vis.py"], check=True)
    
    # Create archive files
    create_zip_archive("LongAlign/Needle_test/prompts", "prompt.zip")
    create_zip_archive("LongAlign/Needle_test/pred_K1", "pred_K1.zip")
    create_zip_archive("LongAlign/Needle_test/pred_K2", "pred_K2.zip")
    create_zip_archive("LongAlign/Needle_test/results", "results.zip")
    
    print("NIAH workflow completed")


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Login to Kaggle
    kagglehub.login()

    # Import model weights
    google_recurrentgemma_pytorch_2b_it_1_path = kagglehub.model_download('google/recurrentgemma/PyTorch/2b/1')
    
    # Set model paths
    VARIANT = '2b'
    weights_dir = pathlib.Path(f"{google_recurrentgemma_pytorch_2b_it_1_path}")
    ckpt_path = weights_dir / f'{VARIANT}.pt'
    vocab_path = weights_dir / 'tokenizer.model'
    preset = recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1 if '2b' in VARIANT else recurrentgemma.Preset.RECURRENT_GEMMA_9B_V1

    # Load parameters
    params = torch.load(str(ckpt_path))
    params = {k : v.to(device=device) for k, v in params.items()}

    # Initialize model
    model_config = recurrentgemma.GriffinConfig.from_torch_params(
        params,
        preset=preset,
    )
    model = recurrentgemma.Griffin(model_config, device=device, dtype=torch.bfloat16)
    model.load_state_dict(params)

    # Enable sparsification
    model.enable_sparsification(k = 3, metric = "entropy", prefill = False)

    # Load vocabulary
    vocab = spm.SentencePieceProcessor()
    vocab.Load(str(vocab_path))

    # Initialize sampler
    sampler = recurrentgemma.Sampler(model=model, vocab=vocab)

    # Generate text
    input_batch = ["I once had a girl, or should I say, she once had  "]
    
    # 30 generation steps
    out_data = sampler(input_strings=input_batch, total_generation_steps=30)

    for input_string, out_string in zip(input_batch, out_data.text):
        print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
        print(10*'#')

    # Test attention sparsification
    input_batch = ["I once had a girl, or should I say, she once had "]

    model.disable_attention_manipulation()
    model.enable_sparsification(k = 3, metric = "entropy", prefill = False)

    # 30 generation steps
    out_data = sampler(input_strings=input_batch, total_generation_steps=30)

    for input_string, out_string in zip(input_batch, out_data.text):
        print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
        print(10*'#')

    # Run NIAH workflow
    run_niah()


if __name__ == "__main__":
    main()
