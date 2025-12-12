import torch
import json
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
MODEL_ID = "swiss-ai/Apertus-8B-Instruct-2509"

# INPUT DATASET
INPUT_DATASET = "obalcells/longfact-augmented-prompts"
SPLIT = "train"

# UPLOAD SETTINGS
TARGET_REPO_ID = "tymciurymciu/longfact-generations" 
UPLOAD_INTERVAL = 1000 
SUBSET_NAME = MODEL_ID.split("/")[-1].replace("-", "_")

# OUTPUT
BASE_OUTPUT_DIR = "/capstor/scratch/cscs/tkwiecinski/hallucination-probes/generation_pipeline/outputs/"
OUTPUT_FILE = BASE_OUTPUT_DIR + "longfact_generations_" + SUBSET_NAME + ".jsonl"

# GENERATION PARAMETERS
BATCH_SIZE = 64 # for apertus-8B
MAX_NEW_TOKENS = 2048 
TEMPERATURE = 0.1
DO_SAMPLE = True 

# ==========================================
# SETUP & LOADING
# ==========================================
print(f"Loading dataset {INPUT_DATASET} ({SPLIT}) from hgf...")
dataset = load_dataset(INPUT_DATASET, split=SPLIT)

print(f"Loading model {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2" 
)
model.eval()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def push_progress_to_hub(current_results, repo_id, step_count):
    """Creates a dataset from current results and pushes to Hub."""
    print(f"\n[Step {step_count}] Syncing {len(current_results)} items to Hugging Face ({repo_id})...")
    try:
        temp_dataset = Dataset.from_list(current_results)
        temp_dataset.push_to_hub(repo_id, split=SPLIT, config_name=SUBSET_NAME, private=False, token=os.environ.get("HF_WRITE_TOKEN"))
        print("Sync complete.")
    except Exception as e:
        print(f"Sync failed: {e}")
        print("Continuing generation (local progress is safe)...")

# ==========================================
# RESUME LOGIC
# ==========================================
results = []
start_idx = 0

if os.path.exists(OUTPUT_FILE):
    print(f"Found existing output file: {OUTPUT_FILE}")
    print("Reading processed items...")
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        num_processed = len(results)
        print(f"Loaded {num_processed} previously generated items.")
        
        if num_processed > 0:
            # 1. Inspect last item
            last_item = results[-1]
            last_prompt = last_item["conversation"][0]["content"]
            print(f"Last processed prompt (preview): {last_prompt[:100]}...")
            
            # 2. Push current state to Hub (Safety check)
            print("Ensuring existing progress is synced to Hub before resuming...")
            push_progress_to_hub(results, TARGET_REPO_ID, num_processed)
            
            # 3. Determine where to resume
            start_idx = num_processed
            print(f"Resuming generation from index {start_idx}...")
            
    except Exception as e:
        print(f"Error reading checkpoint file: {e}")
        print("Starting from scratch (index 0).")
        results = []
        start_idx = 0
else:
    print("No checkpoint found. Starting from scratch.")

# ==========================================
# GENERATION LOOP
# ==========================================
# We slice the dataset to skip already processed items
# Note: Dataset slicing in HF datasets is efficient (lazy)
remaining_dataset = dataset.select(range(start_idx, len(dataset)))

print(f"Starting generation for remaining {len(remaining_dataset)} prompts...")

count_since_last_upload = 0
total_processed = len(results) # Global counter including resumed items

# Iterate over the REMAINING part of the dataset
for batch in tqdm(remaining_dataset.iter(batch_size=BATCH_SIZE), total=len(remaining_dataset)//BATCH_SIZE + 1):
    
    if "prompt" in batch:
        prompts = batch["prompt"]
    else:
        prompts = batch["question"]

    # chat template
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], 
            tokenize=False, 
            add_generation_prompt=True
        ) for p in prompts
    ]

    # tokenize
    inputs = tokenizer(
        formatted_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=4096
    ).to(model.device)

    # generate!
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE, 
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id
        )

    # decode
    input_len = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_len:]
    decoded_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # finish up
    current_batch_size = len(prompts)
    
    for idx in range(current_batch_size):
        original_prompt = prompts[idx]
        response = decoded_responses[idx]
        
        entry = {
            "conversation": [
                {"role": "user", "content": original_prompt},
                {"role": "assistant", "content": response}
            ]
        }
        
        # extra fields
        for key in batch.keys():
            if key not in ["prompt", "question"]: 
                entry[key] = batch[key][idx]
                
        results.append(entry)

    total_processed += current_batch_size
    count_since_last_upload += current_batch_size

    if count_since_last_upload >= UPLOAD_INTERVAL:
        # Save locally (checkpoint) - Append mode is risky if mixed with rewriting, 
        # so we rewrite full list to ensure consistency or append just new ones.
        # Rewriting is safer for avoiding dupes if crash happens during write.
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
        
        # Then push to hub
        push_progress_to_hub(results, TARGET_REPO_ID, total_processed)
        count_since_last_upload = 0 

# ==========================================
# FINAL SAVE
# ==========================================
print(f"Final save to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

print(f"Final push to Hugging Face ({TARGET_REPO_ID})...")
push_progress_to_hub(results, TARGET_REPO_ID, total_processed)

print("done c;")
