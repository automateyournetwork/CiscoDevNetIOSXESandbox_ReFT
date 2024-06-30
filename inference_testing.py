import torch
import transformers
import pyreft

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the prompt template
prompt_no_input_template = """<s>[INST] <<SYS>>
You are an expert specialized in Cisco IOS XE running configurations and computer networking.
<</SYS>>

%s [/INST]
"""

# Load the base model
model_name_or_path = "meta-llama/Meta-Llama-3-8B"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map="auto" if device == "cuda" else None
)

# Load the fine-tuned reft model
reft_model = pyreft.ReftModel.load(
    "automateyournetwork/Cisco_DevNet_Sandbox_Running_Config", model
)

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, 
    padding_side="right", use_fast=False
)

# Set pad_token as eos_token
tokenizer.pad_token = tokenizer.eos_token

# Define the instruction
instruction = "Which dog breed do people think is cuter, poodle or doodle?"

# Tokenize and prepare the input
prompt = prompt_no_input_template % instruction
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate the response
base_unit_location = inputs["input_ids"].shape[-1] - 1  # last position
_, reft_response = reft_model.generate(
    **inputs, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
    eos_token_id=tokenizer.eos_token_id, early_stopping=True
)

# Decode and print the response
print(tokenizer.decode(reft_response[0], skip_special_tokens=True))
