import os
import csv
import torch
import requests
import transformers
import pyreft
from huggingface_hub import login

# Function to load the Hugging Face API key from a file
def load_hf_api_key(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print("API key file not found.")
        return None

# Path to your API key file
api_key_file_path = 'huggingfacekey.txt'

# Load the API key
hf_api_key = load_hf_api_key(api_key_file_path)

if hf_api_key:
    # Use the API key to log in
    login(hf_api_key)
    print("Logged in to Hugging Face Hub successfully.")
else:
    print("Failed to load Hugging Face API key.")

# Citations for academic papers referenced in this implementation
# Citation for ReFT: Representation Finetuning for Language Models
# Wu, Zhengxuan et al. (2024). ReFT: Representation Finetuning for Language Models.
# arXiv:2404.03592. Available at: https://arxiv.org/abs/2404.03592

# Citation for pyvene: A Library for Understanding and Improving PyTorch Models via Interventions
# Wu, Zhengxuan et al. (2024). pyvene: A Library for Understanding and Improving PyTorch Models via Interventions.
# Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: System Demonstrations.
# arXiv:2403.07809. Available at: https://arxiv.org/abs/2403.07809

# Run pyATS job to get the show_run.txt file
def run_pyats_job():
    job_status = os.system("pyats run job pyats_show_run_job.py")
    return "Job executed successfully" if job_status == 0 else "Job failed"

def load_config(file_path='show_run.txt'):
    try:
        with open(file_path, 'r') as file:
            return file.read().split('!')
    except FileNotFoundError:
        return "Configuration file not found."

def send_request(model, running_config_chunk):
    url = f"http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    
    prompt = (
        f"The following text is from a Cisco IOS XE running configuration. Generate randomized questions about the text to form a dataset being used to fine-tune a model. "
        f"Only generate questions if the answer can be directly found within the text provided. Do not generate questions if the answer is not explicitly present in the text. "
        f"Include specific details from the configuration in the questions to make them clear and contextually accurate.\n\n"
        f"For example, if you see a configuration like this:\n"
        f"interface Loopback44\n"
        f" description DevNet Expert\n"
        f" no ip address\n"
        f" shutdown\n"
        f"Or like this:\n"
        f"interface Loopback1\n"
        f" description Configured via RESTCONF\n"
        f" ip address 1.1.1.1 255.255.255.0\n"
        f"!\n"
        f"Make sure your questions are like this:\n"
        f"Q: What is the description of interface Loopback44?\n"
        f"A: DevNet Expert\n"
        f"Q: Does interface Loopback44 have an IP address configured?\n"
        f"A: No\n"
        f"Q: Is interface Loopback44 shut down?\n"
        f"A: Yes\n"
        f"Q: What is the description of interface Loopback1?\n"
        f"A: Configured via RESTCONF\n"
        f"Q: What is the IP address of interface Loopback1?\n"
        f"A: 1.1.1.1\n"
        f"Q: What is the subnet mask of interface Loopback1?\n"
        f"A: 255.255.255.0\n\n"
        f"For VRF configurations like this:\n"
        f"vrf definition CHEMICAL\n"
        f" description CHEMICAL ENGINEERING FIRM\n"
        f" rd 65000:2\n"
        f" route-target export 65000:2\n"
        f" route-target import 65000:2\n"
        f"!\n"
        f"Make sure your questions are like this:\n"
        f"Q: What is the description of the VRF definition CHEMICAL?\n"
        f"A: CHEMICAL ENGINEERING FIRM\n"
        f"Q: What is the RD value for the VRF definition CHEMICAL?\n"
        f"A: 65000:2\n"
        f"Q: What is the export route-target for the VRF definition CHEMICAL?\n"
        f"A: 65000:2\n"
        f"Q: What is the import route-target for the VRF definition CHEMICAL?\n"
        f"A: 65000:2\n\n"
        f"Now generate questions and answers for the following chunk of configuration:\n"
        f"---Running Configuration Chunk---\n{running_config_chunk}\n"
        f"Format each question with a 'Q: ' prefix and each answer with an 'A: ' prefix on the next line. Only include questions that can be answered based on the given text. Do not include questions without answers.\n"
    )


    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json().get('response', '')
        return results
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def send_control_request(instruction):
    url = f"http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    
    prompt = prompt_no_input_template % instruction

    data = {
        "model": "llama3",  # Base model
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json().get('response', '')
        return results
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def save_to_csv(questions_answers, filename="output.csv", mode='a'):
    existing_pairs = set()
    
    if os.path.exists(filename):
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header row if it exists
            for row in reader:
                existing_pairs.add(tuple(row))

    with open(filename, mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        # Write the header if the file is empty
        if os.stat(filename).st_size == 0:
            writer.writerow(["Question", "Fine-tuned Answer"])
            print("Header written to CSV file.")
        
        # Write each question-answer pair, avoiding duplicates
        for item in questions_answers:
            if len(item) == 2:
                question, answer = item
                if (question, answer) not in existing_pairs:
                    print(f"Writing to CSV: {question} | {answer}")  # Debugging output
                    writer.writerow([question, answer])
                    existing_pairs.add((question, answer))

def load_training_examples(filename='output.csv'):
    training_examples = []
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                # Check if the answer field is not empty
                if len(row) >= 2 and row[1].strip():
                    training_examples.append([row[0], row[1]])  # Assuming row format is correct
                else:
                    print(f"Skipped row with empty answer: {row}")
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return training_examples

def format_qa_pairs(model_response):
    lines = model_response.strip().split("\n")
    formatted_pairs = []
    question = None
    for line in lines:
        if line.startswith("Q") and ":" in line:
            question = line.split(":", 1)[1].strip()  # Extract question
        elif line.startswith("A") and question:
            answer = line.split(":", 1)[1].strip()  # Extract answer
            formatted_pairs.append([question, answer])  # Append as list of [question, answer]
            question = None  # Reset question for the next pair
    return formatted_pairs

def generate_dataset(num_iterations=5):
    running_config_chunks = load_config()
    if running_config_chunks == "Configuration file not found.":
        print(running_config_chunks)
        return
    
    models = ["llama3", "gemma2"]
    for iteration in range(num_iterations):
        for i, chunk in enumerate(running_config_chunks):
            if chunk.strip():  # Skip empty chunks
                for model in models:
                    result = send_request(model, chunk)
                    if not result:
                        print(f"No response or an error occurred while fetching the model response for chunk {i+1} in iteration {iteration+1}.")
                        continue  # Skip to the next iteration if no response
                
                    formatted_output = format_qa_pairs(result)
                    if not formatted_output:
                        print(f"No formatted output generated for chunk {i+1} in iteration {iteration+1}.")
                        continue
                
                    save_to_csv(formatted_output)
                    print(f"Batch of data appended to dataset for chunk {i+1} in iteration {iteration+1}.")

# Call the function to start the process
generate_dataset()

# Load the training examples from the file
training_examples = load_training_examples()

##### Fine-tune the model
output_csv_file = 'generated_answers.csv'

# Define the prompt template
prompt_no_input_template = """<s>[INST] <<SYS>>
You are a computer networking expert specialized in Cisco IOS XE running configurations.
<</SYS>>

%s [/INST]
"""

# Load the model
model_name_or_path = "meta-llama/Meta-Llama-3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, 
    padding_side="right", use_fast=False)

# Set pad_token as eos_token
tokenizer.pad_token = tokenizer.eos_token

# Set up the ReFT config
reft_config = pyreft.ReftConfig(representations={
    "layer": 15, "component": "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
    low_rank_dimension=4)})
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device(device)
reft_model.print_trainable_parameters()

# Create data module
data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, model, [prompt_no_input_template % e[0] for e in training_examples], 
    [e[1] for e in training_examples])

# Define training arguments
training_args = transformers.TrainingArguments(
    num_train_epochs=100.0, output_dir="./tmp", per_device_train_batch_size=10, 
    learning_rate=4e-3, logging_steps=20)

# Train the model
trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model, tokenizer=tokenizer, args=training_args, **data_module)
_ = trainer.train()

def load_questions_from_csv(filename='output.csv'):
    questions = []
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            row_count = 0  # Debug: count the rows processed
            for row in reader:
                questions.append(row[0])  # Assuming the question is in the first column
                row_count += 1
            print(f"Total questions loaded: {row_count}")  # Debug: print total questions loaded
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return questions

# Load questions for inference testing
questions = load_questions_from_csv()

# Open CSV file for writing the results
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Question', 'Fine-tuned Answer', 'Base Model Answer'])

    for question in questions:
        instruction = question
        # Tokenize and prepare the input
        prompt = prompt_no_input_template % instruction
        prompt = tokenizer(prompt, return_tensors="pt").to(device)

        base_unit_location = prompt["input_ids"].shape[-1] - 1  # Last position
        _, reft_response = reft_model.generate(
            prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
            intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
            eos_token_id=tokenizer.eos_token_id, early_stopping=True
        )
        
        fine_tuned_answer = tokenizer.decode(reft_response[0], skip_special_tokens=True)
        
        base_model_response = send_control_request(instruction)
        
        writer.writerow([question, fine_tuned_answer, base_model_response])
        print(f"Question: {question}")
        print(f"Fine-tuned Answer: {fine_tuned_answer}")
        print(f"Base Model Answer: {base_model_response}\n")

print("Interactive mode. Type your questions or 'exit' to quit.")

while True:
    instruction = input("You: ")
    
    if instruction.lower() == "exit":
        print("Goodbye!")
        break

    # Tokenize and prepare the input
    prompt = prompt_no_input_template % instruction
    prompt = tokenizer(prompt, return_tensors="pt").to(device)
    
    base_unit_location = prompt["input_ids"].shape[-1] - 1  # Last position
    _, reft_response = reft_model.generate(
        prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
        intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
        eos_token_id=tokenizer.eos_token_id, early_stopping=True
    )
    
    fine_tuned_answer = tokenizer.decode(reft_response[0], skip_special_tokens=True)
    
    base_model_response = send_control_request(instruction)
    
    print(f"Question: {instruction}")
    print(f"Fine-tuned Answer: {fine_tuned_answer}")
    print(f"Base Model Answer: {base_model_response}\n")

# Save and publish model
reft_model.set_device("cpu")  # Send back to CPU before saving.
reft_model.save(
    save_directory="./CiscoDevNetSandboxRunningConfig", 
    save_to_hf_hub=True, 
    hf_repo_name="automateyournetwork/Cisco_DevNet_Sandbox_Running_Config"
)
