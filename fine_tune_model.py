import os
import csv
import torch
import requests
import transformers
import pyreft

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

# Load the running configuration from file
def load_config(file_path='show_run.txt'):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "Configuration file not found."

# Function to send requests to the models
def send_request(model, running_config):
    url = f"http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    prompt = f"Using the following Cisco IOS XE running configuration, generate as many questions and corresponding correct answers as possible. No additional text, indices, or labels. Do not include headings or any other information. Pay close attention to interfaces, sub-interfaces, the number of sub-interfaces, IP addresses, routes, OSPF, ACLs, VRFs, and general configuration settings like NTP, VTY, banner, hostname, and domain. Use these areas of the running configuration provided as your source for the questions and correct answers. Take your time.:\n\n{running_config}"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get('response', '')
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def save_to_csv(questions_answers, filename="output.csv", mode='a'):  # default to append mode
    with open(filename, mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        if file.tell() == 0:  # If file is empty, write header
            writer.writerow(["Question", "Answer"])
        for qa in questions_answers:
            if ',' in qa:
                question, answer = qa.split(',', 1)  # Split only on the first comma
                writer.writerow([question, answer])

def count_csv_entries(filename="output.csv"):
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            return sum(1 for row in reader) - 1  # Subtract one to exclude the header
    except FileNotFoundError:
        return 0  # If file not found, we have 0 entries

def load_training_examples(filename='output.csv'):
    training_examples = []
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                # Append each row as a list within the larger list
                training_examples.append([row[0], row[1]])  # Assuming row format is correct
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
            question = line.split(":", 1)[1].strip()  # Split on the first colon and take the second part as the question
        elif line.startswith("A") and question:
            answer = line.split(":", 1)[1].strip()  # Split on the first colon and take the second part as the answer
            formatted_pairs.append(f"{question},{answer}")
            question = None  # Reset question for the next pair
    return formatted_pairs

# Example of using these functions in a workflow
def generate_dataset():
    # print(run_pyats_job())
    for _ in range(500):  # Loop up to 10 times
        running_config = load_config()
        if running_config == "Configuration file not found.":
            print(running_config)
            return
        
        model_response = send_request('llama3', running_config)
        if not model_response:
            print("No response or an error occurred while fetching the model response.")
            continue  # Skip to the next iteration if no response

        formatted_output = format_qa_pairs(model_response)
        if not formatted_output:
            print("No formatted output generated.")
            continue
        
        save_to_csv(formatted_output)
        print("Batch of data appended to dataset.")
        
        # Check the total number of entries
        if count_csv_entries() >= 1000:
            print("Reached 1000 questions and answers. Stopping the process.")
            break

# # Call the function to start the process
#generate_dataset()

# Load the training examples from the file
training_examples = load_training_examples()

#####Fine tune model
output_csv_file = 'generated_answers.csv'

# # Define the prompt template
prompt_no_input_template = """<s>[INST] <<SYS>>
You are an AI expert specialized in Cisco IOS XE running configurations. Your role is to provide accurate, clear, and concise explanations or solutions directly relevant to the configurations and queries presented. Ensure your responses reflect deep knowledge and practical applicability.
<</SYS>>

%s [/INST]
"""

# # Load the model
model_name_or_path = "meta-llama/Meta-Llama-3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, 
    padding_side="right", use_fast=False)

# # Set pad_token as eos_token
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

# # Create data module
data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, model, [prompt_no_input_template % e[0] for e in training_examples], 
    [e[1] for e in training_examples])

# # Define training arguments
training_args = transformers.TrainingArguments(
    num_train_epochs=100.0, output_dir="./tmp", per_device_train_batch_size=10, 
    learning_rate=4e-3, logging_steps=20)

# # Train the model
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
    writer.writerow(['Question', 'Answer'])

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
        
        answer = tokenizer.decode(reft_response[0], skip_special_tokens=True)
        writer.writerow([question, answer])
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")

# #Save and publish model
reft_model.set_device("cpu")  # Send back to CPU before saving.
reft_model.save(
    save_directory="./CiscoDevNetSandboxRunningConfig", 
    save_to_hf_hub=True, 
    hf_repo_name="automateyournetwork/Cisco_DevNet_Sandbox_Running_Config"
)