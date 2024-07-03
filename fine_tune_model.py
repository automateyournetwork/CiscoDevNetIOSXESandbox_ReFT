import os
import csv
import json
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
    job_status = os.system("pyats run job pyats_job.py")
    return "Job executed successfully" if job_status == 0 else "Job failed"

def load_config(file_path='show_run.txt'):
    try:
        with open(file_path, 'r') as file:
            return file.read().split('!')
    except FileNotFoundError:
        return "Configuration file not found."

def load_show_ip_interface_brief_json(file_path='show_ip_interface_brief.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file).get("interface", {})
    except FileNotFoundError:
        return "JSON configuration file not found."

def load_show_interfaces_json(file_path='show_interfaces.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return "JSON configuration file not found."

def load_show_access_lists_json(file_path='show_access_lists.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return "JSON configuration file not found."

def load_show_ip_route_json(file_path='show_ip_route.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return "JSON configuration file not found."

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

def send_show_ip_interface_brief_json_request(model, interface_name, interface_data):
    url = f"http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    prompt = (
        f"The following JSON data is from a Cisco IOS XE interface configuration. Generate randomized questions about the data to form a dataset being used to fine-tune a model. "
        f"Only generate questions if the answer can be directly found within the JSON data provided. Do not generate questions if the answer is not explicitly present in the data. "
        f"Include specific details from the JSON in the questions to make them clear and contextually accurate.\n\n"
        f"For example, if you see a configuration like this:\n"
        f"\"GigabitEthernet1\": {{\n"
        f"  \"ip_address\": \"10.10.20.48\",\n"
        f"  \"interface_is_ok\": \"YES\",\n"
        f"  \"method\": \"NVRAM\",\n"
        f"  \"status\": \"up\",\n"
        f"  \"protocol\": \"up\"\n"
        f"}},\n"
        f"Make sure your questions are like this:\n"
        f"Q: What is the IP address of GigabitEthernet1?\n"
        f"A: 10.10.20.48\n"
        f"Q: Is the interface GigabitEthernet1 ok?\n"
        f"A: YES\n"
        f"Q: What is the status of GigabitEthernet1?\n"
        f"A: up\n"
        f"Q: What is the method for GigabitEthernet1?\n"
        f"A: NVRAM\n"
        f"Q: What is the protocol status of Loopback0?\n"
        f"A: up\n"
        f"Q: What is the IP address of Loopback0?\n"
        f"A: 10.0.0.1\n\n"
        f"Now generate questions and answers for the following JSON data:\n"
        f"---JSON Configuration Data---\n{json.dumps({interface_name: interface_data}, indent=2)}\n"
        f"Format each question with a 'Q: ' prefix and each answer with an 'A: ' prefix on the next line. Only include questions that can be answered based on the given data. Do not include questions without answers.\n"
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

def send_show_interfaces_json_request(model, interface_name, interface_data):
    url = f"http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    prompt = (
        f"The following JSON data is from a Cisco IOS XE interface configuration. Generate randomized questions about the data to form a dataset being used to fine-tune a model. "
        f"Only generate questions if the answer can be directly found within the JSON data provided. Do not generate questions if the answer is not explicitly present in the data. "
        f"Include specific details from the JSON in the questions to make them clear and contextually accurate.\n\n"
        f"For example, if you see a configuration like this:\n"
        f"\"GigabitEthernet1\": {{\n"
        f"  \"port_channel\": {{\n"
        f"    \"port_channel_member\": false\n"
        f"  }},\n"
        f"  \"is_deleted\": false,\n"
        f"  \"enabled\": true,\n"
        f"  \"line_protocol\": \"up\",\n"
        f"  \"oper_status\": \"up\",\n"
        f"  \"type\": \"vNIC\",\n"
        f"  \"mac_address\": \"0050.56bf.bfe7\",\n"
        f"  \"phys_address\": \"0050.56bf.bfe7\",\n"
        f"  \"description\": \"MANAGEMENT INTERFACE - DON'T TOUCH ME\",\n"
        f"  \"ipv4\": {{\n"
        f"    \"10.10.20.48/24\": {{\n"
        f"      \"ip\": \"10.10.20.48\",\n"
        f"      \"prefix_length\": \"24\"\n"
        f"    }}\n"
        f"  }},\n"
        f"  \"delay\": 10,\n"
        f"  \"mtu\": 1500,\n"
        f"  \"bandwidth\": 1000000,\n"
        f"  \"reliability\": \"255/255\",\n"
        f"  \"txload\": \"1/255\",\n"
        f"  \"rxload\": \"1/255\",\n"
        f"  \"encapsulations\": {{\n"
        f"    \"encapsulation\": \"arpa\"\n"
        f"  }},\n"
        f"  \"keepalive\": 10,\n"
        f"  \"duplex_mode\": \"full\",\n"
        f"  \"port_speed\": \"1000mbps\",\n"
        f"  \"link_type\": \"auto\",\n"
        f"  \"auto_negotiate\": true,\n"
        f"  \"media_type\": \"Virtual\",\n"
        f"  \"flow_control\": {{\n"
        f"    \"receive\": false,\n"
        f"    \"send\": false\n"
        f"  }},\n"
        f"  \"arp_type\": \"arpa\",\n"
        f"  \"arp_timeout\": \"04:00:00\",\n"
        f"  \"last_input\": \"00:00:00\",\n"
        f"  \"last_output\": \"00:00:00\",\n"
        f"  \"output_hang\": \"never\",\n"
        f"  \"queues\": {{\n"
        f"    \"input_queue_size\": 0,\n"
        f"    \"input_queue_max\": 375,\n"
        f"    \"input_queue_drops\": 0,\n"
        f"    \"input_queue_flushes\": 0,\n"
        f"    \"total_output_drop\": 0,\n"
        f"    \"queue_strategy\": \"fifo\",\n"
        f"    \"output_queue_size\": 0,\n"
        f"    \"output_queue_max\": 40\n"
        f"  }},\n"
        f"  \"counters\": {{\n"
        f"    \"rate\": {{\n"
        f"      \"load_interval\": 300,\n"
        f"      \"in_rate\": 4000,\n"
        f"      \"in_rate_pkts\": 3,\n"
        f"      \"out_rate\": 3000,\n"
        f"      \"out_rate_pkts\": 3\n"
        f"    }},\n"
        f"    \"last_clear\": \"never\",\n"
        f"    \"in_pkts\": 1879,\n"
        f"    \"in_octets\": 249491,\n"
        f"    \"in_no_buffer\": 0,\n"
        f"    \"in_multicast_pkts\": 0,\n"
        f"    \"in_broadcast_pkts\": 0,\n"
        f"    \"in_runts\": 0,\n"
        f"    \"in_giants\": 0,\n"
        f"    \"in_throttles\": 0,\n"
        f"    \"in_errors\": 0,\n"
        f"    \"in_crc_errors\": 0,\n"
        f"    \"in_frame\": 0,\n"
        f"    \"in_overrun\": 0,\n"
        f"    \"in_ignored\": 0,\n"
        f"    \"in_watchdog\": 0,\n"
        f"    \"in_mac_pause_frames\": 0,\n"
        f"    \"out_pkts\": 1980,\n"
        f"    \"out_octets\": 693666,\n"
        f"    \"out_underruns\": 0,\n"
        f"    \"out_broadcast_pkts\": 0,\n"
        f"    \"out_multicast_pkts\": 0,\n"
        f"    \"out_errors\": 0,\n"
        f"    \"out_interface_resets\": 0,\n"
        f"    \"out_collision\": 0,\n"
        f"    \"out_unknown_protocl_drops\": 0,\n"
        f"    \"out_babble\": 0,\n"
        f"    \"out_late_collision\": 0,\n"
        f"    \"out_deferred\": 0,\n"
        f"    \"out_lost_carrier\": 0,\n"
        f"    \"out_no_carrier\": 0,\n"
        f"    \"out_mac_pause_frames\": 0,\n"
        f"    \"out_buffer_failure\": 0,\n"
        f"    \"out_buffers_swapped\": 0\n"
        f"  }}\n"
        f"}},\n"
        f"Make sure your questions are like this:\n"
        f"Q: What is the IP address of GigabitEthernet1?\n"
        f"A: 10.10.20.48\n"
        f"Q: Is the interface GigabitEthernet1 ok?\n"
        f"A: YES\n"
        f"Q: What is the status of GigabitEthernet1?\n"
        f"A: up\n"
        f"Q: What is the method for GigabitEthernet1?\n"
        f"A: NVRAM\n"
        f"Q: What is the MAC address of GigabitEthernet1?\n"
        f"A: 0050.56bf.bfe7\n"
        f"Q: What is the MTU of GigabitEthernet1?\n"
        f"A: 1500\n"
        f"Q: What is the bandwidth of GigabitEthernet1?\n"
        f"A: 1000000\n"
        f"Q: What is the duplex mode of GigabitEthernet1?\n"
        f"A: full\n"
        f"Q: What is the port speed of GigabitEthernet1?\n"
        f"A: 1000mbps\n"
        f"Q: What is the link type of GigabitEthernet1?\n"
        f"A: auto\n"
        f"Q: What is the ARP timeout for GigabitEthernet1?\n"
        f"A: 04:00:00\n\n"
        f"Now generate questions and answers for the following JSON data:\n"
        f"---JSON Configuration Data---\n{json.dumps({interface_name: interface_data}, indent=2)}\n"
        f"Format each question with a 'Q: ' prefix and each answer with an 'A: ' prefix on the next line. Only include questions that can be answered based on the given data. Do not include questions without answers.\n"
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

def send_show_access_lists_json_request(model, acl_name, acl_data):
    url = f"http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    prompt = (
        f"The following JSON data is from a Cisco IOS XE access list configuration. Generate randomized questions about the data to form a dataset being used to fine-tune a model. "
        f"Only generate questions if the answer can be directly found within the JSON data provided. Do not generate questions if the answer is not explicitly present in the data. "
        f"Include specific details from the JSON in the questions to make them clear and contextually accurate.\n\n"
        f"For example, if you see a configuration like this:\n"
        f"\"NAT-ACL\": {{\n"
        f"  \"name\": \"NAT-ACL\",\n"
        f"  \"type\": \"ipv4-acl-type\",\n"
        f"  \"acl_type\": \"extended\",\n"
        f"  \"aces\": {{\n"
        f"    \"10\": {{\n"
        f"      \"name\": \"10\",\n"
        f"      \"actions\": {{\n"
        f"        \"forwarding\": \"permit\",\n"
        f"        \"logging\": \"log-none\"\n"
        f"      }},\n"
        f"      \"matches\": {{\n"
        f"        \"l3\": {{\n"
        f"          \"ipv4\": {{\n"
        f"            \"protocol\": \"ipv4\",\n"
        f"            \"source_network\": {{\n"
        f"              \"192.168.1.0 0.0.0.255\": {{\n"
        f"                \"source_network\": \"192.168.1.0 0.0.0.255\"\n"
        f"              }}\n"
        f"            }},\n"
        f"            \"destination_network\": {{\n"
        f"              \"any\": {{\n"
        f"                \"destination_network\": \"any\"\n"
        f"              }}\n"
        f"            }}\n"
        f"          }}\n"
        f"        }},\n"
        f"        \"l4\": {{\n"
        f"          \"ipv4\": {{\n"
        f"            \"established\": false\n"
        f"          }}\n"
        f"        }}\n"
        f"      }}\n"
        f"    }}\n"
        f"  }}\n"
        f"}},\n"
        f"Make sure your questions are like this:\n"
        f"Q: What is the name of the access list?\n"
        f"A: NAT-ACL\n"
        f"Q: What is the type of the access list?\n"
        f"A: ipv4-acl-type\n"
        f"Q: What is the ACL type of NAT-ACL?\n"
        f"A: extended\n"
        f"Q: What is the forwarding action for ACE 10 in NAT-ACL?\n"
        f"A: permit\n"
        f"Q: What is the logging action for ACE 10 in NAT-ACL?\n"
        f"A: log-none\n"
        f"Q: What is the protocol used in ACE 10 of NAT-ACL?\n"
        f"A: ipv4\n"
        f"Q: What is the source network in ACE 10 of NAT-ACL?\n"
        f"A: 192.168.1.0 0.0.0.255\n"
        f"Q: What is the destination network in ACE 10 of NAT-ACL?\n"
        f"A: any\n"
        f"Q: Is the IPv4 protocol established in ACE 10 of NAT-ACL?\n"
        f"A: false\n\n"
        f"Now generate questions and answers for the following JSON data:\n"
        f"---JSON Configuration Data---\n{json.dumps({acl_name: acl_data}, indent=2)}\n"
        f"Format each question with a 'Q: ' prefix and each answer with an 'A: ' prefix on the next line. Only include questions that can be answered based on the given data. Do not include questions without answers.\n"
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

def send_show_ip_route_json_request(model, route_name, route_data):
    url = f"http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    prompt = (
        f"The following JSON data is from a Cisco IOS XE IP route configuration. Generate randomized questions about the data to form a dataset being used to fine-tune a model. "
        f"Only generate questions if the answer can be directly found within the JSON data provided. Do not generate questions if the answer is not explicitly present in the data. "
        f"Include specific details from the JSON in the questions to make them clear and contextually accurate.\n\n"
        f"For example, if you see a configuration like this:\n"
        f"\"0.0.0.0/0\": {{\n"
        f"  \"route\": \"0.0.0.0/0\",\n"
        f"  \"active\": true,\n"
        f"  \"metric\": 0,\n"
        f"  \"route_preference\": 1,\n"
        f"  \"source_protocol_codes\": \"S*\",\n"
        f"  \"source_protocol\": \"static\",\n"
        f"  \"next_hop\": {{\n"
        f"    \"next_hop_list\": {{\n"
        f"      \"1\": {{\n"
        f"        \"index\": 1,\n"
        f"        \"next_hop\": \"10.10.20.254\",\n"
        f"        \"outgoing_interface\": \"GigabitEthernet1\"\n"
        f"      }}\n"
        f"    }}\n"
        f"  }}\n"
        f"}},\n"
        f"Make sure your questions are like this:\n"
        f"Q: What is the route for 0.0.0.0/0?\n"
        f"A: 0.0.0.0/0\n"
        f"Q: Is the route 0.0.0.0/0 active?\n"
        f"A: true\n"
        f"Q: What is the metric for the route 0.0.0.0/0?\n"
        f"A: 0\n"
        f"Q: What is the route preference for 0.0.0.0/0?\n"
        f"A: 1\n"
        f"Q: What are the source protocol codes for 0.0.0.0/0?\n"
        f"A: S*\n"
        f"Q: What is the source protocol for 0.0.0.0/0?\n"
        f"A: static\n"
        f"Q: What is the next hop IP address for 0.0.0.0/0?\n"
        f"A: 10.10.20.254\n"
        f"Q: What is the outgoing interface for the next hop 10.10.20.254 in 0.0.0.0/0?\n"
        f"A: GigabitEthernet1\n\n"
        f"If the route is 0.0.0.0/0, treat it as the default route and generate questions like:\n"
        f"Q: What is my default route?\n"
        f"A: 0.0.0.0/0\n"
        f"Q: What is my default route next hop?\n"
        f"A: 10.10.20.254\n"
        f"Q: What outgoing interface does my default route use?\n"
        f"A: GigabitEthernet1\n\n"
        f"Now generate questions and answers for the following JSON data:\n"
        f"---JSON Configuration Data---\n{json.dumps({route_name: route_data}, indent=2)}\n"
        f"Format each question with a 'Q: ' prefix and each answer with an 'A: ' prefix on the next line. Only include questions that can be answered based on the given data. Do not include questions without answers.\n"
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
    
    show_ip_interface_brief_json = load_show_ip_interface_brief_json()
    if show_ip_interface_brief_json == "JSON configuration file not found.":
        print(show_ip_interface_brief_json)
        return
    
    show_interfaces_json = load_show_interfaces_json()
    if show_interfaces_json == "JSON configuration file not found.":
        print(show_interfaces_json)
        return
        
    show_access_lists_json = load_show_access_lists_json()
    if show_access_lists_json == "JSON configuration file not found.":
        print(show_access_lists_json)
        return

    show_ip_route_json = load_show_ip_route_json()
    if show_ip_route_json == "JSON configuration file not found.":
        print(show_ip_route_json)
        return
    
    models = ["llama3", "gemma2"]
    for iteration in range(num_iterations):

        # Process show ip interface brief
        for interface_name, interface_data in show_ip_interface_brief_json.items():
            if interface_data:  # Skip empty interfaces
                for model in models:
                    result = send_show_ip_interface_brief_json_request(model, interface_name, interface_data)
                    if not result:
                        print(f"No response or an error occurred while fetching the model response for interface {interface_name} in iteration {iteration+1}.")
                        continue  # Skip to the next iteration if no response
            
                    formatted_output = format_qa_pairs(result)
                    if not formatted_output:
                        print(f"No formatted output generated for interface {interface_name} in iteration {iteration+1}.")
                        continue
            
                    save_to_csv(formatted_output)
                    print(f"Batch of data appended to dataset for interface {interface_name} in iteration {iteration+1}.")

        # Process show interfaces
        for interface_name, interface_data in show_interfaces_json.items():
            if interface_data:  # Skip empty interfaces
                for model in models:
                    result = send_show_interfaces_json_request(model, interface_name, interface_data)
                    if not result:
                        print(f"No response or an error occurred while fetching the model response for interface {interface_name} in iteration {iteration+1}.")
                        continue  # Skip to the next iteration if no response

                    formatted_output = format_qa_pairs(result)
                    if not formatted_output:
                        print(f"No formatted output generated for interface {interface_name} in iteration {iteration+1}.")
                        continue

                    save_to_csv(formatted_output)
                    print(f"Batch of data appended to dataset for interface {interface_name} in iteration {iteration+1}.")

        #Process ACLs
        for acl_name, acl_data in show_access_lists_json.items():
            if acl_data:  # Skip empty ACLs
                for model in models:
                    result = send_show_access_lists_json_request(model, acl_name, acl_data)
                    if not result:
                        print(f"No response or an error occurred while fetching the model response for ACL {acl_name} in iteration {iteration+1}.")
                        continue  # Skip to the next iteration if no response
                
                    formatted_output = format_qa_pairs(result)
                    if not formatted_output:
                        print(f"No formatted output generated for ACL {acl_name} in iteration {iteration+1}.")
                        continue
                
                    save_to_csv(formatted_output)
                    print(f"Batch of data appended to dataset for ACL {acl_name} in iteration {iteration+1}.")

        #Process IP Routes
        for vrf, vrf_data in show_ip_route_json["vrf"].items():
            for af, af_data in vrf_data["address_family"].items():
                for route_name, route_data in af_data["routes"].items():
                    if route_data:  # Skip empty routes
                        for model in models:
                            result = send_show_ip_route_json_request(model, route_name, route_data)
                            if not result:
                                print(f"No response or an error occurred while fetching the model response for route {route_name} in iteration {iteration+1}.")
                                continue  # Skip to the next iteration if no response
                            
                            formatted_output = format_qa_pairs(result)
                            if not formatted_output:
                                print(f"No formatted output generated for route {route_name} in iteration {iteration+1}.")
                                continue
                            
                            save_to_csv(formatted_output)
                            print(f"Batch of data appended to dataset for route {route_name} in iteration {iteration+1}.")

        #Process running config chunks
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
run_pyats_job()
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
