# CiscoDevNetIOSXESandbox_ReFT
The code used to build the large language model for the Cisco DevNet IOS XE Always-On Sandbox using the inspiration from the Stanford NLP Team 

https://github.com/stanfordnlp/pyreft

https://nlp.stanford.edu/~wuzhengx/reft/index.html


## How to make your own model with the running-config from the Cisco DevNet IOS XE Sandbox
Start Ollama (ollama.com) locally with Llama3 installed (ollama run llama3)

Make sure the DevNet Sandbox is available 
```console
$ ssh admin@devnetsandboxiosxe.cisco.com
```
With credentials from testbed.yaml file; rename the sandbox to Cat8000V if you have to)

Setup you virtual environment (see requirements.txt)

```console
$ python3 -m venv myvenv
```

## Setup your environment variables
```console
(myvenv) $ export HF_TOKEN=<your write token here>
(myvenv) $ export WANDB_API_KEY<your wandb api key here>
```

Run the fine_tune_model.py file
```console
(myvenv)$ python fine_tune_model.py
```

