# InterCode-ALFA

## Description
A fork of the InterCode benchmark used to evaluate natural language to Bash command translation.  
[HuggingFace Dataset](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA)  
[PyPI Package](https://pypi.org/project/icalfa/)  

![InterCode-ALFA Diagram](https://raw.githubusercontent.com/westenfelder/InterCode-ALFA/main/icalfa.png)


## Installation
- Install Docker Engine - [Instructions](https://docs.docker.com/engine/install/)
- Configure Docker for non-sudo users - [Instructions](https://docs.docker.com/engine/install/linux-postinstall/)
- Create a python virtual environment
```bash
apt install python3.12-venv
python3 -m venv icalfa-venv
source icalfa-venv/bin/activate
```
- Install InterCode-ALFA
```bash
pip install icalfa datasets tqdm
```
- [Optional] If you want to use a local LLM, install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:70b
```
- [Optional] If you want to use the embedding comparison method, install mxbai-embed-large
```bash
ollama pull mxbai-embed-large
```


## Usage
- Run the benchmark
```python
import os
from icalfa import submit_command
from datasets import load_dataset
from tqdm import tqdm

# Store OpenAI key as environment variable 
os.environ['ICALFA_OPENAI_API_KEY'] = '...'

# Load dataset
dataset = load_dataset("westenfelder/NL2SH-ALFA", "test", split="train")

# Iterate through the dataset
score = 0
for index, row in tqdm(enumerate(dataset), total=len(dataset)):

    # Retrieve natural language prompt
    prompt = row['nl']

    # Convert natural language prompt to Bash command here

    # Submit Bash command for benchmark scoring. 0 = incorrect, 1 = correct
    score += submit_command(index=index, command="...")

    # Retrieve ground truth commands
    ground_truth_command = row['bash']
    ground_truth_command2 = row['bash2']

# Print the benchmark result
print(score/len(dataset))
```

- `submit_command` parameters
```python
# By default icalfa uses OpenAI's GPT-4 model and expects an API key
submit_command(index, command, eval_mode="openai", eval_param="gpt-4-0613")

# A local model can be used via Ollama and does not require an API key
submit_command(index, command, eval_mode="ollama", eval_param="llama3.1:70b")

# You can also test the original method used in Princeton's InterCode benchmark
submit_command(index, command, eval_mode="tfidf")

# An embedding based comparison method is also available
# This uses the mxbai-embed-large model via Ollama, with the eval_param specifying the similarity threshold
submit_command(index, command, eval_mode="embed", eval_param=0.75)
```

- Manage Docker containers
```bash
# Stop containers
docker stop $(docker ps -a --filter "name=intercode*" -q)

# Delete containers
docker rm $(docker ps -a --filter "name=intercode*" -q)
```


## Building
```bash
# pip install build twine
# update version in pyproject.toml and __init__.py
rm -rf dist
python3 -m build
python3 -m twine upload --repository pypi dist/*
pip install --upgrade icalfa
```


## Credits
InterCode-ALFA is a fork of the InterCode benchmark developed by the Princeton NLP group.  
[InterCode Website](https://intercode-benchmark.github.io/)  
[InterCode PyPI Package](https://pypi.org/project/intercode-bench/#description)  
