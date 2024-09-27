# InterCode-ALFA

## Description
A fork of the InterCode benchmark used to evaluate natural language to Bash command translation.  
[Dataset](https://huggingface.co/datasets/westenfelder/InterCode-ALFA-Data)  
[PyPI Package](https://pypi.org/project/icalfa/)  

![InterCode-ALFA Diagram](https://raw.githubusercontent.com/westenfelder/InterCode-ALFA/main/icalfa.png)


## Installation
- Install Docker Engine [Instructions](https://docs.docker.com/engine/install/)
- Configure Docker for non-sudo users [Instructions](https://docs.docker.com/engine/install/linux-postinstall/)
- Create python virtual environment
```bash
apt install python3.12-venv
python3 -m venv icalfa-venv
source icalfa-venv/bin/activate
```
- Install InterCode-ALFA
```bash
pip install icalfa
```


## Usage
- Run the benchmark
```python
import os
from icalfa import submit_command
from datasets import load_dataset

# Store OpenAI key as environment variable 
os.environ['ICALFA_OPENAI_API_KEY'] = '...'

# Load dataset
dataset = load_dataset("westenfelder/InterCode-ALFA-Data")['train']

# Iterate through the dataset
score = 0
for index, row in enumerate(dataset):

    # Retrieve natural language prompt
    prompt = row['query']

    # Convert natural language prompt to Bash command here

    # Submit Bash command for benchmark scoring. 0 = incorrect, 1 = correct
    score += submit_command(index=index, command="...")

    # Retrieve ground truth commands
    ground_truth_command = row['gold']
    ground_truth_command2 = row['gold2']

# Print the benchmark result
print(score/len(dataset))
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
