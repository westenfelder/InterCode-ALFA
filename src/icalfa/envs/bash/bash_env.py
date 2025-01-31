import math
import json
import requests
from openai import OpenAI
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple

from icalfa.envs.ic_env import (
    IntercodeEnv,
    AGENT_OBS, EVAL_OBS, CORRUPT_GOLD, ACTION_EXEC, REWARD
)
from icalfa.utils import get_container, timeout

GIT_RESET_SCRIPT = "git reset --hard; git clean -fd;"
GIT_STATUS_SCRIPT = "git status --short;"

IMAGE_TO_SETTINGS = {
    "intercode-bash-1": "/bin/bash",
    "intercode-bash-2": "/bin/bash",
    "intercode-bash-3": "/bin/bash",
    "intercode-bash-4": "/bin/bash",
    "intercode-bash-5": "/bin/sh"
}

class BashEnv(IntercodeEnv):
    """Gym environment for bash shell"""
    name = "ic_bash"

    def __init__(self, image_name: str, **kwargs):
        super(BashEnv, self).__init__(image_name, **kwargs)

        # Establish connection with evaluation container
        self.ctr_name_eval = f"{self.image_name}_ic_ctr_eval"
        self.ctr_name_agent = f"{self.image_name}_ic_ctr"
        self.container_eval = get_container(self.ctr_name_eval, self.image_name)
        self.container_agent = get_container(self.ctr_name_agent, self.image_name)
    
    def reset_container(self) -> None:
        self.workdir = "/"
        exit_code, output = self.container_eval.exec_run(self.clean_cmd(GIT_RESET_SCRIPT))
        if exit_code != 0:
            raise RuntimeError(f"Failed to reset `{self.ctr_name_eval}` container successfully: {output}")
        exit_code, output = self.container_agent.exec_run(self.clean_cmd(GIT_RESET_SCRIPT))
        if exit_code != 0:
            raise RuntimeError(f"Failed to reset `{self.ctr_name_agent}` container successfully: {output}")
    
    def exec_action(self, action: str) -> None:
        """Executes action in bash shell"""
        is_cd_flag = action.startswith("cd")
        if is_cd_flag:
            cd_arg = action[action.index("cd ")+3:].strip()
            new_path = self.simplify_path(self.workdir, cd_arg)
            action = f"cd {new_path}"
        
        try:
            with timeout():
                exit_code, output = self.container.exec_run(
                    self.clean_cmd(action),
                    workdir="/" if is_cd_flag else self.workdir)
                self.observation = output.decode("utf-8")
                self.info[ACTION_EXEC] = exit_code == 0
        except TimeoutError:
            self.observation = f"Command timed out"
            self.info[ACTION_EXEC] = False
        except Exception as e:
            self.observation = f"Exception: {e}"
            self.info[ACTION_EXEC] = False
        
        if is_cd_flag and self.info[ACTION_EXEC]:
            self.workdir = new_path
            
    def get_reward(self, prompt, trajectory, eval_mode, eval_param) -> Tuple[float, Dict]:
        """
        The reward currently is calculated as a weighted sum of the following:
        - 0.33: (File System Diff) Difference in file system states between agent, gold command
        - 0.33: (File Content) Verify each file was correctly changed by agent using hashing
        - 0.33: (Observation) Verify that correct output was generated
        """
        # Reset evaluation container state
        exit_code, output = self.container_eval.exec_run(self.clean_cmd(GIT_RESET_SCRIPT))
        if exit_code != 0:
            raise RuntimeError(f"Failed to reset `{self.ctr_name_eval}` container successfully: {output}")
        
        # Run gold command(s) in evaluation container
        self.observation_eval = None
        try:
            if isinstance(self.gold, str):
                self.observation_eval = self.container_eval.exec_run(
                    self.clean_cmd(self.gold)).output.decode("utf-8")
            elif isinstance(self.gold, List):
                self.observation_eval = self.container_eval.exec_run(
                self.clean_cmd(";".join(self.gold))).output.decode("utf-8")
            self.info[CORRUPT_GOLD] = False
        except Exception as e:
            self.info[CORRUPT_GOLD] = True

        # Calculate Rewards
        reward, info = 0.01, {}
        info[REWARD] = {}

        # PART 1: Compare file system states
        diff_agent = self.parse_status(self.container.exec_run(self.clean_cmd(GIT_STATUS_SCRIPT)).output.decode("utf-8"))
        diff_eval = self.parse_status(self.container_eval.exec_run(self.clean_cmd(GIT_STATUS_SCRIPT)).output.decode("utf-8"))
        info["diff_miss"] = list(set(diff_eval) - set(diff_agent))
        info["diff_extra"] = list(set(diff_agent) - set(diff_eval))
        p1_score = round(0.33 * (1 - math.erf(len(info["diff_miss"]) + len(info["diff_extra"]))), 2)
        info[REWARD]["file_diff"] = p1_score
        reward += p1_score

        # PART 2: Check if files changed by both agent, gold commands were modified correctly
        p2_score = 0.33
        # Only check corrects of common changes that were added or modified
        filter_changes = lambda x: (x[1] in ["A", "??", "C"])
        diff_same = [x for x in list(set(diff_agent) & set(diff_eval)) if filter_changes(x)]
        
        if len(diff_same) > 0:
            same_changes = 0
            # Compute hashes for files and folders differently using md5 checksums
            get_hash_cmd = lambda x: f"md5sum {x}" if "." in x else f"md5deep -r {x}"

            for path in diff_same:
                hash_cmd = get_hash_cmd(path[0])
                agent_hash = self.container.exec_run(hash_cmd).output.decode("utf-8")
                # print(agent_hash)
                gold_hash = self.container_eval.exec_run(hash_cmd).output.decode("utf-8")
                # print(gold_hash)
                same_changes += 1 if agent_hash == gold_hash else 0
            
            info["diff_same"] = {"files": diff_same, "correct": same_changes, "total": len(diff_same)}
            p2_score = round(0.33 * (same_changes / len(diff_same)), 2)
        info[REWARD]["file_changes"] = p2_score
        reward += p2_score
        
        # PART 3: Compare agent, query answers
        info[AGENT_OBS] = self.observation
        info[EVAL_OBS] = self.observation_eval

        # Variables for LLM eval
        gold_command = self.gold
        model_command = trajectory[0][0]
        gold_command_output = self.observation_eval
        model_command_output = self.observation

        p3_score = 0

        if gold_command == model_command:
            p3_score = 0.33
        elif gold_command_output == model_command_output:
           p3_score = 0.33
        else: 
            # Method used in original InterCode paper
            if eval_mode == "tfidf":
                try:
                    vect = TfidfVectorizer()
                    tfidf = vect.fit_transform([info[AGENT_OBS], info[EVAL_OBS]])
                    answer_similarity = tfidf * tfidf.T
                    info["answer_similarity"] = answer_similarity.toarray()[0][1]
                except:
                    info["answer_similarity"] = 1 if info[AGENT_OBS] == info[EVAL_OBS] else 0
                p3_score = round(0.33 * info["answer_similarity"], 2)

            # Embedding-based command output comparison
            elif eval_mode == "embed":
                from scipy.spatial.distance import cosine
                def get_embedding(input_text):
                    url = "http://localhost:11434/api/embeddings"
                    payload = json.dumps({
                        "model": "mxbai-embed-large",
                        "prompt": input_text,
                        "temperature": 0,
                        "seed": 123
                    })
                    response = requests.post(url, data=payload)
                    if response.status_code != 200:
                        raise Exception(f"Error creating request: {response.text}")
                    else:
                        response_json = response.json()
                        embedding = response_json["embedding"]
                        return embedding
                # handle case where output is empty resulting in empty embedding, caught above if both are empty
                if gold_command_output == "" or model_command_output == "":
                    p3_score = 0
                else:
                    ground_truth_embedding = get_embedding(gold_command_output[:1000])
                    model_embedding = get_embedding(model_command_output[:1000])
                    similarity = 1 - cosine(ground_truth_embedding, model_embedding)
                    if similarity > eval_param:
                        p3_score = 0.33

            # Local LLM command output comparison
            elif eval_mode == "ollama":
                result = "false"
                try:
                    url = "http://localhost:11434/api/chat"
                    payload = json.dumps({
                        "model": eval_param,
                        "messages": [
                            {'role': 'system', 'content': "You will be given a task, two Bash commands, and the output of the two Bash commands. The first command is the ground truth. If the second command accomplishes the task, return true. Otherwise, return false. Only output 'true' or 'false'."},
                            {"role": "user", "content": f"Task: {prompt}, Ground Truth Command: {gold_command}, Model Command: {model_command}, Ground Truth Command Output: {gold_command_output[:1000]}, Model Command Output: {model_command_output[:1000]}"}
                        ],
                        "stream": False,
                        "temperature": 0,
                        "seed": 123
                    })
                    response = requests.post(url, data=payload)
                    if response.status_code != 200:
                        raise Exception(f"Error creating request: {response.text}")
                    else:
                        response_json = response.json()
                        result = response_json['message']['content']
                except Exception as e:
                    raise e
                if ('true' in result) or ('True' in result):
                    p3_score = 0.33

            # OpenAI GPT command output comparison
            elif eval_mode == "openai":
                api_key = os.getenv('ICALFA_OPENAI_API_KEY')
                client = OpenAI(api_key=api_key)
                result = "false"
                try:
                    completion = client.chat.completions.create(
                        model=eval_param,
                        messages=[
                        {"role": "system", "content": "You will be given a task, two Bash commands, and the output of the two Bash commands. The first command is the ground truth. If the second command accomplishes the task, return true. Otherwise, return false. Only output 'true' or 'false'."},
                        {"role": "user", "content": f"Task: {prompt}, Ground Truth Command: {gold_command}, Model Command: {model_command}, Ground Truth Command Output: {gold_command_output[:1000]}, Model Command Output: {model_command_output[:1000]}"}
                        ],
                        temperature=0,
                        seed=123,
                    )
                    result = completion.choices[0].message.content
                except Exception as e:
                    raise e
                if ('true' in result) or ('True' in result):
                    p3_score = 0.33

        info[REWARD]["answer_similarity"] = p3_score
        reward += p3_score

        self.reward = reward 
        self.info.update(info)

        self.logger.info(f"Info: {self.info}")
        self.logger.info(f"Reward: {self.reward}")
        return reward, info

    def close(self):
        self.logger.info("Beginning environment shutdown...")
        self.container.stop()
        self.container_eval.stop()
        self.logger.info("Agent, evaluation containers stopped")
    
    ############################
    ### MARK: Helper methods ###
    ############################

    def clean_cmd(self, action: str) -> str:
        """Cleans action string"""
        entrypoint = IMAGE_TO_SETTINGS[self.image_name]
        return f"{entrypoint} -c \"{action.strip()}\""

    def parse_status(self, status: str) -> List:
        """Parses git status output into list of changes"""
        status_lst = status.split()
        changes = []
        for i in range(0, len(status_lst), 2):
            changes.append((status_lst[i+1], status_lst[i]))
        return changes

    def simplify_path(self, current: str, changed: str) -> str:
        """Resolves path from current working directory path and the argument of the `cd` command"""
        if not changed:
            return current
        if changed[0] == "/":
            current = ""

        path = []
        
        for segment in (current + "/" + changed).split("/"):
            if segment == "..":
                if path:
                    path.pop()
            elif segment and segment != ".":
                path.append(segment)

        return "/" + "/".join(path)