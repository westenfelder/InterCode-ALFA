from icalfa.assets import bash_build_docker
from icalfa.envs import BashEnv
import subprocess
import json
import os

def index_to_img(index):
    index = int(index)
    splits = [153, 49, 57, 23, 18]
    cumsum = 0

    for img_num, count in enumerate(splits, start=0):
        if index < cumsum + count:
            idx = index - cumsum
            return idx, img_num
        cumsum += count
    
    raise ValueError("Index out of allowable range")

def submit_command(index, command, eval_mode="openai", eval_param="gpt-4-0613"):
    """
    Test a command against the InterCode-ALFA benchmark.

    Args:
        index (int): index in the test set.
        command (str): bash command.

    Returns:
        int: 0 for an incorrect command, 1 for a correct command.
    
    Example:
        score = submit_command(0, "ls -al")
        print(score)
    """

    try:
        # setup
        image_names = ["intercode-bash-1", "intercode-bash-2", "intercode-bash-3", "intercode-bash-4", "intercode-bash-5"]
        docker_files = ["nl2bash1.Dockerfile", "nl2bash2.Dockerfile", "nl2bash3.Dockerfile", "nl2bash4.Dockerfile", "nl2bash5.Dockerfile"]
        data_files = ["nl2bash_fs_1.json", "nl2bash_fs_2.json", "nl2bash_fs_3.json", "nl2bash_fs_4.json", "nl2bash_fs_5.json"]

        package_root = os.path.dirname(os.path.abspath(__file__))
        data_files_base = os.path.join(package_root, 'assets/datasets/')

        idx, img_num = index_to_img(index)

        # build env
        bash_build_docker(image_names[img_num], docker_files[img_num])
        env = BashEnv(image_names[img_num], data_path=data_files_base+data_files[img_num], traj_dir="logs/", verbose=False, eval_mode=eval_mode, eval_param=eval_param)

        obs, info = env.reset(idx) # pass the index to prevent random data selection
        obs, done = env.observation, False # obs here is the natural language prompt

        while not done:
            obs, reward, done, info = env.step(command)
            obs, reward, done, info = env.step("submit")

        info_json = json.dumps(info, indent=4)
        
        if reward == 1:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Benchmark Error: {e}")
        print(f"Attempting to reset container.")
        try:
            result = subprocess.run("docker stop $(docker ps -a --filter \"name=intercode*\" -q)", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                result = subprocess.run("docker rm $(docker ps -a --filter \"name=intercode*\" -q)", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    print("Reset container successfully.")
                else:
                    print(f"Failed to reset container. Recommend manually rebuilding Docker containers.")
            else:
                print(f"Failed to reset container. Recommend manually rebuilding Docker containers.")
        except Exception as e2:
            print(f"Failed to reset container. Recommend manually rebuilding Docker containers. Error: {e2}")
        print("Error results in score of zero.")
        return 0