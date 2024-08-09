import docker, os, time

def bash_build_docker(bash_image_name, dockerfile_name):
    """
    Build the docker image for the InterCode Bash environment. If the image already exists, do nothing.
    """
    client = docker.from_env()
    available_images = [y for x in client.images.list() for y in x.tags]
    if f"{bash_image_name}:latest" in available_images:
        return
    print(f"`{bash_image_name}:latest` not in list of available local docker images, building...")
    
    client.images.build(
        path=os.path.join(os.path.dirname(__file__), 'docker'),
        dockerfile=dockerfile_name,
        tag=bash_image_name,
        rm=True
    )

    # Give some time for Bash server to start
    print("✓ Intercode Bash Docker image built successfully. " + \
        "Waiting for 5 seconds for Bash container to start...\n" + \
        "If you encounter an error, run `docker ps --all` and check if `intercode-bash` containers were created. " + \
        "Container start up time varies by machine.")
    time.sleep(5)