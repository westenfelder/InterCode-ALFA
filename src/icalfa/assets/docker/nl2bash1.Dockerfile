FROM ubuntu:noble-20240429

# Install required dependencies
RUN apt-get update && \
    apt-get install -y bash python3 psmisc bsdmainutils cron imagemagick dnsutils git tree net-tools iputils-ping coreutils curl cpio jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create custom file structure
COPY ./setup_nl2b_fs_1.sh /
RUN chmod +x /setup_nl2b_fs_1.sh
RUN /setup_nl2b_fs_1.sh

# Set env vars MODIFICATIONS HERE
ENV FILES="/testbed/hello.c /testbed/FooBar.html"

# Commit custom file system to determine diffs
COPY ./docker.gitignore /
RUN mv docker.gitignore .gitignore
RUN git config --global user.email "intercode@pnlp.org"
RUN git config --global user.name "intercode"
RUN git init
RUN git add -A
RUN git commit -m 'initial commit'

# Set the working directory
WORKDIR /


