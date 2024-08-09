FROM alpine:3.20.0

# Install required dependencies
RUN apk add git

# Create custom testbed directory
COPY ./setup_nl2b_fs_5.sh /
RUN chmod +x /setup_nl2b_fs_5.sh
RUN /setup_nl2b_fs_5.sh

# Commit custom testbed directory to determine diffs
COPY ./docker.gitignore /
RUN mv docker.gitignore .gitignore
RUN git config --global user.email "intercode@pnlp.org"
RUN git config --global user.name "intercode"
RUN git init
RUN git add -A
RUN git commit -m 'initial commit'

# Set the working directory
WORKDIR /