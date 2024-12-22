# Download the binary for your system
sudo curl --output /usr/local/bin/gitlab-runner https://gitlab-runner-downloads.s3.amazonaws.com/latest/binaries/gitlab-runner-darwin-arm64

# Give it permission to execute
sudo chmod +x /usr/local/bin/gitlab-runner

# The rest of the commands execute as the user who will run the runner
# Register the runner (steps below), then run
cd ~
gitlab-runner install
gitlab-runner start

# Register runner
gitlab-runner register

# --url <GitLab instance URL> 
# --registration-token <REGISTRATION_TOKER>
# --executor docker
