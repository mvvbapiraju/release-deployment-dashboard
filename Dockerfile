FROM python:3-slim
LABEL maintainer="mvvbapiraju@gmail.com"
LABEL description="This is a Docker Image for Release Deployment Dashboard Webpage"

ENV GITLAB_ACCESS_TOKEN=''
ENV GITLAB_APPLICATION_ID=''
ENV GITLAB_APPLICATION_SECRET=''

ENV TZ='America/New_York'

# Set the working directory inside the container
WORKDIR /app

# Switch to the root user to install packages
USER root
# Update package lists and install utils
RUN apt-get update && apt-get install -y curl jq

# Create app user
RUN useradd --user-group --system --no-log-init app && chown app:app .

# Install the required dependencies
COPY setup.py requirements.txt main.py scheduler.py startup.sh priority_project_ids.yml ./
COPY --chown=app:app static ./static
COPY --chown=app:app templates ./templates
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements.txt

# Switch to app user
USER app
# Set the entry point command to run the Flask app
CMD ["./startup.sh"]
