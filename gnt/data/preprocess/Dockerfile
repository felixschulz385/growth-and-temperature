FROM mambaorg/micromamba:1.5.6

# Set working directory
WORKDIR /app

USER root
# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

# Copy environment file
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Create environment using micromamba
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Copy the code
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

# Activate the base environment
SHELL ["/usr/local/bin/_entrypoint.sh", "micromamba", "shell", "hook", "-s", "bash", "-p", "/opt/conda", "&&"]

# Set the entrypoint to run the script
ENTRYPOINT ["micromamba", "run", "-n", "base", "python", "main.py"]
CMD ["/config/workflow-config.json"]