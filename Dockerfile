# Base image with micromamba for better package management
FROM mambaorg/micromamba:1.5.6

# Set working directory
WORKDIR /app

USER root
# Install only essential system dependencies - minimized list
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For downloading and extracting files
    wget \
    curl \
    unzip \
    # For Chrome installation and ChromeDriver
    gnupg \
    ca-certificates \
    jq \
    # Minimal Chrome dependencies that conda can't provide
    fonts-liberation \
    libnss3 \
    libxss1 \
    libasound2 \
    libx11-xcb1 \
    libgtk-3-0 \
    xdg-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Chrome for web scraping capabilities
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/google.gpg && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y --no-install-recommends google-chrome-stable && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Use the Chrome for Testing JSON API to install a matching ChromeDriver
RUN set -eux; \
    CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+\.\d+' | head -1); \
    CHROME_MAJOR_MINOR=$(echo $CHROME_VERSION | grep -oP '^\d+\.\d+\.\d+'); \
    echo "Installed Chrome version: $CHROME_VERSION"; \
    JSON_URL="https://googlechromelabs.github.io/chrome-for-testing/latest-patch-versions-per-build-with-downloads.json"; \
    DRIVER_URL=$(curl -s $JSON_URL | jq -r --arg ver "$CHROME_MAJOR_MINOR" '.builds[$ver] | select(.downloads.chromedriver) | .downloads.chromedriver[] | select(.platform == "linux64") | .url'); \
    echo "Downloading ChromeDriver from: $DRIVER_URL"; \
    wget -O /tmp/chromedriver.zip "$DRIVER_URL"; \
    unzip /tmp/chromedriver.zip -d /usr/local/bin/chromedriver-tmp; \
    mv /usr/local/bin/chromedriver-tmp/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver; \
    chmod +x /usr/local/bin/chromedriver; \
    rm -rf /tmp/chromedriver.zip /usr/local/bin/chromedriver-tmp

USER $MAMBA_USER

# Copy environment file
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Create environment using micromamba and install pip packages
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Copy the entire project
COPY --chown=$MAMBA_USER:$MAMBA_USER gnt /app/gnt

# Copy the entrypoint script and make it executable
COPY --chown=$MAMBA_USER:$MAMBA_USER entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONPATH=/app

# Set the entrypoint
ENTRYPOINT ["micromamba", "run", "-n", "base", "/app/entrypoint.sh"]
CMD ["/config/workflow-config.json"]