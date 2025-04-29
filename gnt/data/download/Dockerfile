# Use slim Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    unzip \
    jq \
    fonts-liberation \
    libnss3 \
    libxss1 \
    libasound2 \
    libx11-xcb1 \
    libgtk-3-0 \
    xdg-utils \
    ca-certificates \
    gcc \
    build-essential \
    libffi-dev \
    libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/google.gpg && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y google-chrome-stable && \
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

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: set environment variables
ENV PYTHONUNBUFFERED=1

# Set the entrypoint
CMD ["python", "main.py"]
