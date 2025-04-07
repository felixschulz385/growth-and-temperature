from workflow import run
from config import BASE_URLS, GCS_BUCKET_NAME

if __name__ == "__main__":
    run(BASE_URLS, GCS_BUCKET_NAME)
