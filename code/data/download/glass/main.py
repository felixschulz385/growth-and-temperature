from workflow import run
from config import BASE_URL, GCS_BUCKET_NAME, DATA_SOURCE_NAME, DATA_SOURCES, FILE_EXTENSIONS

if __name__ == "__main__":
    source_cls = DATA_SOURCES[DATA_SOURCE_NAME](BASE_URL, FILE_EXTENSIONS)
    run(source_cls, GCS_BUCKET_NAME)
