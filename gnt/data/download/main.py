# main.py
import logging
from gnt.data.download.workflow import run
from gnt.data.download.config import (
    BASE_URL, 
    GCS_BUCKET_NAME, 
    DATA_SOURCE_NAME, 
    DATA_SOURCES, 
    FILE_EXTENSIONS,
    MAX_CONCURRENT_DOWNLOADS,
    MAX_QUEUE_SIZE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info(f"Starting download process for source: {DATA_SOURCE_NAME}")
        source_cls = DATA_SOURCES[DATA_SOURCE_NAME](BASE_URL, FILE_EXTENSIONS)
        
        logger.info(f"Running workflow with max_concurrent_downloads={MAX_CONCURRENT_DOWNLOADS}, "
                   f"max_queue_size={MAX_QUEUE_SIZE}")
        run(
            source_cls, 
            GCS_BUCKET_NAME, 
            max_concurrent_downloads=MAX_CONCURRENT_DOWNLOADS,
            max_queue_size=MAX_QUEUE_SIZE
        )
        logger.info("Download process completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise