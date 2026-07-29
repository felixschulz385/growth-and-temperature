import logging


class _SessionMixin:
    """Selenium session lifecycle for GLASS. Currently unused by crawling/download
    (those use requests/aiohttp directly) — kept for compatibility."""

    def get_selenium_session(self):
        """
        Returns a selenium session (webdriver).
        Creates it if it does not exist.

        Note: The workflow context will be responsible for maintaining
        the persistent session, this method is just a factory.
        """
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        logger = logging.getLogger(__name__)
        logger.info("Creating new Selenium WebDriver for GLASS data source")

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=chrome_options)

        # Set longer page load timeout to handle slow connections
        driver.set_page_load_timeout(120)

        return driver

    def close_selenium_session(self, session):
        """
        Closes the selenium session.

        Args:
            session: The selenium session to close
        """
        if session is not None:
            try:
                logger = logging.getLogger(__name__)
                logger.info("Closing Selenium WebDriver for GLASS data source")
                session.quit()
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Error closing Selenium session: {str(e)}")
