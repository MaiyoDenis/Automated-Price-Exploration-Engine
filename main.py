from project_apex.config.config import Config
from project_apex.utils.logger import setup_logger
from loguru import logger


def main() -> None:
    setup_logger()

    logger.info("Starting Project APEX")

    config = Config()

    logger.info(f"Application: {config.get('application', 'name')}")
    logger.info(f"Market: {config.get('market', 'symbol')}")
    logger.info("Configuration loaded successfully")


if __name__ == "__main__":
    main()