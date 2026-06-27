from loguru import logger

from project_apex.config.config import Config
from project_apex.database.sqlite_manager import Database
from project_apex.utils.logger import setup_logger


def main() -> None:
    setup_logger()

    config = Config()

    database = Database(config.get("database", "path"))

    database.connect()

    database.initialize()

    logger.info("Database initialized successfully.")

    database.close()


if __name__ == "__main__":
    main()