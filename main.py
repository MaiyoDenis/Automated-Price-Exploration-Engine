"""
Project APEX
Application Entry Point
"""

import asyncio
from project_apex.core.application import Application


async def main_async() -> None:
    app = Application()
    await app.run()


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting...")


if __name__ == "__main__":
    main()