"""
Project APEX
Application Entry Point
"""

from project_apex.core.application import Application


def main() -> None:
    app = Application()

    try:
        app.initialize()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()