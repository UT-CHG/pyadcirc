"""
Pyadcirc Log functionality

"""
from loguru import logger
from rich.logging import RichHandler

logger.disable("pyadcirc")


def enable(file=None, level="INFO", fmt="{message}"):
    """
    Turn on logging for module with appropriate message format

    Parameters
    ----------
    file : str, optional
        Path to file to log to. If None, log to stdout and use RichHandler
        for colorful output. If not output to file, with data serialized in json.
    level : str, optional
        Logging level. Default is INFO.
    fmt : str, optional
        Message format. Default is "{message}".
    """
    if file is None:
        logger.configure(
            handlers=[
                {
                    "sink": RichHandler(markup=True, rich_tracebacks=True),
                    "level": level,
                    "format": fmt,
                }
            ]
        )
    else:
        logger.configure(
            handlers=[
                {
                    "sink": file,
                    "serialize": True,
                    "level": level,
                    "format": fmt,
                    "rotation": "10 MB",
                    "enqueue": True,
                }
            ]
        )
    logger.enable("pyadcirc")
    logger.info("Logger initialized")

    return logger


def disable():
    """
    Turn of logging
    """
    logger.disable("pyadcirc")
