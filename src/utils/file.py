import json

from logger import logger


def load_json(path, **rest):
    try:
        with open(path, "r") as f:
            loaded = json.load(f, **rest)
    except json.decoder.JSONDecodeError as error:
        logger.critical(f"Error when loading json file at: '{path}'\n{error}")
        exit(1)
    return loaded
