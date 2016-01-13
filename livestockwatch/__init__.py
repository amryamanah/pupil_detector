import json
import logging
import logging.config

try:
    with open("logging.json", 'rt') as f:
        logger_config = json.load(f)
    logging.config.dictConfig(logger_config)
except AssertionError as e:
    print(e)

logger = logging.getLogger(__package__)