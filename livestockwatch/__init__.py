import json
import logging
import logging.config
import pytz

try:
    with open("logging.json", 'rt') as f:
        logger_config = json.load(f)
    logging.config.dictConfig(logger_config)
except AssertionError as e:
    print(e)

LOCAL_TIMEZONE = pytz.timezone('Asia/Tokyo')
logger = logging.getLogger(__package__)