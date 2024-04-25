import logging
import types
import sys,os
from pathlib import Path

DEBUG_MODE = bool(int(os.environ.get('DEBUG_MODE', 0)))


def create_logger(name: str):
    logger_ = logging.getLogger(name)  # type: logging.Logger

    if DEBUG_MODE:
        logger_.setLevel(logging.DEBUG)
    else:
        logger_.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger_.addHandler(handler)
    return logger_

def log_response_error(self, resp):
    if sys.exc_info() != (None, None, None):
        self.exception('request failed')

    if resp is None:
        self.error('request hasn\'t been sent out')
    else:
        try:
            self.error(resp.json())
        except Exception:
            try:
                self.error(resp.text)
            except Exception:
                self.error(f'unknown resp content (status code {resp.status_code})')
