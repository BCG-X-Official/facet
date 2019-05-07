import logging
from typing import *

log = logging.getLogger(__name__)


def deprecated(message: str):
    """
    Decorator to mark functions as deprecated.
    It will result in a warning being logged when the function is used.
    """

    def _deprecated_inner(func: callable) -> callable:
        def new_func(*args, **kwargs) -> Any:
            """
            Function wrapper
            """
            message_header = "Call to deprecated function {}".format(func.__name__)
            if message is None:
                log.warning(message_header)
            else:
                log.warning("{}: {}".format(message_header, message))
            return func(*args, **kwargs)

        return new_func

    return _deprecated_inner
