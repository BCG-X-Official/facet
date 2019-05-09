import logging
import warnings

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings(
    "ignore", message=r"Starting from version 2", category=UserWarning
)
