import logging
import os

log = logging.getLogger(__name__)

# directory paths
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
DIR_DATA = os.path.join(PROJECT_ROOT, "data")
DIR_CONFIG = os.path.join(PROJECT_ROOT, "config")
