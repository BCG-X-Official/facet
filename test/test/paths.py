import logging
import os

log = logging.getLogger(__name__)

# directory paths
DIR_DATA = "data"
DIR_CONFIG = "config"

# file paths
TEST_DATA_CSV = os.path.join(DIR_DATA, "master_table_clean_anon_144.csv")
TEST_CONFIG_YML = os.path.join(DIR_CONFIG, "test_config.yml")
