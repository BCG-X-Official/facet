import pandas as pd
import yieldengine.core


def load_raw_data(input_path: str) -> pd.DataFrame:
    inputfile_config = yieldengine.core.get_global_config(section="inputfile")
    return pd.read_csv(filepath_or_buffer=input_path,
                       delimiter=inputfile_config["delimiter"],
                       header=inputfile_config["header"])


def validate_raw_data(input_data_df: pd.DataFrame) -> None:
    inputfile_config = yieldengine.core.get_global_config(section="inputfile")

    date_column_name = inputfile_config["date_column_name"]
    yield_column_name = inputfile_config["yield_column_name"]

    if date_column_name not in input_data_df.columns:
        raise ValueError("The expected date column '%s' was not found within the raw data file!" % date_column_name)

    if yield_column_name not in input_data_df.columns:
        raise ValueError("The expected yield column '%s' was not found within the raw data file!" % yield_column_name)
