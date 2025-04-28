import dask.dataframe as df
import os

def load_data(file_path: str, delimiter: str = '\t') -> df.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    df = df.read_csv(file_path, delimiter=delimiter, assume_missing=True)

    return df