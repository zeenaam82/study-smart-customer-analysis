import os
import dask.bag as db

def load_data_dask(file_path=None):
    if file_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))  # src/data/ 기준
        file_path = os.path.join(base_dir, '../../data/test.ft.txt')  # 절대 경로 변환

    bag = db.read_text(file_path)

    def parse_line(line):
        label, text = line.split(' ', 1)
        label = int(label.replace('__label__', ''))
        return {'label': label, 'text': text.strip()}

    parsed = bag.map(parse_line)
    df = parsed.to_dataframe()
    return df

def load_data_pandas(file_path=None):
    df = load_data_dask(file_path).compute()
    return df.reset_index(drop=True)