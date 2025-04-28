import pytest
import dask.dataframe as df

#@pytest.fixture
def load_data():
    dd = df.read_csv('../data/test.ft.txt', sep='\t', header=None, names=['label', 'text'])
    return dd

# 컬럼명 확인
def test_columns(load_data):
    dr = load_data
    print(dr.columns)

# 데이터 로딩 테스트
def test_loading(load_data):
    dr = load_data
    assert dr.shape[1] == 2  # 두 개의 컬럼이 있는지
    assert 'label' in dr.columns  # 'label' 컬럼이 있는지
    assert 'text' in dr.columns  # 'text' 컬럼이 있는지

# 결측치 확인
def test_missing_values(load_data):
    dr = load_data
    missing_values = dr.isnull().sum().compute()
    assert missing_values['label'] == 0  # 'label' 컬럼에 결측치가 없다고 가정
    assert missing_values['text'] == 0  # 'text' 컬럼에 결측치가 없다고 가정

# 기본 통계량 계산 테스트 (예: label 컬럼 값의 분포)
def test_label_distribution(load_data):
    dr = load_data
    label_counts = dr['label'].value_counts().compute()
    assert label_counts.sum() > 0  # 라벨 값이 있는지
    assert label_counts[0] > 0  # 최소 하나 이상의 라벨이 있어야 한다고 가정