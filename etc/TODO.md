1. 데이터 로딩
목표: 데이터를 메모리로 로딩하고, 데이터 구조를 이해한다.

작업:

 데이터를 CSV 파일에서 pandas 또는 Dask를 이용해 로딩한다.

 데이터셋에 있는 주요 컬럼 확인 (예: ProductID, CustomerID, Quantity, Price 등)

 로딩된 데이터의 샘플 출력하여 데이터의 형태 확인

# pandas 사용 예시
import pandas as pd
df = pd.read_csv('data/OnlineRetail.csv')
print(df.head())  # 데이터 상위 5개 행 출력



2. 데이터 타입 확인
목표: 각 열의 데이터 타입을 확인하여, 필요한 전처리 작업을 계획한다.

작업:

 데이터프레임의 각 열에 대한 데이터 타입 확인 (df.dtypes)

 적절한 타입으로 변환이 필요한 열을 찾아 변환한다 (예: 날짜 데이터를 datetime 타입으로 변환)

print(df.dtypes)  # 각 열의 데이터 타입 확인



3. 결측치 확인
목표: 결측치(Missing values)가 있는지 확인하고, 이를 처리할 방법을 결정한다.

작업:

 각 열의 결측치 개수 확인 (df.isnull().sum())

 결측치가 있는 열을 처리 (삭제/대체 등)

print(df.isnull().sum())  # 결측치 개수 확인



4. 기본 통계 정보 확인
목표: 데이터의 분포나 주요 통계 값을 확인한다.

작업:

 데이터셋의 기본 통계 정보 확인 (df.describe())

 각 열의 최대값, 최소값, 평균값 등을 확인하여 데이터 특성을 파악

print(df.describe())  # 기본 통계 정보 출력



5. 데이터 시각화 (기본 탐색)
목표: 데이터를 시각적으로 탐색하여, 데이터의 분포나 이상치를 확인한다.

작업:

 matplotlib 또는 seaborn을 이용해 데이터를 시각화한다.

 Quantity, Price, CustomerID 등의 변수에 대해 히스토그램, 박스플롯 등을 그려본다.

# Price의 히스토그램 그리기
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(df['Price'], bins=50, kde=True)
plt.show()



6. 특이값(이상치) 탐지
목표: 데이터에 특이값이 있는지 확인하고, 이상치를 어떻게 처리할지 결정한다.

작업:

 Quantity, Price 등 중요한 컬럼에 대해 이상치(Outliers)를 탐지한다.

 이상치를 시각화하여 이를 처리할 방법을 결정한다 (제거/수정 등)

# 박스플롯으로 이상치 탐지
sns.boxplot(x=df['Price'])
plt.show()



7. 필요한 특성(Feature) 생성
목표: 모델링을 위해 필요한 특성을 생성한다.

작업:

 고객별 구매 횟수, 총 구매 금액 등을 계산하여 새로운 특성 생성

 제품별 판매 횟수, 총 판매 금액 등을 계산하여 추가적인 특성 생성

# 고객별 구매 횟수 계산
customer_purchase_count = df.groupby('CustomerID').size()
df['CustomerPurchaseCount'] = df['CustomerID'].map(customer_purchase_count)



8. 데이터 저장 및 준비
목표: 데이터를 전처리 후 저장하여, 모델 학습 및 예측에 사용될 수 있도록 준비한다.

작업:

 전처리된 데이터를 CSV 파일로 저장하거나 데이터베이스에 저장

 모델 학습에 필요한 형태로 데이터를 준비

# 전처리 후 데이터 저장
df.to_csv('processed_data.csv', index=False)