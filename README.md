# 파이썬 라이브러를 활용한 머신러닝 - Andreas Muller 등 저
### 중요사항
1. Scikit-Learn의 기초 부분을 보고 복습할 수 있도록 다시 학습 및 코드 추가
2. 책에 있는 코드를 단순히 따라하는 것이 아니라 나만의 코드로 작성
3. 변수명은 Snake Case 사용
4. 책에 없는 Scikit-Learn 외 기타 통계학 내용 등 추가

### Chapter 1. 소개
1. fit(X, y), predict(X), score(X, y) 메서드
2. train_test_split(X, y, stratify, random_state) 함수

### Chapter 2. 지도학습
1. K-Nearest Neighbors
    - neighbors.KNeighborsClassifier(n_neighbors, metric)
    - neighbors.KNeighborsRegressor(n_neighbors, metric, weights)
    - 훈련 세트가 매우 크면 예측이 느려지고, 특성 값 대부분이 0인 데이터셋과는 안 맞는 방식
