# 파이썬 라이브러를 활용한 머신러닝 - Andreas Muller 등 저
### 중요사항
1. Scikit-Learn의 기초 부분을 보고 복습할 수 있도록 다시 학습 및 코드 추가
2. 책에 있는 코드를 단순히 따라하는 것이 아니라 나만의 코드로 작성
3. 변수명은 Snake Case 사용
4. 책에 없는 Scikit-Learn 외 기타 통계학 내용 등 추가

### Chapter 1. 소개 및 기타
1. fit(X, y), predict(X), score(X, y) 메서드
2. train_test_split(X, y, stratify, random_state) 함수
3. predict_proba() 메서드

### Chapter 2. 지도학습
1. K-Nearest Neighbors
    - neighbors.KNeighborsClassifier(n_neighbors, metric)
    - neighbors.KNeighborsRegressor(n_neighbors, metric, weights)
    - 훈련 세트가 매우 크면 예측이 느려지고, 특성 값 대부분이 0인 데이터셋과는 안 맞는 방식
    - 거리가 중요하므로 알고리즘을 사용할 때 정규화를 수행하는 것이 일반적
2. Linear Model: Linear Regression
    - linear_model.LinearRegression()
    - coef_, intercept_ 속성
    - Ridge(), Lasso(), ElasticNet(l1_ratio) 등으로 대체 가능
3. Linear Model: Ridge
    - linear_model.Ridge(alpha, solver, max_iter)
    - L2 Regularization: 가중치가 0이 되지는 않음
4. Linear Model: Lasso
    - linear_model.Lasso(alpha, solver, max_iter)
    - L1 Regularization: 가중치가 0이 될 수도 있음
5. Linear Model: LogisticRegression
    - linear_model.LogisticRegression(C, solver, penalty, max_iter)
    - 기본적으로 L2 Regularization 사용(solver='liblinear' 시 L1규제 사용 가능)
    - predict_proba() 메서드에서 sigmoid function을 적용한 확률값 제공
6. Linear Model: etc
    - SGDClassifier, SGDRegressor: 확률적 경사 하강법 활용
        - linear_model.SGDClassifier(alpha, learning_rate, eta0, loss, penalty, random_state, n_jobs)
        - linear_model.SGDRegressor(alpha, learning_rate, eta0, loss, penalty, random_state, n_jobs)
