# 파이썬 라이브러를 활용한 머신러닝 - Andreas Muller 등 저
### 중요사항
1. Scikit-Learn의 기초 부분을 보고 복습할 수 있도록 다시 학습 및 코드 추가
2. 책에 있는 코드를 단순히 따라하는 것이 아니라 나만의 코드로 작성
3. 변수명은 Snake Case 사용
4. 책에 없는 Scikit-Learn 외 기타 통계학 내용 등 추가

### Chapter 1. 소개 및 기타
1. fit(X, y), predict(X), score(X, y) 메서드
2. train_test_split(X, y, stratify, random_state) 함수
3. predict_proba(), decision_function() 메서드
4. n_features_in_ 속성

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
7. Naive Bayes Model(only Classification)
    - GaussianNB(연속적), BernoulliNB(이진), MultinomialNB(카운트)
    - naive_bayes.GaussianNB()
    - naive_bayes.BernoulliNB()
    - naive_bayes.MultinomialNB()
    - 훈련과 예측 속도가 빠름 → 매우 큰 데이터셋에는 시도해볼만
8. Decision Tree
    - tree.DecisionTreeClassifier(max_depth, max_leaf_nodes, min_samples_leaf, ccp_alpha, criterion, random_state)
    - tree.DecisionTreeRegressor(max_depth, max_leaf_nodes, min_samples_leaf, ccp_alpha, criterion, random_state)
    - feature_importances_ 속성
    - 회귀의 경우 extrapolation(외삽)이 불가능 → 훈련 데이터 범위 밖 포인트 예측 불가
9. Ensemble: RandomForest
    - ensemble.RandomForestClassifier(n_estimators, bootstrap, max_samples, max_features, n_jobs, etc(== decision tree))
    - ensemble.RandomForestRegressor(n_estimators, bootstrap, max_samples, max_features, n_jobs, etc(== decision tree))
    - estimators_, feature_importances_ 속성
    - 개개의 트리보다 덜 과대적합되고 훨씬 좋은 결정 경계
    - 텍스트 데이터 같은 높은 차원의 희소한 데이터에는 잘 작동하지 않음 → 선형 모델이 더 적합
10. Ensemble: GradientBoosting
    - ensemble.GradientBoostingClassifier(n_estimators, learning_rate, loss, random_state, etc(== decision tree))
    - ensemble.GradientBoostingRegressor(n_estimators, learning_rate, loss, random_state, etc(== decision tree))
    - GradientBoosting에서는 n_estimators를 키우면 과대적합될 가능성
    - GradientBoosting에서는 max_depth를 매우 작게 설정할 필요 → 약한 학습기에서 시작
11. Ensemble: etc
    - Bagging
        - ensemble.BaggingClassifier(estimator, n_estimators, max_samples, oob_score, random_state, n_jobs)
        - ensemble.BaggingRegressor(estimator, n_estimators, max_samples, oob_score, random_state, n_jobs)
        - RandomForest도 oob_score 지원
    - ExtraTree
    - AdaBoost
        - ensemble.AdaBoostClassifier(estimator, n_estimators, learning_rate, random_state)
        - ensemble.AdaBoostRegressor(estimator, n_estimators, learning_rate, random_state)
        - 이전 모델이 잘못 분류한 샘플에 가중치를 높여 다음 모델을 훈련
    - HistGradientBoosting
12. Support Vector Machine
    - svm.SVC(kernel, C, gamma)
    - svm.SVR(kernel, C, gamma)
    - support_vectors_, dual_coef_ 속성
    - 하이퍼파라미터 설정과 데이터 스케일에 매우 민감, 샘플이 많으면 어려움
13. Neural Network: Multilayer Perceptron
    - neural_network.MLPClassifier(solver, hidden_layer_sizes, max_iter, activation, alpha, random_state)
    - neural_network.MLPRegressor(solver, hidden_layer_sizes, max_iter, activation, alpha, random_state)
    - coefs_ 속성
    - 데이터 스케일이 영향을 미침 → 표준화/정규화 필요

### Chapter 3. 비지도 학습과 데이터 전처리
1. Data Scale Preprocessing
    - StandardScaler
        - preprocessing.StandardScaler()
        - mean을 0, variance를 1
    - RobustScaler
        - preprocessing.RobustScaler()
        - median과 quartile을 사용
    - MinMaxScaler
        - preprocessing.MinMaxScaler()
        - 모든 값들이 0과 1 사이에 위치
    - Preprocessing: etc
        - Normalizer: preprocessing.Normalizer()
        - QuantileTransformer: preprocessing.QuantileTransformer(n_quantiles, output_distribution)
        - PowerTransformer: preprocessing.PowerTransformer(method)
        - 모든 scaler들은 훈련세트와 테스트세트에 같은 변환을 적용해야
    - Preprocessing: Handling missing values
        - impute.SimpleImputer(missing_values, strategy, fill_value)
        - impute.KNNImputer(missing_values, n_neighbors, weights)
        - to be written
2. PCA(Principal Component Analysis)
    - decomposition.PCA(n_components, whiten, random_state)
    - inverse_transform() 메서드
    - components_, explained_variance_ratio_, n_components_ 속성