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
    - sklearn.neighbors.KNeighborsClassifier(n_neighbors, metric)
    - sklearn.neighbors.KNeighborsRegressor(n_neighbors, metric, weights)
    - 훈련 세트가 매우 크면 예측이 느려지고, 특성 값 대부분이 0인 데이터셋과는 안 맞는 방식
    - 거리가 중요하므로 알고리즘을 사용할 때 정규화를 수행하는 것이 일반적
2. Linear Model: Linear Regression
    - sklearn.linear_model.LinearRegression()
    - coef_, intercept_ 속성
    - Ridge(), Lasso(), ElasticNet(l1_ratio) 등으로 대체 가능
3. Linear Model: Ridge
    - sklearn.linear_model.Ridge(alpha, solver, max_iter)
    - L2 Regularization: 가중치가 0이 되지는 않음
4. Linear Model: Lasso
    - sklearn.linear_model.Lasso(alpha, solver, max_iter)
    - L1 Regularization: 가중치가 0이 될 수도 있음
5. Linear Model: LogisticRegression
    - sklearn.linear_model.LogisticRegression(C, solver, penalty, max_iter)
    - 기본적으로 L2 Regularization 사용(solver='liblinear' 시 L1규제 사용 가능)
    - predict_proba() 메서드에서 sigmoid function을 적용한 확률값 제공
6. Linear Model: etc
    - SGDClassifier, SGDRegressor: 확률적 경사 하강법 활용
        - sklearn.linear_model.SGDClassifier(alpha, learning_rate, eta0, loss, penalty, random_state, n_jobs)
        - sklearn.linear_model.SGDRegressor(alpha, learning_rate, eta0, loss, penalty, random_state, n_jobs)
7. Naive Bayes Model(only Classification)
    - GaussianNB(연속적), BernoulliNB(이진), MultinomialNB(카운트)
    - sklearn.naive_bayes.GaussianNB()
    - sklearn.naive_bayes.BernoulliNB()
    - sklearn.naive_bayes.MultinomialNB()
    - 훈련과 예측 속도가 빠름 → 매우 큰 데이터셋에는 시도해볼만
8. Decision Tree
    - sklearn.tree.DecisionTreeClassifier(max_depth, max_leaf_nodes, min_samples_leaf, ccp_alpha, criterion, random_state)
    - sklearn.tree.DecisionTreeRegressor(max_depth, max_leaf_nodes, min_samples_leaf, ccp_alpha, criterion, random_state)
    - feature_importances_ 속성
    - 회귀의 경우 extrapolation(외삽)이 불가능 → 훈련 데이터 범위 밖 포인트 예측 불가
9. Ensemble: RandomForest
    - sklearn.ensemble.RandomForestClassifier(n_estimators, bootstrap, max_samples, max_features, n_jobs, etc(== decision tree))
    - sklearn.ensemble.RandomForestRegressor(n_estimators, bootstrap, max_samples, max_features, n_jobs, etc(== decision tree))
    - estimators_, feature_importances_ 속성
    - 개개의 트리보다 덜 과대적합되고 훨씬 좋은 결정 경계
    - 텍스트 데이터 같은 높은 차원의 희소한 데이터에는 잘 작동하지 않음 → 선형 모델이 더 적합
10. Ensemble: GradientBoosting
    - sklearn.ensemble.GradientBoostingClassifier(n_estimators, learning_rate, loss, random_state, etc(== decision tree))
    - sklearn.ensemble.GradientBoostingRegressor(n_estimators, learning_rate, loss, random_state, etc(== decision tree))
    - GradientBoosting에서는 n_estimators를 키우면 과대적합될 가능성
    - GradientBoosting에서는 max_depth를 매우 작게 설정할 필요 → 약한 학습기에서 시작
11. Ensemble: etc
    - Bagging
        - sklearn.ensemble.BaggingClassifier(estimator, n_estimators, max_samples, oob_score, random_state, n_jobs)
        - sklearn.ensemble.BaggingRegressor(estimator, n_estimators, max_samples, oob_score, random_state, n_jobs)
        - RandomForest도 oob_score 지원
    - ExtraTree
    - AdaBoost
        - sklearn.ensemble.AdaBoostClassifier(estimator, n_estimators, learning_rate, random_state)
        - sklearn.ensemble.AdaBoostRegressor(estimator, n_estimators, learning_rate, random_state)
        - 이전 모델이 잘못 분류한 샘플에 가중치를 높여 다음 모델을 훈련
    - HistGradientBoosting
12. Support Vector Machine
    - sklearn.svm.SVC(kernel, C, gamma)
    - sklearn.svm.SVR(kernel, C, gamma)
    - support_vectors_, dual_coef_ 속성
    - 하이퍼파라미터 설정과 데이터 스케일에 매우 민감, 샘플이 많으면 어려움
13. Neural Network: Multilayer Perceptron
    - sklearn.neural_network.MLPClassifier(solver, hidden_layer_sizes, max_iter, activation, alpha, random_state)
    - sklearn.neural_network.MLPRegressor(solver, hidden_layer_sizes, max_iter, activation, alpha, random_state)
    - coefs_ 속성
    - 데이터 스케일이 영향을 미침 → 표준화/정규화 필요

### Chapter 3. 비지도 학습과 데이터 전처리
1. Data Scale Preprocessing
    - StandardScaler
        - sklearn.preprocessing.StandardScaler()
        - mean을 0, variance를 1
    - RobustScaler
        - sklearn.preprocessing.RobustScaler()
        - median과 quartile을 사용
    - MinMaxScaler
        - sklearn.preprocessing.MinMaxScaler()
        - 모든 값들이 0과 1 사이에 위치
    - Preprocessing: etc
        - Normalizer: sklearn.preprocessing.Normalizer()
        - QuantileTransformer: sklearn.preprocessing.QuantileTransformer(n_quantiles, output_distribution)
        - PowerTransformer: sklearn.preprocessing.PowerTransformer(method)
        - 모든 scaler들은 훈련세트와 테스트세트에 같은 변환을 적용해야
    - Preprocessing: Handling missing values
        - sklearn.impute.SimpleImputer(missing_values, strategy, fill_value)
        - sklearn.impute.KNNImputer(missing_values, n_neighbors, weights)
        - to be written
2. PCA(Principal Component Analysis)
    - sklearn.decomposition.PCA(n_components, whiten, random_state)
    - inverse_transform() 메서드
    - components_, explained_variance_ratio_, n_components_ 속성
3. NMF(Non-negative Matrix Factorization)
    - sklearn.decomposition.NMF(n_components, init, tol, max_iter, random_state)
    - 덮어써진 데이터에서 원본 성분을 구하는데 유용 + NMF로 생성한 성분은 순서가 없음
4. t-SNE(t-distribution Stochastic Neighbor Embedding)
    - sklearn.manifold.TSNE(n_components, init, perplexity, random_state)
    - 훈련 데이터를 새로운 표현으로 변환, but 새로운 데이터에는 적용 불가능 → transform() 메서드 미존재
5. Clustering: K-Means
    - sklearn.cluster.KMeans(n_clusters, random_state)
    - labels_, cluster_centers_, inertia_ 속성
    - transform() 메서드가 반환하는 값은 데이터 포인트에서 각 클러스터 중심까지의 거리
    - K-Means는 모든 클러스터의 반경이 동일, 모든 방향이 똑같이 중요하다는 가정
6. Clustering: Agglomerative Clustering
    - sklearn.cluster.AgglomerativeClustering(n_clusters, linkage)
    - 새로운 데이터 포인트 예측이 불가하므로 predict() 메서드 미존재 → fit_predict() 메서드 활용
    - children_, distances_ 속성
7. Clustering: DBSCAN
    - sklearn.cluster.DBSCAN(min_samples, eps)
    - 한 데이터 포인트에서 eps 거리 안에 데이터가 min_samples 개수만큼 들어있으면 핵심 샘플로 분류
    - 데이터의 밀집 지역이 한 군집을 구성 → 클러스터 개수 미리 지정 불필요
    - 새로운 데이터 포인트 예측이 불가하므로 predict() 메서드 미존재 → fit_predict() 메서드 활용
8. 군집 알고리즘의 비교와 평가
    - ARI
        - sklearn.metrics.cluster.adjusted_rand_score(y, pred)
    - Silhouette coefficient
        - sklearn.metrics.cluster.silhouette_score(X, pred)
        - 군집의 밀집 정도를 계산 → 모양이 복잡할 때는 잘 안들어맞는 문제
    - 군집 평가에 더 적합한 전략은 견고성 기반(robustness-based) → Scikit-Learn에는 미구현

### Chapter 4. 데이터 표현과 특성 공학
1. Variables Encoding: One-Hot-Encoding(Dummy Variables)
    - pandas.get_dummies(df, columns)
    - sklearn.preprocessing.OneHotEncoder(sparse)
        - get_feature_names_out() 메서드
        - 모든 열에 인코딩 수행
    - sklearn.compose.ColumnTransformer([(name, estimator, columns)])
        - named_transformers_ 속성
        - 변환된 출력열에 대응하는 입력열을 찾지 못하는 것이 단점
    - sklearn.compose.make_column_transformer((estimator, columns))
2. Variables Encoding: Discretization(Binning)
    - sklearn.preprocessing.KBinsDiscretizer(n_bins, strategy, encode)
    - bin_edges_ 속성
3. Feature Encoding: Interaction & Polynomial Features
    - sklearn.preprocessing.PolynomialFeatures(degree, include_bias, interaction_only)
    - get_feature_names_out() 메서드
    - cf. np.log(), np.exp() 등을 활용하여 분포를 정규분포와 비슷하게 만들수도
4. Feature Selection
    - sklearn.feature_selection.SelectKBest(score_func, k)
        - get_support() 메서드
    - sklearn.feature_selection.SelectPercentile(score_func, percentile)
    - sklearn.feature_selection.SelectFromModel(estimator, threshold)
        - 특성의 중요도를 측정할 수 있는 모델이어야 → Decision Tree 기반, Linear 기반 등
    - sklearn.feature_selection.RFE(estimator, n_features_to_select)

### Chapter 5. 모델 평가와 성능 향상
1. Cross Validation
    - sklearn.model_selection.cross_validate(estimator, X, y, cv, groups, return_train_score)
    - sklearn.model_selection.cross_val_score(estimator, X, y, cv, groups)
    - Cross Validation Splitter
        - sklearn.model_selection.(Stratified)KFold(n_splits, shuffle, random_state)
        - sklearn.model_selection.LeaveOneOut()
        - sklearn.model_selection.(Stratified)ShuffleSplit(train_size, test_size, n_splits)
        - sklearn.model_selection.(Stratified)GroupKFold(n_splits)
        - sklearn.model_selection.Repeated(Stratified)KFold(n_splits, n_repeats)
2. Grid Search
    - sklearn.model_selection.GridSearchCV()