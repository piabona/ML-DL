

- **상관관계 기반 선택 (Correlation-based Feature Selection)**: 타겟 변수와 각 feature 간의 상관관계를 계산하고, 상관관계가 높은 feature들을 선택합니다. 주로 Pearson 상관계수나 Spearman 상관계수를 사용합니다.

- **재귀적 feature 제거 (Recursive Feature Elimination, RFE)**: 모든 feature를 사용한 상태에서 모델을 학습한 후, 중요도가 낮은 feature들을 하나씩 제거하면서 최적의 feature subset을 찾습니다.

- **SelectKBest**: 특정 통계적 테스트를 사용하여 가장 좋은 k개의 feature를 선택합니다. 예를 들어, chi-square 테스트나 ANOVA 등을 사용할 수 있습니다.

- **Lasso Regression**: L1 regularization을 사용한 Lasso 회귀를 적용하여 계수가 0이 되는 feature들을 제거합니다.

- **Tree-based methods**: 의사결정나무나 랜덤 포레스트와 같은 트리 기반 모델을 사용하여 feature의 중요도를 평가하고 중요한 feature들을 선택합니다.

- **차원 축소 (Dimensionality Reduction)**: PCA (주성분 분석)나 t-SNE와 같은 차원 축소 기법을 사용하여 feature들을 잘 요약된 공간으로 압축할 수 있습니다.

- **Embedded methods**: 일부 모델은 feature selection을 내장하고 있습니다. 예를 들어, Lasso나 Elastic Net 회귀는 자체적으로 feature selection을 수행합니다.
