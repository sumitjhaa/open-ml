# More Details from ES 335

> ES 335 Machine Learning, Semester I 2025–26
> Sources: cheat sheets ML01–ML16 and the Anscombe, dummy-variable, hyperparameter, contour and weighted-tree notebooks

**What this file is really about:**
$$\text{plots before statistics}\ \to\ \text{encoding categories}\ \to\ \text{honest metrics}\ \to\ \text{tuning}\ \to\ \text{small facts that matter}$$

**How to use this file.** These are details that fill small gaps in the earlier lectures. Each has an interactive section in the lecture it belongs to.

## Contents

---

# Part A — Maths From the Ground Up

## A1. Anscombe's quartet

Four small datasets share almost the same mean of $x$ and $y$, the same variance, the same correlation of about $0.82$, and the same best-fit line, $y\approx3+0.5x$. Plotted, they look nothing alike: one is a clean line, one is a curve, one has a single outlier, and one is almost vertical with a single point deciding the slope.

**Insight.** Plot the data before trusting a summary statistic.

## A2. Dummy variables

A linear model needs numbers, but colour is a category. Coding red, green and blue as 1, 2 and 3 invents an order and equal spacing that do not exist. Use **one-hot** columns instead, with one column per category.

**Careful.** With $k$ dummies and an intercept, the dummy columns add up to the intercept column, so $X^\top X$ is singular. This is the **dummy variable trap**. Drop one column, which becomes the baseline category. Each remaining coefficient then measures the difference from that baseline.

## A3. Multicollinearity

When two features are nearly equal, the data cannot tell them apart. Predictions stay good, but individual coefficients become unstable: their sum is well determined, and their difference is not. Ridge regression stabilises them.

## A4. Regression metrics

With errors $e_i=y_i-\hat y_i$:

$$\text{MAE}=\frac1n\sum_i|e_i|,\qquad\text{RMSE}=\sqrt{\frac1n\sum_ie_i^2}.$$

MAE treats all errors in proportion, while RMSE punishes large ones more. So one outlier moves RMSE much further. Mean error is not a metric at all, because errors of $+5$ and $-5$ average to zero.

## A5. Metrics for imbalanced classes

**Balanced accuracy** averages the recall of each class:

$$\text{BA}=\frac12\Big(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\Big).$$

**Matthews correlation** uses all four cells, and is 0 for a useless classifier:

$$\text{MCC}=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.$$

## A6. Grid search and random search

Grid search tries every combination of listed values. Random search samples combinations. If only some hyperparameters matter, nine random trials test nine different values of the important one, while a $3\times3$ grid tests only three.

Fit any preprocessing inside each fold. Otherwise information leaks from validation data.

## A7. Contours and gradients

A **contour** joins all points where $f$ takes one value. The gradient at a point is perpendicular to the contour through it, and points uphill. In code, `np.meshgrid` builds the grid of $(x,y)$ pairs at which $f$ is evaluated.

## A8. Weighted samples in trees

Boosting needs trees that respect sample weights. Impurity then uses weighted fractions:

$$p_c=\frac{\sum_{i:y_i=c}w_i}{\sum_iw_i}.$$

## A9. How long a tree takes to fit

For each feature, sorting costs $O(n\log n)$. With $d$ features, one level costs $O(dn\log n)$, and a tree of depth $D$ costs at most about $O(Ddn\log n)$. Prediction takes only $O(D)$ comparisons.

## A10. Subset selection

Trying every subset of $d$ features means $2^d-1$ models. With 8 features that is 255, and with 30 it is over a billion. Forward selection adds the best feature each round, and backward selection removes the worst. Both are greedy and need about $d^2/2$ fits.

## A11. The variance inflation factor, derived

The variance of a coefficient is

$$\text{Var}(\hat\beta_j)=\frac{\sigma^2}{(n-1)\text{Var}(x_j)}\cdot\frac1{1-R_j^2},$$

where $R_j^2$ comes from regressing feature $j$ on all the others. The last factor is the **VIF**. If $R_j^2=0.9$, the variance is 10 times larger, and values above 5 to 10 are commonly treated as a warning.

## A12. Adjusted $R^2$

$R^2$ can only rise as features are added. Adjusting for $p$ features,

$$\bar R^2=1-(1-R^2)\frac{n-1}{n-p-1}.$$

It falls if a new feature adds less than chance would.

## A13. Greedy subset selection

Examining all subsets costs $2^p$ fits. **Forward selection** starts empty, and adds the feature that best improves validation performance, stopping when nothing helps. This needs at most $p(p+1)/2$ fits. It is not guaranteed to find the best subset, because two features may be useful only together.

## A14. Gini with sample weights

Replace counts with sums of weights: $p_k=\sum_{i\in k}w_i/\sum_iw_i$. A split is scored by the weighted average of the children's impurity, with weights being the total weight on each side. Weights of 2 and duplicated rows give the same tree.

---

# Part B — In Code

## B1. One-hot without the trap

```python
pd.get_dummies(df['colour'], drop_first=True)
```

## B2. Random search

```python
from sklearn.model_selection import RandomizedSearchCV
RandomizedSearchCV(model, params, n_iter=20, cv=5).fit(X_train, y_train)
```

## B3. A contour plot

```python
X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
plt.contour(X, Y, X**2 + 3 * Y**2)
```

## B4. Variance inflation factors

```python
import numpy as np

def vif(X):
    out = []
    for j in range(X.shape[1]):
        A = np.c_[np.ones(len(X)), np.delete(X, j, 1)]
        pred = A @ np.linalg.lstsq(A, X[:, j], rcond=None)[0]
        r2 = 1 - ((X[:, j] - pred) ** 2).sum() / ((X[:, j] - X[:, j].mean()) ** 2).sum()
        out.append(1 / (1 - r2))
    return np.array(out)

rng = np.random.default_rng(0)
a, b = rng.normal(size=(2, 500))
X = np.c_[a, b, a + 0.1 * rng.normal(size=500)]
v = vif(X)
print(v.round(1))
assert v[0] > 50 and v[1] < 2
```

## B5. Forward selection

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

X, y = make_regression(300, 20, n_informative=4, noise=5, random_state=0)
chosen, best = [], -np.inf
while True:
    left = [j for j in range(X.shape[1]) if j not in chosen]
    scores = [(cross_val_score(LinearRegression(), X[:, chosen + [j]], y, cv=5).mean(), j) for j in left]
    score, j = max(scores)
    if score <= best + 1e-3:
        break
    best = score; chosen.append(j)

print('chosen', sorted(chosen), 'cv R2', round(best, 3))
assert len(chosen) <= 8
```

## B6. A grid search inside a pipeline

Everything fitted on data, including scaling, sits inside the pipeline, so each fold is clean.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(1000, 20, random_state=0)
Xa, Xb, ya, yb = train_test_split(X, y, random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
search = GridSearchCV(pipe, {'logisticregression__C': [0.01, 0.1, 1, 10]}, cv=5).fit(Xa, ya)
print(search.best_params_, 'test', round(search.score(Xb, yb), 3))
```

---

# Part C — Practice

**P1.** A colour has 4 categories. How many dummy columns do you keep with an intercept?

<details><summary>Answer</summary>

$3$.

</details>

**P2.** Errors are $+3,-3,+4$. Find the MAE and the mean error.

<details><summary>Answer</summary>

MAE $=10/3=3.33$, and the mean error is $4/3=1.33$.

</details>

**P3.** $TP=8$, $FN=2$, $TN=80$, $FP=10$. Find the balanced accuracy.

<details><summary>Answer</summary>

$\tfrac12(0.8+80/90)=0.844$.

</details>

**P4.** A grid has 4 values for each of 3 hyperparameters. How many trials is that?

<details><summary>Answer</summary>

$64$.

</details>

**P5.** How many subsets of 10 features are there, excluding the empty one?

<details><summary>Answer</summary>

$1023$.

</details>

**P6.** Weights are $0.5$ on class A and $0.25$ on each of two class B points. Find the weighted fraction of A.

<details><summary>Answer</summary>

$0.5$.

</details>

**P7.** $R_j^2=0.8$. Find the VIF.

<details><summary>Answer</summary>

$1/0.2=5$.

</details>

**P8.** $R^2=0.5$, $n=21$, $p=10$. Find adjusted $R^2$.

<details><summary>Answer</summary>

$1-0.5\times20/10=0$.

</details>
