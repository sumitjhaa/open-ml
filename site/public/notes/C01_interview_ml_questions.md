# Machine Learning Interview Questions

> Career preparation notes
> Sources: standard material, listed under References

**What this file is really about:**

$$
\text{what you are asked}\ \to\ \text{a precise answer}\ \to\ \text{the follow-up they will ask next}
$$


**How to use this file.** Cover the answer, say yours aloud, then compare. Answer in this order: the idea, the formula, one example, and one limitation.

<div class="ladder">

## Step-up path

**You need.** The core machine learning sections.

1. Answer the fundamentals precisely.
2. Derive the standard results.
3. Discuss models and their limits.
4. Implement classics from scratch.
5. Handle the follow-up questions.

**You can stop when** you can answer standard technical questions under pressure.

</div>

## Contents

---

# Part A — Fundamentals

## A1. Learning and evaluation

**What is the bias-variance trade-off?** Expected squared error splits into bias squared, variance and noise. Simple models have high bias, flexible models high variance. Follow-up: bagging mainly reduces variance, and boosting mainly reduces bias.

**Training error is low, test error high. What do you do?** It is overfitting. Get more data, regularise, simplify the model, or use ensembling, and first check for leakage and data mismatch.

**Accuracy is 99%. Is the model good?** It depends on the base rate. With 1% positives, predicting all negative gets 99%. Report precision, recall, and PR-AUC.

**ROC-AUC or PR-AUC?** ROC-AUC is insensitive to class balance, so it can look good on rare events. PR-AUC reflects performance on the rare class.

**What is data leakage?** Information unavailable at prediction time reaches training, as with target encoding on all the data, or scaling before splitting. Fit preprocessing inside each fold.

**How should you split time-series data?** By time, using rolling origins. Random splits train on the future.

## A2. Linear and logistic models

**Derive the least-squares solution.** Minimise $\|y-X\theta\|^2$. The gradient $-2X^\top(y-X\theta)$ set to zero gives $\theta=(X^\top X)^{-1}X^\top y$.

**Why does L1 give sparse weights and L2 does not?** The L1 ball has corners on the axes, so solutions land where weights are zero. Algebraically, the L1 subgradient gives soft thresholding.

**Why not use squared loss for classification?** With a sigmoid, it is non-convex and gives vanishing gradients on confident errors. Log loss is convex and punishes confident mistakes.

**What does a logistic regression coefficient mean?** A one-unit increase adds $w$ to the log-odds, and multiplies the odds by $e^w$.

**How do correlated features affect linear regression?** Predictions stay fine, while coefficients become unstable.

## A3. Trees and ensembles

**How does a tree choose a split?** The split that most reduces impurity, using Gini or entropy for classes, and variance for regression.

**Random forest or boosted trees?** Forests are robust with little tuning, and reduce variance. Boosting usually scores higher, but needs tuning and can overfit noise.

**How does gradient boosting work?** Each new tree fits the negative gradient of the loss at the current predictions.

**How does XGBoost differ?** It uses second derivatives, and regularises tree size and leaf weights.

**Are tree feature importances trustworthy?** Impurity importance favours high-cardinality features. Prefer permutation importance on held-out data.

## A4. Other classics

**What is the kernel trick?** Replacing dot products with a kernel gives the effect of high-dimensional features, without computing them.

**What are support vectors?** Points on or inside the margin. Only these determine the boundary.

**K-means assumptions and failures?** Round, similar-sized clusters. It fails on rings, moons and strongly unequal groups.

**PCA versus t-SNE?** PCA is linear and preserves variance, with interpretable axes. t-SNE preserves neighbours, and its global distances are not meaningful.

**Why standardise features?** For distance- and gradient-based methods. Trees do not need it.

## A5. Probability and statistics

**State Bayes' rule, and give a use.** $P(A\mid B)=P(B\mid A)P(A)/P(B)$. A rare disease with a good test still gives mostly false positives.

**MLE versus MAP?** MLE maximises the likelihood. MAP adds a prior, and a Gaussian prior is ridge.

**What does a p-value mean?** The probability, under the null, of data at least this extreme. It is not the probability the null is true.

**What is the central limit theorem?** Sums of independent variables with finite variance tend to a normal distribution, so sample means are approximately normal.

![The value of a positive result rises steeply with how common the condition is.](images/C01/figures/base_rate.svg)

## A6. Neural networks and deep learning

**Why are non-linearities needed?** Otherwise a stack of layers is one linear map.

**Explain backpropagation.** Reverse-mode automatic differentiation: one backward sweep applying the chain rule gives all gradients.

**What causes vanishing gradients, and what are the fixes?** Multiplying many small derivatives. Fixes are ReLU, residual connections, normalisation and good initialisation.

**What does batch norm do?** It normalises activations using batch statistics, which allows higher learning rates. It behaves differently at training and inference.

**Why does dropout work?** It prevents co-adaptation, acting like averaging many thinned networks.

**Adam or SGD?** Adam converges quickly with little tuning. SGD with momentum can generalise slightly better on some vision tasks.

**How does attention work?** Each token forms a query against every key, and a softmax over their scores mixes the values.

**Why do transformers need positional information?** Attention treats tokens as a set.

**What is the difference between a CNN and a ViT?** A CNN has locality and translation equivariance built in, while a ViT learns relationships and needs more data.

**How does diffusion work?** Add noise step by step, train a network to predict that noise, and generate by removing it from pure noise.

**What are hallucinations, and how do you reduce them?** Confident invented output. Use retrieval, require citations, and evaluate groundedness.

---

# Part B — In Code

## B1. Implement K-means

```python
import numpy as np

def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        labels = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        new = np.array([X[labels == j].mean(0) if (labels == j).any() else C[j] for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    return C, labels
```

## B2. Implement logistic regression

```python
def fit_logistic(X, y, lr=0.1, steps=1000):
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X @ w))
        w -= lr * X.T @ (p - y) / len(y)
    return w
```

## B3. Implement a decision stump

```python
def gini(y):
    p = np.bincount(y) / len(y)
    return 1 - (p ** 2).sum()

def best_split(x, y):
    best = (None, np.inf)
    for t in np.unique(x)[:-1]:
        left, right = y[x <= t], y[x > t]
        score = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
        if score < best[1]:
            best = (t, score)
    return best
```

## B4. Precision, recall and AUC from scratch

```python
def precision_recall(y, pred):
    tp = ((pred == 1) & (y == 1)).sum()
    return tp / max(pred.sum(), 1), tp / max(y.sum(), 1)

def roc_auc(y, score):
    order = np.argsort(score)
    ranks = np.empty(len(y)); ranks[order] = np.arange(1, len(y) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
```

---

# Part C — Practice

**P1.** Precision 0.5, recall 1. Find the F1 score.

<details><summary>Answer</summary>

$2\times0.5\times1/1.5=0.667$.

</details>

**P2.** A model with 100 features has perfect training accuracy on 50 rows. What would you suspect?

<details><summary>Answer</summary>

Overfitting, since there are more parameters than examples. Check validation performance, and regularise.

</details>

**P3.** What is the expected number of unique items in a bootstrap sample of size $n$?

<details><summary>Answer</summary>

About $0.632n$.

</details>

**P4.** A 3-class problem has a starting loss of 1.10. Is it healthy?

<details><summary>Answer</summary>

Yes. $\ln3=1.10$.

</details>
