# Probability and Statistics in Full

> Extended mathematics notes
> Sources: listed under References

**What this file is really about:**

$$
\text{counting outcomes}\ \to\ \text{distributions}\ \to\ \text{estimates with honest uncertainty}\ \to\ \text{information}
$$


**How to use this file.** This is a reference that starts from the axioms. Each result here is used elsewhere in the course. Part B checks the results by simulation.

<div class="ladder">

## Step-up path

**You need.** Fractions, sums and basic calculus.

1. Use total probability.
2. Compute a mean and variance.
3. Pick a distribution.
4. Build a confidence interval.
5. Compute entropy and KL.

**You can stop when** you can derive any loss from a likelihood.

</div>

## Contents

---

# Part A — Maths From the Ground Up

## A1. Sets, sample spaces and the axioms

The **sample space** $\Omega$ is the set of possible outcomes. An **event** is a subset of it. Probability obeys three **axioms**:

1. $P(A)\ge0$;
2. $P(\Omega)=1$;
3. for disjoint events, $P(A\cup B)=P(A)+P(B)$.

Everything else follows from these. Two results are used constantly: $P(A^c)=1-P(A)$ and $P(A\cup B)=P(A)+P(B)-P(A\cap B)$. The **union bound** follows: $P(\bigcup_iA_i)\le\sum_iP(A_i)$.

## A2. Counting

A **factorial** counts orderings of $n$ items: $n!$. The number of ordered selections of $k$ is $n!/(n-k)!$. The **binomial coefficient** counts unordered selections:


$$
\binom nk=\frac{n!}{k!\,(n-k)!}.
$$


## A3. Conditional probability and independence


$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}.
$$


$A$ and $B$ are **independent** if $P(A\cap B)=P(A)P(B)$. The **law of total probability** splits an event over a partition $B_i$:


$$
P(A)=\sum_iP(A\mid B_i)\,P(B_i).
$$


**Bayes' rule** combines these: $P(B_i\mid A)=P(A\mid B_i)P(B_i)/P(A)$.

## A4. Random variables, mass, density and CDF

A **random variable** assigns a number to each outcome. A discrete one has a **probability mass function** $p(x)=P(X=x)$. A continuous one has a **density** $p(x)$, where $P(a<X<b)=\int_a^bp$. A density can exceed 1, but it must integrate to 1. The **cumulative distribution function** is $F(x)=P(X\le x)$. It rises from 0 to 1, and $F'=p$. If $U$ is uniform, then $F^{-1}(U)$ has CDF $F$, which is inverse-CDF sampling.

A **joint distribution** $p(x,y)$ gives both at once. The **marginal** sums out the other variable, $p(x)=\int p(x,y)\,dy$.

## A5. Expectation

$\mathbb E[X]=\sum_xxp(x)$, or $\int xp(x)\,dx$. Three properties do most of the work.

- **Linearity of expectation**: $\mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y]$, whether or not they are independent.
- **Law of the unconscious statistician**: $\mathbb E[g(X)]=\int g(x)p(x)\,dx$, with no need for the density of $g(X)$.
- **Law of total expectation**, or the tower property: $\mathbb E[X]=\mathbb E[\mathbb E[X\mid Y]]$.

## A6. Variance, covariance and correlation


$$
\text{Var}(X)=\mathbb E[(X-\mu)^2]=\mathbb E[X^2]-\mu^2,\qquad\text{Cov}(X,Y)=\mathbb E[XY]-\mathbb E[X]\mathbb E[Y].
$$


The rules are $\text{Var}(aX+b)=a^2\text{Var}(X)$ and $\text{Var}(X+Y)=\text{Var}(X)+\text{Var}(Y)+2\text{Cov}(X,Y)$. **Correlation** divides by both standard deviations and lies in $[-1,1]$. Independence implies zero covariance, but zero covariance does not imply independence: take $Y=X^2$ with symmetric $X$.

The **law of total variance** splits variance into explained and unexplained parts:


$$
\text{Var}(X)=\mathbb E[\text{Var}(X\mid Y)]+\text{Var}(\mathbb E[X\mid Y]).
$$


## A7. Discrete distributions

| Name | $p(x)$ | Mean | Variance | Models |
|---|---|---|---|---|
| Bernoulli($p$) | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | One binary label |
| Binomial($n,p$) | $\binom nxp^x(1-p)^{n-x}$ | $np$ | $np(1-p)$ | Successes in $n$ trials |
| Categorical | $\prod_kp_k^{[x=k]}$ | | | One class label |
| Multinomial | $\frac{n!}{\prod x_k!}\prod p_k^{x_k}$ | $np_k$ | $np_k(1-p_k)$ | Word counts |
| Geometric($p$) | $(1-p)^{x-1}p$ | $1/p$ | $(1-p)/p^2$ | Trials until a success |
| Poisson($\lambda$) | $e^{-\lambda}\lambda^x/x!$ | $\lambda$ | $\lambda$ | Counts per interval |

## A8. Continuous distributions

| Name | Density | Mean | Variance | Models |
|---|---|---|---|---|
| Uniform($a,b$) | $1/(b-a)$ | $(a+b)/2$ | $(b-a)^2/12$ | No preference |
| **Exponential distribution**($\lambda$) | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | Waiting time |
| Normal($\mu,\sigma^2$) | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | Sums of many effects |
| **Laplace distribution**($\mu,b$) | $\frac1{2b}e^{-|x-\mu|/b}$ | $\mu$ | $2b^2$ | L1 loss and lasso prior |
| **Gamma distribution**($\alpha,\beta$) | $\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$ | $\alpha/\beta$ | $\alpha/\beta^2$ | Positive quantities, rate priors |
| **Beta distribution**($a,b$) | $\frac{x^{a-1}(1-x)^{b-1}}{B(a,b)}$ | $\frac a{a+b}$ | $\frac{ab}{(a+b)^2(a+b+1)}$ | A probability |
| **Log-normal** | $\ln X$ is normal | $e^{\mu+\sigma^2/2}$ | | Positive skewed values |
| **Student's t**($\nu$) | Heavy tails | $0$ for $\nu>1$ | $\nu/(\nu-2)$ for $\nu>2$ | Means with unknown variance |
| **Chi-squared**($k$) | Sum of $k$ squared normals | $k$ | $2k$ | Variances and goodness of fit |
| **Dirichlet**($\alpha$) | $\propto\prod_kx_k^{\alpha_k-1}$ | $\alpha_k/\sum\alpha$ | | A probability vector |

![The common densities have very different shapes and tails.](images/G05/figures/density_families.svg)

## A9. Conjugate priors and the exponential family

A prior is **conjugate** if the posterior has the same form.

| Likelihood | Prior | Posterior |
|---|---|---|
| Bernoulli | Beta($a,b$) | Beta($a+s,\ b+f$) |
| Categorical | Dirichlet($\alpha$) | Dirichlet($\alpha+\text{counts}$) |
| Poisson | Gamma($\alpha,\beta$) | Gamma($\alpha+\sum x,\ \beta+n$) |
| Normal with known variance | Normal | Normal, combining precisions |

The **exponential family** has the form $p(x\mid\eta)=h(x)\exp(\eta^\top T(x)-A(\eta))$. Bernoulli, Poisson, normal and many others belong. $\nabla A(\eta)=\mathbb E[T(x)]$, and $T$ is a **sufficient statistic**: once you know it, the rest of the data adds nothing about $\eta$. Generalised linear models are built on this family.

## A10. Moment generating functions

$M(t)=\mathbb E[e^{tX}]$ produces moments by differentiation, $M^{(k)}(0)=\mathbb E[X^k]$. For independent variables, $M_{X+Y}=M_XM_Y$. This is the standard tool for proving the central limit theorem and exponential tail bounds.

## A11. Inequalities

**Markov's inequality**: for $X\ge0$, $P(X\ge a)\le\mathbb E[X]/a$.

**Chebyshev's inequality**: $P(|X-\mu|\ge k\sigma)\le1/k^2$.

**Hoeffding's inequality**: for independent variables in $[0,1]$ with sample mean $\bar X$,


$$
P(|\bar X-\mu|\ge t)\le2e^{-2nt^2}.
$$


Tail bounds of this kind are the basis of generalisation theory, confidence bounds for bandits and sample size calculations.

## A12. Law of large numbers and central limit theorem

The **law of large numbers**: the mean of $n$ independent copies converges to $\mu$.

The **central limit theorem**: for finite variance,


$$
\sqrt n\,(\bar X-\mu)/\sigma\to\mathcal N(0,1).
$$


This is why errors of averages are often normal, and why the **standard error** is $\sigma/\sqrt n$.

## A13. Order statistics

Sort a sample as $X_{(1)}\le\dots\le X_{(n)}$. The maximum of $n$ independent copies has CDF $F(x)^n$. The minimum has $1-(1-F(x))^n$. This is why, in high dimensions, nearest and farthest neighbours come to have similar distances, and why the largest of many noisy estimates is biased upwards.

## A14. Estimators

An **estimator** is a function of the sample. Its bias is $\mathbb E[\hat\theta]-\theta$. It is **unbiased** if that is zero, and **consistent** if it converges to $\theta$ as $n$ grows. Mean squared error decomposes as $\text{bias}^2+\text{variance}$.

The sample variance divides by $n-1$, because using $\bar x$ in place of $\mu$ removes one degree of freedom.

**Maximum likelihood** picks $\arg\max_\theta\sum_i\ln p(x_i\mid\theta)$. It is consistent under regular conditions and its asymptotic variance is the inverse Fisher information. **MAP** adds a log prior.

## A15. Confidence intervals and bootstrap

A 95% **confidence interval** procedure captures the true value in 95% of repeated samples. It is not a probability that this particular interval contains the value. For a mean it is $\bar x\pm t\cdot s/\sqrt n$.

The **bootstrap** resamples the data with replacement and recomputes the statistic. It gives intervals for quantities with no simple formula.

**Kernel density estimation** estimates a density by placing a small bump at each point:


$$
\hat p(x)=\frac1{nh}\sum_iK\Big(\frac{x-x_i}h\Big).
$$


The bandwidth $h$ trades bias against variance.

## A16. Hypothesis tests

A test sets a null hypothesis, computes a statistic and reports a **p-value**: the probability under the null of a result at least as extreme. It is not the probability that the null is true.

| Error | Meaning | Rate |
|---|---|---|
| Type I | Rejecting a true null | $\alpha$ |
| Type II | Missing a real effect | $\beta$ |

The **power of a test** is $1-\beta$. It rises with effect size and sample size.

- **t-test**: compare means, $t=(\bar x-\mu_0)/(s/\sqrt n)$.
- **Chi-squared test**: compare counts, $\sum(O-E)^2/E$.
- **ANOVA**: compare several means, using between-group variance over within-group variance.

**Multiple testing**. Running $m$ tests at level $\alpha$ inflates false alarms. The **Bonferroni** correction tests each at $\alpha/m$.

## A17. Stochastic processes and Markov chains

A **stochastic process** is a family of random variables indexed by time. A **Markov chain** has a future that depends only on the present. With **transition matrix** $P_{ij}=P(x_{t+1}=j\mid x_t=i)$, a distribution evolves by $\pi_{t+1}=\pi_tP$. A **stationary distribution** satisfies $\pi=\pi P$.

A **martingale** satisfies $\mathbb E[X_{t+1}\mid\text{history}]=X_t$, a fair game with no drift. It is the basis of concentration results for dependent data.

## A18. Information theory

| Quantity | Definition |
|---|---|
| Entropy | $H(X)=-\sum_xp\log p$ |
| **Joint entropy** | $H(X,Y)=-\sum p(x,y)\log p(x,y)$ |
| **Conditional entropy** | $H(Y\mid X)=H(X,Y)-H(X)$ |
| Mutual information | $I(X;Y)=H(Y)-H(Y\mid X)$ |
| Cross-entropy | $H(p,q)=-\sum p\log q$ |
| **Kullback-Leibler divergence** | $\text{KL}(p\|q)=\sum p\log(p/q)$ |
| Jensen-Shannon | $\tfrac12\text{KL}(p\|m)+\tfrac12\text{KL}(q\|m)$, with $m=\tfrac12(p+q)$ |

Logs in base 2 give bits, and natural logs give nats.

**Gibbs' inequality** states $\text{KL}(p\|q)\ge0$, with equality only when $p=q$. It follows from Jensen. So cross-entropy equals $H(p)+\text{KL}(p\|q)$, and minimising it over $q$ matches the distribution. KL is not symmetric.

The **data processing inequality**: if $X\to Y\to Z$ is a Markov chain, then $I(X;Z)\le I(X;Y)$. Processing cannot add information about $X$.

Between two normals with the same dimension $d$,


$$
\text{KL}(\mathcal N_0\|\mathcal N_1)=\tfrac12\Big[\text{Tr}(\Sigma_1^{-1}\Sigma_0)+(\mu_1-\mu_0)^\top\Sigma_1^{-1}(\mu_1-\mu_0)-d+\ln\frac{\det\Sigma_1}{\det\Sigma_0}\Big].
$$


---

# Part B — In Code

## B1. Total probability, linearity and total variance

```python
import numpy as np
rng = np.random.default_rng(0)
n = 2_000_000
g = rng.integers(0, 2, n); y = rng.normal(np.where(g, 3.0, 0.0), np.where(g, 2.0, 1.0))
within = 0.5 * 4 + 0.5 * 1; between = 0.25 * 9     # variance of the conditional means 0 and 3
print(round(y.var(), 3), within + between)
assert abs(y.var() - (within + between)) < 0.02
x = rng.normal(size=n); z = x**2
assert abs(np.mean(x * z)) < 0.01        # zero covariance, yet z depends on x
assert abs(np.mean(x + 3 * z) - 3) < 0.01
```

## B2. Distribution summaries

```python
import numpy as np
rng = np.random.default_rng(1); n = 3_000_000
checks = [('geometric', rng.geometric(0.25, n), 4, 12), ('poisson', rng.poisson(3.5, n), 3.5, 3.5),
          ('exponential', rng.exponential(2.0, n), 2, 4), ('gamma', rng.gamma(3, 0.5, n), 1.5, 0.75),
          ('beta', rng.beta(2, 5, n), 2/7, 10/(49*8)), ('laplace', rng.laplace(0, 2, n), 0, 8),
          ('chi-squared', rng.chisquare(4, n), 4, 8)]
for name, s, m, v in checks:
    assert abs(s.mean() - m) < 0.02 and abs(s.var() - v) < 0.05 * max(v, 1), name
s = rng.lognormal(0.2, 0.5, n); assert abs(s.mean() - np.exp(0.2 + 0.125)) < 0.01
print('all distribution formulas agree')
```

## B3. Tail bounds really are bounds

```python
import numpy as np
rng = np.random.default_rng(2)
x = rng.exponential(1.0, 3_000_000)
for a in (2, 4): assert np.mean(x >= a) <= 1 / a
for k in (2, 3): assert np.mean(np.abs(x - 1) >= k) <= 1 / k**2
n, t = 100, 0.1
means = rng.binomial(n, 0.5, 2_000_000) / n
actual = np.mean(np.abs(means - 0.5) >= t); bound = 2 * np.exp(-2 * n * t * t)
print(round(actual, 4), round(bound, 4)); assert actual <= bound
```

## B4. The central limit theorem and the maximum

```python
import numpy as np
rng = np.random.default_rng(3)
u = rng.random((400_000, 30)); z = (u.mean(1) - 0.5) / (np.sqrt(1/12) / np.sqrt(30))
assert abs(np.mean(np.abs(z) < 1.96) - 0.95) < 0.003
m = u.max(1); assert abs(np.mean(m < 0.9) - 0.9**30) < 0.002
print('CLT and order statistics agree')
```

## B5. A confidence interval has the coverage it claims

```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(4)
n = 12; R = 40000; t = stats.t.ppf(0.975, n - 1)
x = rng.normal(5, 2, (R, n))
m, s = x.mean(1), x.std(1, ddof=1)
cover = np.mean(np.abs(m - 5) <= t * s / np.sqrt(n))
print(round(cover, 4)); assert abs(cover - 0.95) < 0.006
p = rng.random((R, 20)); assert abs(np.mean(p.min(1) < 0.05) - (1 - 0.95**20)) < 0.01
assert np.mean(p.min(1) < 0.05 / 20) < 0.06
```

## B6. Information identities

```python
import numpy as np
P = np.array([[0.3, 0.1], [0.2, 0.4]])
h = lambda p: -np.sum(p[p > 0] * np.log2(p[p > 0]))
Hxy, Hx, Hy = h(P.ravel()), h(P.sum(1)), h(P.sum(0))
Hy_x = Hxy - Hx; I = Hy - Hy_x
assert I >= 0 and np.isclose(I, np.sum(P * np.log2(P / np.outer(P.sum(1), P.sum(0)))))
p, q = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.2, 0.6])
kl = np.sum(p * np.log2(p / q)); assert kl > 0 and np.isclose(-np.sum(p * np.log2(q)), h(p) + kl)
assert not np.isclose(kl, np.sum(q * np.log2(q / p)))
print(round(I, 4), round(kl, 4))
```

---

# Part C — Practice

**P1.** A 1% disease has a test with 90% sensitivity and a 5% false-positive rate. What is $P(\text{positive})$?

<details><summary>Answer</summary>

$0.01\times0.9+0.99\times0.05=0.0585$.

</details>

**P2.** What is the mean number of die rolls until the first six?

<details><summary>Answer</summary>

Geometric with $p=1/6$, so $6$.

</details>

**P3.** A Beta(2, 2) prior sees 3 heads and 1 tail. What is the posterior mean?

<details><summary>Answer</summary>

Beta(5, 3), with mean $5/8$.

</details>

**P4.** A non-negative variable has mean 2. Bound $P(X\ge10)$.

<details><summary>Answer</summary>

By Markov, at most $0.2$.

</details>

**P5.** What is the standard error of a mean with $\sigma=10$ and $n=100$?

<details><summary>Answer</summary>

$1$.

</details>

**P6.** Twenty tests are run at $\alpha=0.05$, and every null is true. What is the chance of at least one false alarm if they are independent?

<details><summary>Answer</summary>

$1-0.95^{20}\approx0.64$.

</details>

**P7.** Does zero mutual information mean independence?

<details><summary>Answer</summary>

Yes. $I(X;Y)=\text{KL}(p(x,y)\|p(x)p(y))$ is zero only when they are equal.

</details>
