# Anomaly Detection

> ES 335 / ES 678 Machine Learning, Sem I 2026
> Sources: `sources/course-2026/lectures/L13b_unsupervised_learning_handbook.pdf` (Chapter 5, pp. 78–89 of the file)

**What this file is really about:**

$$
\text{describe usual}\ \to\ \text{score how badly a point fits}\ \to\ \text{choose what "fits" means}\ \to\ \text{evaluate with few labels}
$$


**How to use this file.** One feature uses the seven latencies from the mixture note plus a slow request of 95 ms. Two features use the eight training runs plus two new runs, $P=(2,8)$ and $Q=(11,11)$. Part C is practice with answers.

<div class="ladder">

## Step-up path

**You need.** Mean and standard deviation, the median, and PCA.

1. Name the kind of unusual you are looking for.
2. See how an anomaly hides itself from the mean, and use robust scores.
3. Bring in the shape of the data with Mahalanobis distance.
4. Score by neighbours and by isolation, where no shape is assumed.
5. Evaluate a ranking with few labels, and monitor a deployed model for drift.

**You can stop when** you can choose a score for a given kind of anomaly and evaluate it honestly.

</div>

## Contents

---

# Part A — Maths From the Ground Up

## A1. What an anomaly is

An **anomaly** is an example that differs markedly from the bulk of the data: a run that diverges, a request whose latency spikes, a corrupted input. A card used in Ahmedabad and, an hour later, in Delhi is suspicious because it does not fit the card's usual pattern.

- **Point** anomalies are single unusual examples.
- **Contextual** anomalies are unusual given their context, like a latency normal at peak load but not at night.
- **Collective** anomalies are unusual only together, like a sequence of small failures.

The task is hard for three reasons: anomalies are rare, so labels are scarce; "unusual" depends on a model of usual; and anomalies distort the statistics used to find them. **Masking** is when an anomaly inflates the estimated spread and hides itself. **Swamping** is when it shifts the centre so normal points appear unusual.

Every method assigns an **anomaly score** and flags scores above a threshold. They differ in what the score measures.

## A2. Standard scores, and a limit you can prove

The **standard score** is $z=(x-\bar x)/s$, and a common rule flags $|z|>3$.

**Samuelson's inequality.** For $n$ numbers, with $s$ computed using $1/n$, every value satisfies


$$
|x_i-\bar x|\le s\sqrt{n-1}.
$$


**Why.** The deviations sum to zero, so $x_i-\bar x=-\sum_{j\ne i}(x_j-\bar x)$. Applying Cauchy-Schwarz to the $n-1$ remaining terms gives $(x_i-\bar x)^2\le(n-1)\big(ns^2-(x_i-\bar x)^2\big)$, which rearranges to the result.

So with $n\le10$ values, $|z|\le\sqrt9=3$ and **no value can ever exceed the rule**.

## A3. Masking, and robust scores

Take the eight latencies $10,12,14,22,31,33,36,95$. The mean is $31.625$ and $s=25.724$, so the 95 ms request has $z=2.464$, below 3. Computed from the other seven alone, its score would be $7.22$. The slow request has hidden itself.

The **median** and the **median absolute deviation** barely notice one bad value. Sorted, the median is $(22+31)/2=26.5$. The absolute deviations have median $(9.5+12.5)/2$, so MAD $=11$.


$$
\text{robust score}=\frac{0.6745\,(x-\text{median})}{\text{MAD}}.
$$


For 95 this gives $0.6745\times68.5/11=4.20$, which is flagged. The factor $0.6745$ makes the MAD comparable with $\sigma$ for normal data.

![Three rows of latencies showing the bands of mean plus or minus three standard deviations, with and without the outlier, and the robust median band.](images/L17/figures/masking_bands.svg)
*Masking. The band holds values that would not be flagged. With all eight values the slow request widens its own band enough to escape. Without it, or with the median and MAD, it is flagged.*

## A4. Scoring by density

Fit a model of normal data, and score by how improbable the point is, $-\log p(x)$. Work with log-densities to avoid underflow: under the two-component mixture from the mixture note, $p(95)=e^{-123.90}=1.555\times10^{-54}$.

Set the threshold at a low quantile of training log-densities, for example the 1st percentile, which fixes the false-alarm rate on normal data.

**Careful.** The method is only as good as its model. A single Gaussian fitted to two clusters assigns high density to the empty region between them.

## A5. Two features: when distance misleads

A person both tall and heavy is unremarkable. A person very tall and very light is unusual, though neither number alone is extreme. Four scores for the new runs:

| score | $P=(2,8)$ | $Q=(11,11)$ |
|---|---|---|
| Euclidean distance from the mean | 4.002 | 8.048 |
| Mahalanobis distance | 4.316 | 2.351 |
| distance from the first principal line | 3.950 | 0.144 |
| mean distance to the 2 nearest runs | 4.236 | 3.924 |

Euclidean distance ranks $Q$ as twice as unusual. The others reverse that, because $Q$ lies along the trend of the data.

## A6. Mahalanobis distance


$$
d_M(x)=\sqrt{(x-\bar x)^\top S^{-1}(x-\bar x)}.
$$


**In principal coordinates.** Using the spectral theorem, with $z_j=v_j^\top(x-\bar x)$,


$$
d_M(x)^2=\sum_j\frac{z_j^2}{\lambda_j}.
$$


It is the Euclidean distance after rotating to principal axes and dividing each by its standard deviation, which is called **whitening**. A step along a direction of small variance counts for much more.

**Worked example.** For $P$, $z_1=-0.643$ and $z_2=3.950$. So $d_M^2=0.0352+18.5962=18.631$ and $d_M=4.316$. Almost all of it comes from the second direction. For $Q$, $z_1=8.046$ and $z_2=0.144$, so $d_M=2.351$.

**Invariance.** Under any invertible transformation $x\mapsto Ax+b$, Mahalanobis distances are unchanged. So changing hours to minutes changes nothing, unlike K-means.

![Two panels of the eight runs with contours of Euclidean and Mahalanobis distance, and the new runs P and Q.](images/L17/figures/euclid_mahalanobis.svg)
*Euclidean distance grows equally in every direction and ranks $Q$ as more unusual. Mahalanobis distance grows slowly along the data's trend and quickly across it, and ranks $P$ as more unusual.*

## A7. Reconstruction and neighbour scores

**Distance from a subspace.** $|z_2|$ is the part of the Mahalanobis distance from the minor direction, without dividing by $\sqrt{\lambda_2}$. With many features, the reconstruction error after keeping $k$ components measures how far a point is from where normal data live.

**Distance to nearest neighbours.** Average the distance to the $k$ nearest training examples. For $P$ with $k=2$: $(4+4.4721)/2=4.236$. It assumes no centre and no elliptical shape, so it also works for several clusters, at the cost of a nearest-neighbour search per query.

## A8. Isolation forest

In twenty questions, a 2.1 m visitor is identified quickly, and an average person takes many questions. An **isolation tree** repeatedly chooses a random feature and a random cut, until each point is alone. Anomalies are isolated near the root.

The average path length of a point in a random tree on $m$ points is


$$
c(m)=2H_{m-1}-\frac{2(m-1)}{m},\qquad H_m=1+\tfrac12+\dots+\tfrac1m.
$$


So $c(2)=1$, $c(8)=3.436$ and $c(256)=10.249$. The **score** is


$$
s(x)=2^{-\bar h(x)/c(\psi)},
$$


where $\bar h$ is the mean depth over the trees and $\psi$ is the subsample size. Scores near 1 are anomalies. On the eight latencies, 95 ms has mean depth 1.35 and score 0.753, while the others score at most 0.485.

It needs no distances, density model or scaling, so it is a good first method on tables of features.

![Two panels of random axis-aligned cuts: an outlier isolated after a few cuts, and a typical point needing many.](images/L17/figures/isolation_cuts.svg)
*Random cuts of one isolation tree, followed until the circled point is alone. The outlier is separated in a few cuts, and a point in the middle needs many more.*

## A9. Evaluating with few labels

In practice a small set of known anomalies is collected. The table below is the handbook's experiment on its own two datasets, and the interactive version regenerates similar data, so its numbers differ slightly.

**Precision at $k$** is the fraction of anomalies among the top $k$ scores. **Average precision** averages over the known anomalies at ranks $r_1<\dots<r_a$:


$$
\text{AP}=\frac1a\sum_{j=1}^a\frac{j}{r_j}.
$$


A perfect ranking has AP 1, and a random one has AP near the base rate $a/n$.

| score | A: elongated, $P@20$ | A: AP | B: two clusters, $P@10$ | B: AP |
|---|---|---|---|---|
| Euclidean from the mean | 0.00 | 0.070 | 0.00 | 0.011 |
| Mahalanobis | 0.70 | 0.847 | 0.20 | 0.188 |
| 5th nearest neighbour | 0.15 | 0.202 | 0.90 | 0.991 |
| isolation forest | 0.10 | 0.207 | 0.50 | 0.580 |

**Insight.** No score is best in general. Mahalanobis wins when anomalies differ in the direction of small variance. A neighbour score wins when normal data form two clusters, since a single ellipse covers the gap between them.

## A10. Choosing a method

Start from what kind of unusual you expect. Use robust scores for a single feature, Mahalanobis for one elongated cloud, neighbours for several clusters, isolation forest as a fast first attempt on tables, and reconstruction error for high-dimensional data on a low-dimensional subspace.

## A11. Anomaly detection in deployed systems

**Drift.** A deployed model can fail silently when its inputs stop resembling its training data. Monitoring treats each recent batch as a possible anomaly against a reference set, by:

- the fraction of recent inputs whose nearest-neighbour distance exceeds a high quantile of the reference's own distances;
- a statistical test of whether the two samples come from the same distribution;
- the accuracy of a classifier trained to tell recent from reference inputs, which stays near one half when nothing has changed.

Thresholds must weigh the cost of false alarms, since frequent spurious alerts are soon ignored. Distance-based scores over millions of vectors use approximate nearest-neighbour search.

---

# Part C — Practice

**P1.** With $n=10$ values, what is the largest possible $|z|$?

<details><summary>Answer</summary>

$\sqrt{n-1}=3$, so the $|z|>3$ rule can never fire.

</details>

**P2.** For $1,2,3,4,20$, compute the standard score and robust score of 20.

<details><summary>Answer</summary>

Mean 6, variance $(25+16+9+4+196)/5=50$, so $z=14/7.07=1.98$. Median 3, absolute deviations $2,1,0,1,17$, MAD 1, so the robust score is $0.6745\times17=11.47$.

</details>

**P3.** A point has scores $z=(3,1)$ with $\lambda=(9,1)$. Find its Mahalanobis distance.

<details><summary>Answer</summary>

$d^2=9/9+1/1=2$, so $d=1.414$.

</details>

**P4.** Find $c(4)$, and the score of a point with mean depth 1 when $\psi=4$.

<details><summary>Answer</summary>

$c(4)=2(1+\tfrac12+\tfrac13)-\tfrac{6}{4}=2.167$. So $s=2^{-1/2.167}=0.726$.

</details>

**P5.** Anomalies are at ranks 1, 3 and 6. Find the AP and the precision at 3.

<details><summary>Answer</summary>

AP $=(1/1+2/3+3/6)/3=0.722$. Precision at 3 is $2/3$.

</details>

**P6.** Why does a single Gaussian density fail on two clusters?

<details><summary>Answer</summary>

Its centre sits in the empty gap, so points in the gap get high density even though no normal data are there.

</details>

**P7.** Prove the Mahalanobis distance is unchanged when one feature is converted from hours to minutes.

<details><summary>Answer</summary>

With $x\mapsto Ax$, the deviation becomes $Ad$ and the covariance $ASA^\top$. So $d^\top A^\top(ASA^\top)^{-1}Ad=d^\top S^{-1}d$.

</details>
