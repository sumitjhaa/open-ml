# Clustering with K-means

> ES 335 / ES 678 Machine Learning, Sem I 2026 — Prof. Udit Bhatia
> Sources: `Lectures/L13_kmeans_annotated_slides.pdf` (35 slides), `Lectures/L13a_kmeans_handwritten_class_notes.pdf` (3 pages), `Lectures/L13b_unsupervised_learning_handbook.pdf` (Chapter 1 and Chapter 2)

**What this file is really about:**
$$\text{no labels}\ \to\ \text{define "together"}\ \to\ \text{minimise }J\ \to\ \text{start well}\ \to\ \text{choose }K\ \to\ \text{know when it fails}$$

**How to use this file.** Everything runs on one small dataset, eight training runs, and every number can be checked by hand. Part A builds the mathematics from the start. Part B walks through all 35 slides and the handwritten notes. Part C is practice with answers.

**The one dataset for the whole lecture** ($x_1$ is training time in hours, $x_2$ is peak GPU memory in GB):

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $x_1$ | 4 | 4 | 2 | 1 | 8 | 8 | 8 | 6 |
| $x_2$ | 4 | 3 | 3 | 3 | 8 | 6 | 9 | 8 |

## Contents

---

# Part A — Maths From the Ground Up

## A1. Learning without labels

In **supervised** learning each example comes with an answer: we are given pairs $(x_i,y_i)$ and learn a function $f(x)\to y$. In **unsupervised** learning we are given only the $x_i$, and we ask which examples belong together.

There is no $y$, so there is no error to measure against a correct answer. A child sorting LEGO might sort by colour, size or shape, and each is a reasonable answer. The method has to be told what "together" means, and that choice is the whole model.

An example is a list of numbers called **features**, so it is a point in a space with one axis per feature. A **cluster** is a group of points that lie close together in that space.

## A2. Distance between two points

The straight-line or **Euclidean distance** between $x$ and $y$ comes from Pythagoras, applied one coordinate at a time:

$$\|x-y\|=\sqrt{\sum_{j}(x_j-y_j)^2}.$$

**Example.** Runs 1 and 8 are $(4,4)$ and $(6,8)$. The differences are $2$ and $4$, so the squared distance is $4+16=20$ and the distance is $\sqrt{20}\approx4.47$.

Squared distance adds one term per feature. So a feature measured on a large scale contributes large terms and dominates, and that fact returns in A12.

## A3. Mean, variance and standardisation

The **mean** of $m$ numbers is their sum over $m$. Their **variance** is the average squared distance from the mean, and the **standard deviation** is its square root.

To **standardise** a feature, subtract its mean and divide by its standard deviation. The result has mean 0 and variance 1, and has no units.

## A4. The best single number: why the mean

Suppose one cluster has values $x_1,\dots,x_m$ in one coordinate, and we must pick one number $c$ to represent them. Measure the cost by squared distance:

$$f(c)=\sum_{j=1}^m(x_j-c)^2.$$

**Step 1 — Differentiate.** $f'(c)=-2\sum_j(x_j-c)$.

**Step 2 — Set to zero.** $\sum_j x_j-mc=0$, so $c=\tfrac1m\sum_jx_j$, the mean.

**Step 3 — Check it is a minimum.** $f''(c)=2m>0$.

**Example.** The small runs have first coordinates $4,4,2,1$, with mean $2.75$. Then $f(2)=9$, $f(2.75)=6.75$ and $f(3.5)=9$.

Using absolute distance instead gives the **median**, and that version is called K-medians. The mean is cheap to compute but pulled by outliers: $1,2,3,100$ has mean $26.5$ and median $2.5$.

## A5. The K-means objective

Assign each example a label $g_i\in\{1,\dots,K\}$ and give each cluster a representative point $\mu_k$, its **centroid**. Minimise the total squared distance from each point to its own centroid:

$$J=\sum_{i=1}^n\|x_i-\mu_{g_i}\|^2.$$

There are two sets of unknowns, the labels and the centroids. Trying every labelling is hopeless, since there are $K^n$ of them: $2^{1000}$ has 302 digits.

**Intuition.** A pizza company opens $K$ outlets. Households are points, outlets are centroids, and $J$ is the total squared delivery distance.

| partition of the eight runs | clusters | $J$ |
|---|---|---|
| all eight together | 1 | 100.875 |
| a mixed split $\{1,2,5,8\},\{3,4,6,7\}$ | 2 | 99.25 |
| small and large $\{1,2,3,4\},\{5,6,7,8\}$ | 2 | 15.25 |

The key observation: **each unknown is easy when the other is fixed.**

## A6. Lloyd's algorithm

Repeat until no label changes:

1. **Assign.** Each point joins its nearest centroid: $g_i\leftarrow\arg\min_k\|x_i-\mu_k\|^2$.
2. **Update.** Each centroid moves to the mean of its points.

An assignment step costs about $nKd$ operations, and the number of iterations is usually small, so it works on millions of points. If a cluster becomes empty its mean is undefined: implementations either leave its centroid alone or move it to the point farthest from its own centroid.

## A7. A complete worked run

Start deliberately badly, with $\mu_A=(2,3)$ and $\mu_B=(1,3)$, both among the small runs.

**Iteration 1, assign.** Every run but run 4 is nearer to $A$, so $J=228.00$.

**Iteration 1, update.** $A$ holds runs 1, 2, 3, 5, 6, 7, 8, with mean $(40/7,\ 41/7)=(5.714,\ 5.857)$. $B$ holds run 4 alone, so it stays at $(1,3)$.

**Iteration 2, assign.** Runs 2 and 3 are now closer to $B$. $J=51.22$.

**Iterations 3 and 4.** Run 1 moves to $B$ too, $J=20.40$, and then $J=15.25$. Nothing changes after that.

![Four panels of the eight runs, one per iteration of Lloyd's algorithm, with the objective falling from 228 to 15.25.](images/L14/figures/eight_runs_lloyd.svg)
*Lloyd's algorithm on the eight runs from a poor start. Colours are the assignment in each iteration and rings are the centroids used for it. The objective printed above each panel falls at every step, and the final partition is the best of all 256.*

## A8. The boundary between two clusters

A point $x$ is nearer to $\mu_a$ than to $\mu_b$ when $\|x-\mu_a\|^2<\|x-\mu_b\|^2$. Expand both sides. The term $\|x\|^2$ cancels and what remains is **linear** in $x$:

$$2(\mu_b-\mu_a)^\top x<\|\mu_b\|^2-\|\mu_a\|^2.$$

The boundary is a straight line, in fact the **perpendicular bisector** of the two centroids. For the final centroids of the eight runs it is $9.5x_1+9x_2=98.1875$. With $K$ centroids each region is convex, which is why K-means can only ever produce straight boundaries.

## A9. Within-cluster and between-cluster spread

For any partition, the total spread splits exactly in two:

$$\underbrace{\sum_i\|x_i-\bar x\|^2}_{T}=\underbrace{\sum_i\|x_i-\mu_{g_i}\|^2}_{W=J}+\underbrace{\sum_kn_k\|\mu_k-\bar x\|^2}_{B}.$$

**Why.** Add and subtract $\mu_{g_i}$ inside the first norm. The cross term sums to zero within each cluster, since the deviations from a cluster mean sum to zero.

$T$ is fixed by the data, so **a smaller $W$ is the same thing as a larger $B$**. For the eight runs, $100.875=15.25+85.625$. The factor $n_k$ is there because each cluster mean stands for $n_k$ points.

![Two panels of the eight runs: segments to the overall mean, then segments to cluster means and from cluster means to the overall mean.](images/L14/figures/within_between.svg)
*The decomposition $T=W+B$ on the eight runs. Left: every run to the overall mean. Right: every run to its cluster mean (solid, $W$), and each cluster mean to the overall mean (dashed, counted once per point, $B$).*

## A10. Why it stops, and why that is not enough

Neither step can raise $J$. The assignment gives each point its closest centroid, and the update uses the mean, which is the best centroid for those points. There are finitely many labellings and $J$ can never return to an old value without being unchanged, so the algorithm stops.

**Careful.** Stopping does not mean best. For the numbers $0,1,2,4$ with $K=2$, Lloyd can stop at $\{0,1\},\{2,4\}$ with $J=0.25+0.25+1+1=2.5$, though $\{0,1,2\},\{4\}$ has $J=1+0+1+0=2$. Both are **local minima**: no step improves either. Finding the global minimum is NP-hard.

![A heat map of the objective over two centroid positions for the data 0, 1, 2, 4, with separate valleys.](images/L14/figures/local_minima_valley.svg)
*The objective as a function of the two centroids for the data $0,1,2,4$, where darker is lower. There are separate valleys, so where you start decides where you stop.*

## A11. Starting well: K-means++

With 90 points in three equal groups, the three random starting points land in three different groups with probability $\tfrac{60}{89}\times\tfrac{30}{88}=0.230$. So 77% of random starts put two centroids in one group, which can end at $J=453.1$ instead of the best $77.6$.

**K-means++** spreads the starts:

1. Choose the first centroid uniformly from the data.
2. For each point let $D_i$ be the squared distance to its nearest chosen centroid.
3. Choose the next centroid to be point $i$ with probability $D_i/\sum_lD_l$. Repeat.

**Why not simply take the farthest point?** Because the farthest point is usually an outlier, and a centroid would be wasted on it. A random draw gives one outlier only its own share, while thirty moderately far points together outweigh it. Over 400 starts on the 90 points, the best partition was reached 84% of the time from random starts and 97.8% from K-means++.

**Example.** For data $0,1,10$ with first centroid $0$, the values $D$ are $0,1,100$. So $1$ is chosen with probability $1/101$, and $10$ with probability $100/101$.

## A12. Standardise the features

Squared distance adds across features, so units decide what matters. On 80 runs recorded in hours and megabytes, K-means split on memory and agreed with the true groups only 51% of the time. After standardising, it agreed 100%.

Standardising is a modelling choice: it says every feature should have equal influence. If some are known to be more important, select or weight them instead.

## A13. How many clusters?

$K$ must be chosen before running, and $J$ cannot choose it, because more clusters always gives a smaller or equal best $J$, reaching 0 at $K=n$.

| $K$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| best $J$ for the eight runs | 100.875 | 15.250 | 8.750 | 4.333 | 1.500 | 1.000 | 0.500 | 0.000 |

Five approaches:

- **Elbow.** Plot the best $J$ against $K$, and find where the fall slows. Often there is no clear bend.
- **Silhouette.** For point $i$, let $a_i$ be its mean distance to the rest of its own cluster and $b_i$ the mean distance to the nearest other cluster. Then $s_i=(b_i-a_i)/\max(a_i,b_i)$. Values near 1 mean well placed, near 0 on a border, and below 0 probably wrong. Pick the $K$ with the largest mean.
- **Calinski-Harabasz.** Between-spread per degree of freedom over within-spread per degree of freedom: $\text{CH}(K)=\dfrac{B/(K-1)}{W/(n-K)}$. Pick the largest. The divisors matter because $B$ grows and $W$ shrinks with $K$ regardless.
- **Gap statistic.** $\text{Gap}(K)=\overline{\log W^*_K}-\log W_K$, where $W^*_K$ comes from uniform random data in the same box. Pick where the data beat random data by the most.
- **Domain knowledge.** Three marketing budgets means $K=3$. Often the application decides.

![Inertia falling with K with a bend at three, beside the average silhouette peaking at three.](images/L14/figures/choosing_k.svg)
*Two ways to read the same runs. Inertia always falls, so look for the bend. The silhouette can go down as well as up, and here it peaks at the true number of groups.*

## A14. Where K-means fails

| Limitation | Where it bites |
|---|---|
| $K$ must be chosen | data with no clear groups |
| The result depends on the start | a random start splits a group |
| It depends on feature units | hours against megabytes |
| Outliers pull centroids | one extreme run drags a mean |
| Round, similar clusters and straight boundaries | long clusters, half-moons |
| Numbers and Euclidean distance only | categories such as GPU type |

![K-means results on elongated bands, rings and very unequal groups, each divided incorrectly.](images/L14/figures/kmeans_failures.svg)
*Colours show what K-means finds with two clusters. Each shape has two true groups, and each is cut wrongly because K-means prefers round groups of similar size.*

---

# Part B — Slide by Slide

## Slide 1 — Title

![Slide 1: title of the lecture, Clustering with K-means.](images/L14/slides/s01.png)
*Slide 1: the lecture title.*

## Slide 2 — Learning without labels

![Slide 2: a pile of objects that could be sorted by colour or by size, contrasted with supervised learning.](images/L14/slides/s02.png)
*Slide 2: the same objects could be sorted by colour or by size. Supervised learning has a $y$ to measure against, and unsupervised learning has no $y$.*

Unsupervised learning finds which points belong together. See A1.

## Slide 3 — Where clustering is used

![Slide 3: applications including customer segments, image compression and vector search.](images/L14/slides/s03.png)
*Slide 3: customer segments, image compression with $K$ colours, grouping embeddings, vector search and scheduling training runs by resource use.*

Vector search compares a query with cluster centres first, so it only searches inside the closest clusters.

## Slide 4 — Eight training runs, no labels

![Slide 4: scatter plot of the eight runs by training hours and memory, showing two groups by eye.](images/L14/slides/s04.png)
*Slide 4: the running dataset. Two groups are obvious by eye here, but not with 10,000 runs and 50 features.*

## Slide 5 — The K-means objective

![Slide 5: the objective J as a sum of squared distances to assigned centroids.](images/L14/slides/s05.png)
*Slide 5: two unknowns, labels and centroids, and $K^n$ possible labellings.*

See A5.

## Slide 6 — The idea: two pizza outlets

![Slide 6: three rounds of customers ordering from the nearest outlet while outlets move to the average of their customers.](images/L14/slides/s06.png)
*Slide 6: order from the nearest outlet, then move each outlet to the average of its customers. Three rounds and it stops.*

## Slide 7 — Lloyd's algorithm

![Slide 7: Lloyd's algorithm with assign and update steps and a deliberately bad start.](images/L14/slides/s07.png)
*Slide 7: the algorithm, and a deliberately poor start at $(2,3)$ and $(1,3)$.*

## Slide 8 — Iteration 1: assign

![Slide 8: table of squared distances from each run to the two centroids, with J equal to 228.](images/L14/slides/s08.png)
*Slide 8: run 1 is at squared distance 5 from $\mu_A$ and 10 from $\mu_B$, so it joins $A$. Only run 4 joins $B$. $J=228$.*

## Slide 9 — Iteration 1: update

![Slide 9: the updated centroid of cluster A at (5.714, 5.857).](images/L14/slides/s09.png)
*Slide 9: $\mu_A$ moves to $(5.714,5.857)$. It is dragged towards the large runs, since most of its points are large.*

## Slide 10 — Iteration 2: runs 2 and 3 move to B

![Slide 10: second assignment table, with runs 2 and 3 moving to cluster B and J falling to 51.22.](images/L14/slides/s10.png)
*Slide 10: runs 2 and 3 switch sides, and $J$ falls from 228 to 51.22.*

## Slide 11 — Iterations 3 and 4: stop

![Slide 11: the final clusters of the eight runs and the falling sequence of J values.](images/L14/slides/s11.png)
*Slide 11: $J$ falls $228\to51.22\to20.40\to15.25$, which is the best of all 256 labellings.*

## Slide 12 — Update step: the best centroid is the mean

![Slide 12: derivation that the mean minimises the sum of squared distances.](images/L14/slides/s12.png)
*Slide 12: the derivative gives the mean, and the second derivative $2m>0$ confirms a minimum. See A4.*

## Slide 13 — Why squared distance?

![Slide 13: squared distance gives the mean and absolute distance gives the median.](images/L14/slides/s13.png)
*Slide 13: squared distance gives the mean, absolute distance gives the median. Outliers pull the mean.*

## Slide 14 — Assignment step: the boundary is a straight line

![Slide 14: the linear boundary between two centroids drawn on the eight runs.](images/L14/slides/s14.png)
*Slide 14: the boundary is the perpendicular bisector, here $9.5x_1+9x_2=98.1875$. See A8.*

## Slide 15 — $J$ never increases, so the algorithm stops

![Slide 15: argument that J cannot rise, and an example of stopping at a worse answer.](images/L14/slides/s15.png)
*Slide 15: neither step raises $J$, but stopping is not the same as being best. See A10.*

## Slide 16 — Within-cluster and between-cluster spread

![Slide 16: the identity T = W + B illustrated on the eight runs.](images/L14/slides/s16.png)
*Slide 16: $T=W+B$, with $100.875=15.25+85.625$. See A9.*

## Slide 17 — $J$ has several valleys

![Slide 17: heat map of the objective with two stopping points marked.](images/L14/slides/s17.png)
*Slide 17: stars mark the two stopping points for $0,1,2,4$, with $J=2.5$ and $J=2$.*

## Slide 18 — A random start can fail

![Slide 18: a start with two centroids in one group, ending in a poor clustering.](images/L14/slides/s18.png)
*Slide 18: two centroids in one group happen in 77% of random starts, giving $J=453.1$ against the best $77.6$.*

## Slide 19 — The fix: spread the starting centroids out

![Slide 19: K-means++ choosing centres with probability proportional to squared distance.](images/L14/slides/s19.png)
*Slide 19: bigger discs mark points more likely to be chosen next.*

## Slide 20 — Why not simply take the farthest point?

![Slide 20: a lone outlier against thirty moderately far points in an empty group.](images/L14/slides/s20.png)
*Slide 20: the farthest point is always the outlier, while drawing at random picks the empty group 90% of the time.*

## Slide 21 — K-means++

![Slide 21: the K-means++ algorithm, and 84 percent against 97.8 percent success.](images/L14/slides/s21.png)
*Slide 21: over 400 starts on 90 points, random starts reach the best partition 84% of the time and K-means++ 97.8%.*

## Slide 22 — Standardise the features first

![Slide 22: clustering on hours and megabytes before and after standardising.](images/L14/slides/s22.png)
*Slide 22: agreement rises from 51% to 100%.*

## Slide 23 — How many clusters, K?

![Slide 23: the table of smallest J for K from 1 to 8.](images/L14/slides/s23.png)
*Slide 23: more clusters always means a smaller $J$, so $J$ alone cannot choose $K$.*

## Slide 24 — 1. Elbow method

![Slide 24: two elbow plots, one with a clear bend at three and one with no clear bend.](images/L14/slides/s24.png)
*Slide 24: drops of 628.1, 386.3 and then 11.2 give a bend at $K=3$. The right panel shows the weakness: often no clear bend.*

## Slide 25 — 2. Silhouette of one point

![Slide 25: distances a and b for one point, and the silhouette formula.](images/L14/slides/s25.png)
*Slide 25: $a_i$ looks inward, $b_i$ looks at the nearest other cluster.*

## Slide 26 — Choosing K with the silhouette

![Slide 26: the silhouette score plotted against K with its largest value marked.](images/L14/slides/s26.png)
*Slide 26: pick the $K$ with the largest mean silhouette.*

## Slide 27 — Silhouette plot: which cluster is wrong

![Slide 27: silhouette bars for each point, sorted within clusters.](images/L14/slides/s27.png)
*Slide 27: one bar per point, sorted within each cluster. It shows which cluster is responsible for a low score.*

## Slide 28 — 3. Calinski-Harabasz

![Slide 28: the Calinski-Harabasz score plotted against K, largest at three.](images/L14/slides/s28.png)
*Slide 28: $K=3$ gives $507.2/0.892=568.8$ and $K=4$ gives $341.9/0.772=443.1$, so pick $K=3$.*

## Slide 29 — 4. Gap statistic

![Slide 29: within-cluster spread on data and on uniform random data, and the gap between them.](images/L14/slides/s29.png)
*Slide 29: compare against 20 copies of uniform data in the same box.*

## Slide 30 — 5. Domain knowledge

![Slide 30: examples where the application fixes the number of clusters.](images/L14/slides/s30.png)
*Slide 30: 3 budgets, 4 T-shirt sizes, 2 machine types, a 16-colour image.*

## Slide 31 — Limitations of K-means

![Slide 31: a table of the limitations of K-means and where each one bites.](images/L14/slides/s31.png)
*Slide 31: see the table in A14.*

## Slide 32 — Where K-means fails: shape

![Slide 32: K-means on a long cluster with a small one, and on half-moons.](images/L14/slides/s32.png)
*Slide 32: a long cluster with a small one gives 56% agreement, and half-moons give 77%.*

## Slide 33 — What handles these cases

![Slide 33: Gaussian mixtures with ellipses, and DBSCAN following dense chains.](images/L14/slides/s33.png)
*Slide 33: Gaussian mixtures make each cluster an ellipse, and DBSCAN makes clusters out of dense chains. These are the next note.*

## Slide 34 — Live demonstration

![Slide 34: the outline of the live notebook demonstration.](images/L14/slides/s34.png)
*Slide 34: the notebook covers pizza outlets, random and K-means++ starts, elbow and silhouette, and round groups then half-moons.*

## Slide 35 — Review questions

![Slide 35: five review questions.](images/L14/slides/s35.png)
*Slide 35: the review questions are answered in Part C.*

## Board notes — What was written in class

### Board p. 1 — Why unsupervised learning

![Handwritten notes listing K-means, DBSCAN and PCA, contrasting supervised learning where somebody can correct you with unsupervised learning where nobody can.](images/L14/board/h01.png)
*In supervised learning somebody can correct or penalise the model. In unsupervised learning there are many possible outcomes and no one to correct you. Social networks are given as an example: communities, modularity and graph colouring.*

### Board p. 2 — Where this shows up

![Handwritten notes on graph communities, locality, anomaly detection with time-series models, segmentation, and the Swiss roll manifold.](images/L14/board/h02.png)
*Communities in a graph of about 2000 nodes, locality, anomaly detection using a time-series model such as $y[k]=\alpha y[k-1]+\epsilon_k$ (AR, ARMA, ARIMA), image segmentation, and data on a manifold such as the Swiss roll. Methods split into DBSCAN and t-SNE on one side, and K-means, anomaly detection and PCA on the other.*

### Board p. 3 — K-means as houses

![Handwritten sketch of houses grouped around centres on a plane, with K as the number of dark stores.](images/L14/board/h03.png)
*A thousand houses on a map and a choice of $K$ dark stores. Each house is served by its nearest store, and the question is how many stores, and where.*

---

# Part C — Practice

**P1.** Data $1,2,9$ with $K=2$, starting at centroids $1$ and $2$. Find the final clusters and $J$.

<details><summary>Answer</summary>

Assign: $\{1\}$ and $\{2,9\}$, so the centroids become $1$ and $5.5$. Assign: $2$ is at distance $1$ from $1$ and $3.5$ from $5.5$, so it moves. Centroids $1.5$ and $9$. Nothing changes. $J=0.25+0.25+0=0.5$.

</details>

**P2.** Why can the update step not increase $J$?

<details><summary>Answer</summary>

With the labels fixed, $J$ separates over clusters and coordinates, and the mean is the exact minimiser of each piece.

</details>

**P3.** Why does the smallest $J$ never rise with $K$?

<details><summary>Answer</summary>

Take the best partition into $K$ clusters and move one point into its own new cluster. That point's term becomes zero, and removing a point from a cluster and recomputing its mean cannot raise that cluster's sum.

</details>

**P4.** Two runs differ by 3 hours and 2000 MB. Standard deviations are 1.6 hours and 3634 MB. Find memory's share of the squared distance, raw and standardised.

<details><summary>Answer</summary>

Raw: $9+4{,}000{,}000$, so memory's share is about $99.9998\%$. Standardised: $(3/1.6)^2=3.516$ and $(2000/3634)^2=0.303$, so its share is $0.303/3.819=7.9\%$.

</details>

**P5.** K-means++ on $0,1,2,10,11$ with first centroid $0$. Give the probabilities for the second centroid. If $10$ is chosen, give those for the third.

<details><summary>Answer</summary>

$D=0,1,4,100,121$, summing to $226$, so $1/226$, $4/226$, $100/226$ and $121/226$. With centroids $0$ and $10$: $D=0,1,4,0,1$, summing to 6, so $1$ has $1/6$, $2$ has $4/6$ and $11$ has $1/6$.

</details>

**P6.** Data $2,4,6,10,12$ with $K=2$. Find $J$ for $\{2,4,6\},\{10,12\}$ and for $\{2,4\},\{6,10,12\}$. Is the second a fixed point?

<details><summary>Answer</summary>

First: $8+2=10$. Second: $2+18.67=20.67$. It is not a fixed point, because the centroids are $3$ and $9.33$, and $6$ is closer to $3$.

</details>

**P7.** Find all fixed points of Lloyd's algorithm for $0,1,2,4$ with $K=2$.

<details><summary>Answer</summary>

$\{0,1\},\{2,4\}$ with $J=2.5$, and $\{0,1,2\},\{4\}$ with $J=2$. The split $\{0\},\{1,2,4\}$ is not fixed: its centroids are $0$ and $2.33$, and $1$ is nearer to $0$.

</details>

**P8.** Why divide by $K-1$ and $n-K$ in Calinski-Harabasz?

<details><summary>Answer</summary>

$B$ tends to grow and $W$ tends to shrink as $K$ rises, whether or not structure exists. Dividing by the degrees of freedom removes that automatic trend.

</details>
