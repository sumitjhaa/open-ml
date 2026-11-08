// Step-up path added to each published note: what you need, the climb from intuition to the hard part, and how to know you are done.
// `extra` adds ideas the interactive sections use that the original note does not spell out.
const R = String.raw;

export const STEPUPS = {
  L01_ml_intro_past_present: {
    need: 'Arithmetic and curiosity. Part A rebuilds everything else from the start.',
    rungs: ['Think of a model as a machine with knobs, and learning as turning them.', 'Measure wrongness with a loss, and average it to get the empirical risk $\\hat R$.', 'Separate $\\hat R$, which you can compute, from the true risk $R$, which you actually care about.', 'Use the four paradoxes as a checklist of how a good-looking number can mislead.'],
    done: 'you can say why a 99% accurate model might be worthless.',
    extra: R`**Why error can fall twice.** Past the point where a model fits every training point, many perfect fits exist. A training rule that picks the one with the smallest weights, the **minimum-norm** solution, tends to give a smooth function, so test error can fall a second time. The spike happens exactly at the threshold, where only one fit exists and it has to contort.`,
  },
  L02_mathematical_foundations: {
    need: 'School algebra and the idea of a function. Sets, sums and logs are in the introduction notes.',
    rungs: ['Name each object: scalar, vector or matrix, and check its shape.', 'Read a slope as a rate, then a gradient as a slope in every direction at once.', 'Walk downhill with gradient descent, and use convexity to know that the bottom is unique.', 'Move from counting to densities, where area and not height is probability.', "Combine densities with Bayes' rule to invert a generative story."],
    done: 'you can compute a gradient by hand and explain what its direction means.',
    extra: R`**Bayes in odds form.** Dividing Bayes' rule for two hypotheses cancels the evidence term:
$$\frac{P(H_1\mid x)}{P(H_0\mid x)}=\frac{p(x\mid H_1)}{p(x\mid H_0)}\cdot\frac{P(H_1)}{P(H_0)}.$$
In words, **posterior odds equal the likelihood ratio times the prior odds**. It is often the fastest way to do these calculations by hand, because the normalizing constants of the densities cancel too.

**Classifying critical points.** A point where $f'(u)=0$ is a critical point. The second-derivative test says $f''>0$ means a local minimum, $f''<0$ a local maximum, and $f''=0$ is inconclusive. For $g(u)=u^4-2u^2$, $g'(u)=4u^3-4u$ is zero at $u=-1,0,1$, and $g''(u)=12u^2-4$ equals $8,-4,8$ there. So there are two global minima of value $-1$, with a local maximum between them. Objectives with several valleys are why optimization can be hard.

**A function with a corner.** ReLU, $m(z)=\max(0,z)$, has slope 0 for $z<0$ and 1 for $z>0$. At $z=0$ the derivative does not exist: any slope between 0 and 1 is a valid subgradient, and software simply picks one.

**Shapes through a chain.** For $z=W_2(W_1x+b_1)+b_2$ with $x\in\mathbb R^4$, $W_1\in\mathbb R^{6\times4}$ and $W_2\in\mathbb R^{2\times6}$, the shapes go $4\to6\to2$. Writing the shape of each intermediate object catches most errors. This chain is the forward pass of a two-layer network.`,
  },
  L03_estimating_parameters: {
    need: 'Derivatives, logs, and the idea of a probability distribution.',
    rungs: ['Fix the data, and let the parameter vary: that is a likelihood.', 'Take logs, because products become sums and the peak stays in place.', 'Differentiate, set to zero, and solve: that is the MLE recipe.', 'Add a prior to get a posterior, and read the MAP and posterior mean off it.', 'Ask how sure you are, by looking at how sharp the peak is.'],
    done: 'you can derive the MLE of a new distribution without looking anything up.',
    extra: R`**How sharp is the peak?** The negative second derivative of the log-likelihood at the peak is the **Fisher information** (observed). For a Bernoulli with $n$ tosses, $I(\hat\theta)=n/\big(\hat\theta(1-\hat\theta)\big)$, and the standard error is about $1/\sqrt{I}$. Four times the data halves the standard error.

**Averages form a bell.** By the central limit theorem, the mean of $n$ independent draws is approximately normal with spread $\sigma/\sqrt n$, whatever the shape of the original distribution.

**Modelling a skewed positive quantity.** $X$ is **lognormal** when $\log X$ is normal. Times and sizes often behave this way. Writing $y_i=\log x_i$, the MLEs are the mean and variance of the $y_i$. The median of $X$ is $e^{\theta_1}$, and the long right tail pulls the mean up to $e^{\theta_1+\theta_2/2}$.

**A conjugate prior for a variance.** With the mean known, the likelihood has the shape $\theta_2^{-n/2}e^{-S/(2\theta_2)}$. The **inverse-gamma** prior $\theta_2^{-a-1}e^{-b/\theta_2}$ has the same shape, so powers and constants simply add: $\text{IG}(a,b)$ becomes $\text{IG}(a+\tfrac n2,\ b+\tfrac S2)$. With $a=3$, $b=4$, $n=8$ and $S=16$, the posterior is $\text{IG}(7,12)$. Its mode is $b/(a+1)=1.5$, midway between the prior's 1 and the data's 2, and its mean is $b/(a-1)=2$.`,
  },
  L04_linear_regression_loss_projection: {
    need: 'Dot products, matrix shapes, and the MLE recipe.',
    rungs: ['See residuals as errors, and squared loss as the area of squares.', 'Learn where the loss comes from: Gaussian noise gives squares, and Laplace noise gives absolute values.', 'Stack everything into $X$ and $y$, and write the loss as a single norm.', 'Differentiate to get the normal equations.', 'Read those equations as geometry: the residual is perpendicular to every column.'],
    done: 'you can derive $X^\\top X\\theta=X^\\top y$ and draw what it means.',
  },
  L05_normal_equation_eigendirections_pca: {
    need: 'The normal equations, and what a matrix does to a vector.',
    rungs: ['Hold two pictures at once: coefficient space and observation space.', 'Find the directions a symmetric matrix only stretches, its eigenvectors.', 'Read the loss bowl through those directions: eigenvalues are curvatures.', 'Connect small eigenvalues to unstable coefficients.', 'Maximise variance along a direction, which gives PCA as an eigenvalue problem.'],
    done: 'you can explain why near-duplicate columns make coefficients unstable, using eigenvalues.',
  },
  L06_curvature_logistic_regression: {
    need: 'Gradients, eigenvalues, and the likelihood recipe.',
    rungs: ['Read curvature from a second derivative, and then from a Hessian.', 'Approximate a surface by a quadratic, and jump to its bottom: that is Newton.', 'Pass from probability to odds to log-odds, the scale where a linear score makes sense.', 'Apply the likelihood recipe to a coin at each $x$, which gives cross-entropy.', 'Show the Hessian is positive semidefinite, so logistic loss is convex.'],
    done: 'you can write the gradient and Hessian of logistic loss, and say what happens on separable data.',
  },
  L07_regularization_ridge_lasso_elastic_net: {
    need: 'OLS, eigen-directions, and the idea of test error.',
    rungs: ['Split test error into bias, variance and noise.', 'See that small eigenvalues amplify noise by $1/\\mu$.', 'Add a penalty, and watch ridge replace that factor by a bounded one.', 'Change the penalty to $\\ell_1$, and see corners produce exact zeros.', 'Choose the strength on held-out data.'],
    done: 'you can say which method you would pick for near-duplicate features, and why.',
    extra: R`**The one-standard-error rule.** Cross-validation scores are noisy, so the lowest one is not reliably the best. Find the minimum, then choose the simplest model whose score lies within one standard error of it. Differences smaller than the noise do not justify extra complexity.

**Writing the penalty as a matrix.** To leave the intercept unpenalized, use $P=\operatorname{diag}(0,1,\dots,1)$ and solve
$$(X^\top X+\alpha P)\beta=X^\top y.$$
Solve this system directly, rather than computing an inverse. For logistic ridge, the same $P$ appears in the gradient, $X^\top(p-y)+\lambda P\beta$, and the Hessian, $X^\top WX+\lambda P$.`,
  },
  L08_ols_ridge_lasso_worked_example: {
    need: 'The ridge and lasso formulas from the previous note.',
    rungs: ['Solve a small system by hand, and see two equations as two crossing lines.', 'Make the columns nearly equal, and watch the crossing point run away.', 'Rotate into eigen-directions, where gradient descent splits into separate problems.', 'Compute the lasso solution from its optimality condition.'],
    done: 'you can predict from eigenvalues how many gradient steps a problem will need.',
    extra: R`**Why each direction converges at its own rate.** In eigen-coordinates, one step of gradient descent multiplies coordinate $j$ by the **contraction factor** $1-\eta\mu_j$. The step is stable when every factor is smaller than one in size, so $\eta<2/\mu_{\max}$. A negative factor means zig-zag, and the flattest direction sets the overall speed.`,
  },
  L09_metrics_thresholds_costs: {
    need: 'Conditional probability, and the logistic model’s output.',
    rungs: ['Separate the scorer from the decision rule.', 'Count the four outcomes, and form each rate with its own denominator.', 'Sweep the threshold to get ROC and precision-recall curves.', 'Check whether the probabilities can be trusted, with calibration.', 'Put prices on errors, and derive the threshold.'],
    done: 'you can derive $t^*=C_{FP}/(C_{FP}+C_{FN})$ and say when it applies.',
    extra: R`**Overconfidence is invisible to AUC.** Multiplying every log-odds by a constant above one pushes probabilities towards 0 and 1. The ranking does not change, so AUC does not change, but the probabilities become wrong. Check a reliability diagram whenever decisions use the probabilities themselves. Ways to repair it are in the features and diagnostics notes.`,
  },
  L10_precision_recall_roc_auc: {
    need: 'The confusion matrix and the idea of a threshold.',
    rungs: ['Compute metrics at one threshold, by counting.', 'Walk down the ranked list to draw the ROC curve by hand.', 'Read AUC as a probability about random pairs.', 'See why precision depends on prevalence, and ROC does not.', 'Connect a cost per case to a total cost per population.'],
    done: 'you can say which curve to report for a rare event, and against what baseline.',
  },
  L11_support_vector_machines: {
    need: 'Dot products, distance to a line, and constrained optimization at a basic level.',
    rungs: ['Choose the separating line with the widest street.', 'Fix the scale so the street width is $2/\\|w\\|$, which gives a convex problem.', 'Move to the dual, where only dot products of the data appear.', 'Replace the dot product by a kernel to get curved boundaries.', 'Add slack, and see the same objective as hinge loss plus ridge.'],
    done: 'you can say what $\\alpha_i=0$, $0<\\alpha_i<C$ and $\\alpha_i=C$ each mean geometrically.',
  },
  L12a_decision_trees: {
    need: 'Fractions and logs in base two.',
    rungs: ['Follow a tree from root to leaf.', 'Measure how mixed a group is, with entropy or Gini.', 'Score a question by the drop in impurity.', 'Grow greedily, and see why deep trees memorise.', 'Prune, and see why a small data change rebuilds the tree.'],
    done: 'you can grow a small tree by hand and explain why it is unstable.',
    extra: R`**Trees are greedy.** Each split is the best one available right then, with no look-ahead. Finding the truly best tree is computationally intractable, so a question that looks useless now, as on XOR, may be skipped even though it would unlock better questions below it.

**Interpretability.** A small tree can be read as a list of rules, which is why trees are popular where decisions must be explained. That advantage fades as depth grows, and ensembles of hundreds of trees give it up.

**Pruning by price.** Cost-complexity pruning minimises training error plus $\alpha$ per leaf. See the features and diagnostics notes for the formula and a worked example.`,
  },
  L12b_bagging_random_forests: {
    need: 'Variance, and how a single tree is built.',
    rungs: ['See that averaging independent predictions divides the variance by $B$.', 'Allow for correlation, which leaves a floor at $\\rho\\sigma^2$.', 'Make many training sets from one, with the bootstrap.', 'Lower $\\rho$ by hiding features at each split: a random forest.', 'Use the rows a tree never saw as a free validation set.'],
    done: 'you can explain what bagging fixes and what it cannot fix.',
    extra: R`**Why a committee beats its members.** If $B$ independent voters are each right with probability $p>\tfrac12$, the probability that the majority is right climbs towards 1 as $B$ grows. With $p<\tfrac12$ it falls towards 0, so a committee needs members better than chance and errors that are not all the same.

**Robustness.** Averaging makes bagging tolerant of noise and label errors, because a bad point appears in only about 63% of the samples and is outvoted. Boosting is more exposed, since it keeps concentrating on the points it finds hard, and mislabeled points are hard for ever.`,
  },
  L12c_gradient_boosting: {
    need: 'Regression trees, and the idea of a gradient.',
    rungs: ['Start at the average.', 'Fit a small tree to what is still wrong, and add a fraction of it.', 'Recognise the residual as the negative gradient of the loss.', 'Move to classification, with scores in log-odds and a Newton step in each leaf.', 'Control overfitting with step size and stopping.'],
    done: 'you can run two rounds by hand, for regression and for classification.',
  },
  L12d_adaboost: {
    need: 'Stumps and weighted averages.',
    rungs: ['Weight the points, and find the stump with the lowest weighted error.', 'Give that stump a vote based on its error.', 'Raise the weight of its mistakes, and repeat.', 'Derive the vote by minimising $Z$.', 'See AdaBoost as boosting on the exponential loss.'],
    done: 'you can explain why the stump just used scores exactly 50% on the new weights.',
  },
  L12e_xgboost_and_comparison: {
    need: 'Gradient boosting and a second-order Taylor expansion.',
    rungs: ['Approximate the loss with slopes and curvatures.', 'Add penalties, and solve for the best leaf weight.', 'Score a split by what it saves, minus a price.', 'Compare all the ensembles on what they reduce and what they risk.'],
    done: 'you can compute a gain from $G$, $H$, $\\lambda$ and $\\gamma$, and say what each penalty does.',
    extra: R`**XGBoost for regression.** With loss $\tfrac12(F-y)^2$, the slope is $g=F-y$ and the curvature is $h=1$. Take four points $x=1,2,3,4$, $y=3,5,9,11$, with $\lambda=1$, $\gamma=1$ and step $\tfrac12$. Start at $F_0=7$, so $g=4,2,-2,-4$. For the split $x\le2$, $G_L=6$ and $G_R=-6$, with $H=2$ on each side, so each leaf scores $36/3=12$. The parent scores 0, so the gain is $\tfrac12(12+12-0)-1=11$, the largest of the three splits. The leaf weights are $-6/3=-2$ and $+2$, giving $F_1=6,6,8,8$, and the total squared error falls from 40 to 20.`,
  },
  L12f_supplement_missing_topics: {
    need: 'Train and test error, and at least one fitted model.',
    rungs: ['Ask what the test set actually promises.', 'List the ways test information leaks into training.', 'Ask whether the deployment world matches the training world.', 'Choose a validation design that matches how the data arose.'],
    done: 'you can read an evaluation and name its most likely flaw.',
  },
  L12g_supplement_features_diagnostics: {
    need: 'Linear models, the loss curve of gradient descent, and the idea of a held-out set.',
    rungs: ['Design columns so a simple model can express the truth.', 'Compare with a baseline, and read the residuals.', 'Make training work: scaling, momentum, and diagnosing the loss curve.', 'Repair probabilities and prune trees on held-out data.', 'State how uncertain the final score is.'],
    done: 'you can say what you would check next when a model disappoints.',
  },
  L13_svd_representation_learning: {
    need: 'Eigenvalues, PCA and matrix multiplication.',
    extra: R`**Dimensionality reduction in practice.** Replacing $d$ features by $k$ scores is used for compression, for plotting, and as a preprocessing step that removes weak, noisy directions. Fit it inside the training fold, and choose $k$ by what the downstream task needs and not only by variance kept.`,
    rungs: ['Extend eigen-thinking to rectangular matrices, using two sets of directions.', 'See any matrix as rotate, stretch, rotate.', 'Keep only the biggest stretches, which gives the best low-rank approximation.', 'Identify PCA as the SVD of the centred data.', 'Use the same coordinates as a learned representation.'],
    done: 'you can compute a small SVD by hand and say how many layers to keep, and why.',
  },
  L14_clustering_unsupervised: {
    need: 'Distance between two points, the mean, and the idea of a derivative.',
    rungs: ['Say what "together" means when there are no labels.', 'Write the objective, and derive why each centroid is a mean and each boundary a straight line.', 'Run Lloyd’s algorithm by hand, and see why it stops but may stop badly.', 'Start well with K-means++, and standardise the features.', 'Choose $K$ with several scores, and name where K-means fails.'],
    done: 'you can run K-means by hand, prove that $J$ never rises, and name data that break it.',
  },
  L15_mixtures_hierarchical_density: {
    need: 'K-means, the normal density and Bayes’ rule.',
    rungs: ['Replace hard labels with responsibilities.', 'Alternate E and M steps, and prove the likelihood never falls.', 'Build a whole tree of merges, and compare linkages.', 'Define clusters by density, with core, border and noise points.', 'Test whether any clustering is real, and choose a method.'],
    done: 'you can run a step of EM by hand and say which method suits a given dataset.',
  },
  L16_pca_visualisation_maps: {
    need: 'Dot products, variance and eigenvalues.',
    rungs: ['Project a point onto a direction and split it into kept and lost parts.', 'Write the variance along a direction as $u^\\top Su$.', 'Find the best direction with a Lagrange multiplier, which gives an eigenvector.', 'Choose how many components to keep, and how units change the answer.', 'Go beyond straight lines with t-SNE, and learn what such maps cannot say.'],
    done: 'you can compute a small PCA by hand and say what a t-SNE picture does and does not show.',
  },
  L17_anomaly_detection: {
    need: 'Mean and standard deviation, the median, and PCA.',
    rungs: ['Name the kind of unusual you are looking for.', 'See how an anomaly hides itself from the mean, and use robust scores.', 'Bring in the shape of the data with Mahalanobis distance.', 'Score by neighbours and by isolation, where no shape is assumed.', 'Evaluate a ranking with few labels, and monitor a deployed model for drift.'],
    done: 'you can choose a score for a given kind of anomaly and evaluate it honestly.',
  },
};
