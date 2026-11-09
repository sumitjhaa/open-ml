# Imports the 2025 course material from ../sources/course-2025 into the site, so the site depends on nothing outside this repo.
# Reads notebooks as JSON only; nothing in them is executed. Only text and images are kept from outputs, never HTML or scripts.
# Usage: python3 -I scripts/import-course.py   (writes public/notebooks, public/cheatsheets, public/slides, public/handouts, public/quizzes and lib/)
import base64
import html
import json
import os
import re
import shutil
import struct
import subprocess

SITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(SITE), 'sources', 'course-2025')
OUT = os.path.join(SITE, 'public', 'notebooks')
MAX_W, MAX_LINES = 900, 30

# Data the notebooks would fetch from other sites, kept beside them instead.
DATA_URLS = {
    'https://raw.githubusercontent.com/balasahebgulave/Dataset-Indian-Names/master/Indian_Names.csv': 'data/Indian_Names.csv',
    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv': 'data/airline-passengers.csv',
    'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv': 'data/co2_mm_mlo.csv',
}
NOTEBOOK_FILES = ('.csv', '.xlsx', '.jpg', '.py')

# Hosted demos are replaced by this site's own interactive sections.
LOCAL_LABS = [
    ('gpt-demo', 'Generate names', '/lectures/sequence-models/explore/name-sampler'),
    ('convolution-viz-multi', 'Filters as detectors', '/lectures/vision/explore/edge-filter-lab'),
    ('convolution-viz', 'Convolution by hand', '/lectures/vision/explore/conv-by-hand'),
    ('gradcam', 'Class activation maps', '/lectures/vision/explore/cam-lab'),
    ('dino', 'Pretrained features', '/lectures/vision/explore/frozen-features'),
    ('clustering', 'K-means step by step', '/lectures/clustering/explore/kmeans-steps'),
    ('rl-exploration', 'Explore or exploit', '/lectures/reinforcement-learning/explore/explore-tradeoff'),
    ('rl-demo', 'Q-learning on a grid', '/lectures/reinforcement-learning/explore/q-learn-lab'),
]

HANDOUTS = {
    'foundations': [('basics/tutorials/tutorial-0-prerequisites.pdf', 'Mathematical prerequisites')],
    'evaluation': [('basics/tutorials/mathematical-conventions.pdf', 'Accuracy conventions and metrics'), ('supervised/tutorials/cross-validation.pdf', 'Cross-validation'), ('supervised/tutorials/bias-variance.pdf', 'Bias and variance')],
    'trees-ensembles': [('basics/tutorials/decision-trees.pdf', 'Decision trees'), ('supervised/tutorials/ensemble-methods.pdf', 'Ensemble methods')],
    'linear-regression': [('supervised/tutorials/linear-regression.pdf', 'Linear regression')],
    'logistic-regression': [('supervised/tutorials/logistic-regression.pdf', 'Logistic regression')],
    'regularization': [('supervised/tutorials/lasso-regression.pdf', 'Lasso regression'), ('supervised/tutorials/feature-selection.pdf', 'Feature selection')],
    'neighbours-bayes': [('supervised/tutorials/knn.pdf', 'K-nearest neighbours'), ('supervised/tutorials/naive-bayes.pdf', 'Naive Bayes')],
    'support-vector-machines': [('supervised/tutorials/svm.pdf', 'Support vector machines')],
    'dimensionality-reduction': [('supervised/tutorials/matrix-factorization.pdf', 'Matrix factorisation')],
}

NOTEBOOKS = {
    'introduction': [('rule-based-vs-ml', 'Rules against learning'), ('mnist-digits', 'A first look at handwritten digits'), ('transcript', 'Speech transcription with local models')],
    'foundations': [('numpy-pandas-basics', 'NumPy and pandas basics'), ('tips', 'Python tips for machine learning'), ('visualisation', 'Plotting data'), ('binomial', 'The binomial distribution'), ('maths-for-ml-2', 'Maths for machine learning'), ('contour', 'Contour plots and gradients'), ('meshgrid-contour-explanation', 'Meshgrid explained'), ('taylor-series', 'Taylor series')],
    'linear-regression': [('linear-regression', 'Linear regression in code'), ('linear-regression-notebook', 'Linear regression worked through'), ('lin-reg-tutorial', 'Linear regression tutorial'), ('anscombe', 'Anscombe’s quartet'), ('geometric-linear-regression', 'The geometry of least squares'), ('basis', 'Basis expansion'), ('basis2', 'More basis expansion'), ('polynomial-features', 'Polynomial features'), ('dummy-variables-multi-colinearity', 'Dummy variables and multicollinearity'), ('gradient-descent', 'Gradient descent'), ('gradient-descent-2d', 'Gradient descent on a surface'), ('gradient-descent-fit-contour', 'Fitting a line on the contours'), ('cost-iteration-notebook', 'Cost of each iteration')],
    'logistic-regression': [('logits-usage', 'Why use logits'), ('logistic-regression-cost', 'The logistic cost'), ('logistic-apple-oranges', 'Apples and oranges'), ('logistic-circular', 'Circular classes'), ('logistic-iris', 'Logistic regression on iris'), ('logistic-regression-torch', 'Logistic regression in PyTorch'), ('logistic-torch-second-order', 'Second-order logistic regression')],
    'regularization': [('ridge', 'Ridge regression'), ('lasso-regression', 'Lasso regression'), ('sfs-and-bfs', 'Forward and backward selection')],
    'evaluation': [('confusion-mnist', 'Confusion matrix on digits'), ('dummy-baselines', 'Dummy baselines'), ('pr-curve', 'Precision-recall curves'), ('cross-validation-diagrams', 'Cross-validation diagrams'), ('hyperparams-experiments', 'Tuning hyperparameters'), ('hyperparameter-optimisation', 'Hyperparameter optimisation'), ('bias-variance', 'Bias and variance analysis'), ('bias-variance-charts', 'Bias and variance charts')],
    'support-vector-machines': [('svm', 'A first support vector machine'), ('svm-cvxopt', 'Solving the dual with CVXOPT'), ('svm-primal-dual', 'Primal and dual'), ('svm-kernel-understanding', 'Understanding kernels'), ('svm-kernel', 'Kernels in practice'), ('svm-soft-margin', 'The soft margin')],
    'trees-ensembles': [('entropy', 'Entropy'), ('classes-tree', 'Classes in a tree'), ('decision-tree-discrete-input-discrete-output', 'Trees on discrete inputs'), ('decision-tree-real-input-discrete-output', 'Real inputs, discrete output'), ('decision-tree-real-output', 'Real output'), ('decision-tree-real-input-real-output', 'Real inputs and real output'), ('dt-pruning', 'Pruning a tree'), ('dt-weighted', 'Weighted samples in a tree'), ('boosting-explanation', 'Boosting explained'), ('ensemble-feature-importance', 'Feature importance'), ('ensemble-representation', 'What an ensemble represents')],
    'dimensionality-reduction': [('pca', 'Principal component analysis'), ('pca-tutorial-1', 'PCA tutorial'), ('pca-tutorial-2', 'PCA tutorial, continued'), ('movie-recommendation-knn-mf', 'Movie recommendation'), ('tensor-factorisation', 'Tensor factorisation')],
    'neighbours-bayes': [('knn-variants', 'Variants of nearest neighbours'), ('curse-dimensionality', 'The curse of dimensionality')],
    'neural-networks': [('perceptron-learning', 'The perceptron learning algorithm'), ('autodiff', 'Automatic differentiation'), ('autograd-from-scratch', 'Autograd from scratch'), ('nn-vectorization-with-vmap-jaxtyping-and-beartype-zeel', 'Vectorising a network'), ('siren', 'Sinusoidal networks')],
    'vision': [('convolution-operation', 'The convolution operation'), ('convolution-operation-stride', 'Stride and padding'), ('1d-cnn', 'Convolution on sequences'), ('cnn-edge', 'Edge detection'), ('cnn', 'A convolutional network'), ('lenet', 'LeNet'), ('vgg-minst', 'VGG on digits'), ('dinov3-classification', 'Classifying with DINO features'), ('dinov3-regression', 'Regression with DINO features'), ('zeroshot-fewshot', 'Zero-shot and few-shot learning'), ('object-detection', 'Object detection'), ('generate-image', 'Generating images from prompts')],
    'sequence-models': [('names', 'Generating names'), ('autoregressive-model', 'Autoregressive models')],
    'reinforcement-learning': [('rl-gym-environments', 'Gym environments'), ('rl-qlearning', 'Q-learning'), ('rl-qlearning-deep', 'Deep Q-learning')],
    'practice': [('sklearn-on-gpu', 'scikit-learn on a GPU')],
}

SECTION_OF_LECTURE = {
    'lecture-1': 'introduction', 'lecture-2': 'evaluation', 'lecture-3': 'trees-ensembles', 'lecture-4': 'trees-ensembles', 'lecture-5': 'evaluation',
    'lecture-6': 'trees-ensembles', 'lecture-7': 'trees-ensembles', 'lecture-8': 'foundations', 'lecture-9': 'linear-regression', 'lecture-10': 'linear-regression',
    'lecture-11': 'linear-regression', 'lecture-12': 'dimensionality-reduction', 'lecture-13': 'regularization', 'lecture-14': 'regularization', 'lecture-15': 'support-vector-machines',
    'lecture-16': 'logistic-regression', 'lecture-17': 'logistic-regression', 'lecture-18': 'logistic-regression', 'lecture-19': 'evaluation', 'lecture-20': 'neural-networks',
    'lecture-21': 'neural-networks', 'lecture-22': 'neural-networks', 'lecture-23': 'sequence-models', 'lecture-24': 'vision', 'lecture-25': 'vision', 'lecture-26': 'vision',
    'lecture-27': 'clustering', 'lecture-28': 'dimensionality-reduction', 'lecture-29': 'reinforcement-learning', 'lecture-30': 'neighbours-bayes', 'lecture-31': 'support-vector-machines',
    'lecture-32': 'support-vector-machines', 'lecture-33': 'support-vector-machines', 'topic-naive-bayes-and-bayesian-networks': 'neighbours-bayes',
    'topic-time-series-and-sequence-models': 'sequence-models', 'topic-object-detection': 'vision',
}


# Decks the 2025 schedule does not link, on topics this site covers.
MORE_SLIDES = {
    'foundations': [('maths/slides/mvn.pdf', 'Multivariate normal'), ('maths/slides/mathematical-ml.pdf', 'Mathematics for machine learning')],
    'logistic-regression': [('optimization/slides/convexity.pdf', 'Convexity')],
    'regularization': [('optimization/slides/coordinate-descent.pdf', 'Coordinate descent'), ('optimization/slides/subgradient.pdf', 'Subgradients')],
    'support-vector-machines': [('maths/slides/constrained-1.pdf', 'Constrained optimisation'), ('maths/slides/constrained-2.pdf', 'Constrained optimisation, continued'), ('maths/slides/kkt-conditions.pdf', 'KKT conditions')],
    'trees-ensembles': [('maths/slides/decision-tree-complexity.pdf', 'Time complexity of trees')],
    'neighbours-bayes': [('supervised/slides/bayesian-nets.pdf', 'Bayesian networks')],
    'vision': [('neural-networks/slides/CNN.pdf', 'Convolutional networks'), ('neural-networks/slides/object-detection.pdf', 'Object detection')],
    'sequence-models': [('advanced/slides/forecasting.pdf', 'Forecasting'), ('neural-networks/slides/rnn.pdf', 'Recurrent networks')],
}
# From the probabilistic machine learning and deep learning courses.
PROB_DIR = os.path.join(os.path.dirname(SRC), 'course-probabilistic')
DL_DIR = os.path.join(os.path.dirname(SRC), 'course-deep-learning')
MORE_NOTEBOOKS = {
    'probabilistic-ml': [('mle-univariate', 'Likelihood for a normal'), ('mle_multivariate', 'Multivariate likelihood'), ('MAP', 'Maximum a posteriori'), ('blr', 'Bayesian linear regression'), ('bayesian-logistic-regression', 'Bayesian logistic regression'), ('laplace-approx', 'The Laplace approximation'), ('mc_sampling_intro', 'Monte Carlo sampling'), ('inverse-cdf', 'Inverse CDF sampling'), ('rejection-sampling-importance-sampling', 'Rejection and importance sampling'), ('markov-chain', 'Markov chains'), ('mcmc-optimization', 'MCMC and optimisation'), ('variational-inference', 'Variational inference'), ('monte-carlo-dropout', 'Monte Carlo dropout'), ('quantile-regression', 'Quantile regression'), ('active-learning', 'Active learning'), ('bayesopt-motivation', 'Why Bayesian optimisation'), ('bayesopt', 'Bayesian optimisation')],
    'generative-models': [('vae', 'A variational autoencoder')],
    'more-ml': [('mle_bernoulli', 'Likelihood for a coin'), ('biased-mle-normal', 'Why the variance estimate is biased'), ('bayes-librarian', 'Bayes and the librarian'), ('distributions', 'Common distributions'), ('visualise_normal', 'Visualising the normal'), ('information-theory', 'Information theory'), ('calculus', 'Calculus for machine learning'), ('random-uniform', 'Generating uniform numbers'), ('sampling-normal', 'Sampling from a normal'), ('log-likelihood-linreg', 'Log-likelihood of linear regression'), ('normal_likelihood_lin_reg', 'The normal likelihood and regression'), ('mle_log_reg', 'Likelihood for logistic regression'), ('blr_posterior_figures', 'Posterior figures for linear regression'), ('mc-linreg-evidence', 'Monte Carlo evidence'), ('mc-linreg-predictive', 'Monte Carlo predictive distribution'), ('mcmc-hamiltorch', 'Hamiltonian Monte Carlo'), ('hierarchical_lr_numpyro', 'Hierarchical logistic regression'), ('baseball', 'Partial pooling on batting averages'), ('reliability', 'Reliability diagrams'), ('ssl', 'Semi-supervised learning'), ('hypernet-np', 'Hypernetworks and neural processes'), ('nn-variants-final', 'Neural network variants'), ('cats-dogs', 'Cats and dogs'), ('error-code-dilbert', 'Random number quality')],
}
PROB_SLIDES = [('Prob-refresher.pdf', 'Probability refresher'), ('MLE.pdf', 'Maximum likelihood'), ('map.pdf', 'Maximum a posteriori'), ('mvn-blr.pdf', 'Bayesian linear regression'), ('bayesian-logistic-regression.pdf', 'Bayesian logistic regression'), ('laplace-approximation.pdf', 'The Laplace approximation'), ('Sampling.pdf', 'Sampling'), ('MCMC.pdf', 'Markov chain Monte Carlo'), ('Variational-Inference.pdf', 'Variational inference'), ('Active-Learning.pdf', 'Active learning'), ('Bayesian-Optimization.pdf', 'Bayesian optimisation'), ('Information-Theory.pdf', 'Information theory')]
DL_CHEATS = {
    'estimation': [('S01', 'Where losses come from'), ('S02', 'Likelihood and classification'), ('S03', 'From likelihood to practice')],
    'neural-networks': [('S04', 'From a sum to a neuron'), ('S05', 'Gradient estimates'), ('S06', 'Reverse mode'), ('S07', 'Momentum to Adam'), ('S08', 'Vector autograd'), ('S09', 'Valid outputs'), ('S10', 'Diagnose before regularising')],
    'sequence-models': [('P01', 'From characters to next tokens')],
}
DL_FILES = os.path.join(DL_DIR, 'files')
DL_NOTEBOOKS = {
    'dl-foundations': [('01_linear_regression_gd', 'Linear regression by gradient descent'), ('02_mlp_xor', 'A network learning XOR')],
    'dl-optimization': [('00_backprop_autograd_complete', 'Backpropagation, complete'), ('04_scalar_autograd_pytorch_to_scratch', 'Autograd from PyTorch to scratch'), ('01_overfitting_and_early_stopping', 'Overfitting and early stopping'), ('02_dropout_from_scratch', 'Dropout from scratch')],
    'dl-vision': [('clip-applications', 'CLIP applications'), ('vlm-three-connections', 'Three ways to connect vision and language')],
    'generative-models': [('lecture1-variational-inference', 'Variational inference'), ('lecture5-autoencoders-vae', 'Autoencoders and VAEs')],
}
DL_LABS = [('EmbeddingGemma', 'Retrieval with embeddings', '/lectures/transformers/explore/retrieval-lab'), ('Train a classifier', 'Classes from sentences', '/lectures/transformers/explore/clip-zero-shot'), ('NMS', 'Removing duplicate boxes', '/lectures/dl-vision/explore/nms-lab'), ('Beyond Boxes', 'Dice against cross-entropy', '/lectures/dl-vision/explore/dice-vs-ce-lab')]


def dl_section(rid):
    if rid.startswith('session'):
        return 'dl-vision'
    n = int(rid.split('-')[1])
    return 'dl-foundations' if n <= 4 else 'dl-optimization' if n <= 10 else 'transformers' if n <= 15 else 'dl-vision' if n <= 23 else 'generative-models'


def dl_schedule(lectures):
    s = open(os.path.join(DL_DIR, 'site', 'schedule.qmd'), encoding='utf8').read()
    tidy = lambda v: html.unescape(re.sub('<[^>]+>', '', v)).strip()
    plan = json.load(open(os.path.join(DL_FILES, 'pdf-plan.json')))
    for r in re.split(r'(?=<tr id=")', s)[1:]:
        r = r.split('</tbody>')[0]
        rid = re.match(r'<tr id="([^"]+)"', r).group(1)
        h = re.search(r'<h3[^>]*>(.*?)</h3>', r, re.S)
        if not h or not re.match(r'(lecture-\d|session)', rid):
            continue
        sub = re.search(r'<p class="sub">(.*?)</p>', r, re.S)
        item = {'title': tidy(h.group(1)), 'sub': tidy(sub.group(1)) if sub else '', 'slides': [], 'cheats': [], 'videos': [], 'labs': []}
        names = plan.get(rid, [])
        for f in names:
            item['slides'].append({'label': 'Slides' if len(names) == 1 else ' '.join(dict.fromkeys(re.sub(r'[-_]+', ' ', f[:-4]).split())).capitalize(), 'href': copy_pdf(os.path.join(DL_FILES, 'pdf', f), 'dl-slides')})
        for m in re.finditer(r'<a class="r (r-[\w-]+)" href="([^"]+)"[^>]*?aria-label="([^"]*)"', r):
            kind, href, label = m.group(1), html.unescape(m.group(2)), re.sub(r', (lecture \d+|session.*)$', '', html.unescape(m.group(3)))
            if kind == 'r-cheat':
                name = href.split('/')[-1]
                shutil.copy(os.path.join(DL_DIR, 'slides-pdf', name), os.path.join(SITE, 'public', 'cheatsheets'))
                if not any(c['href'].endswith(name) for c in item['cheats']):
                    item['cheats'].append({'label': label, 'href': '/cheatsheets/' + name})
            elif kind in ('r-short', 'r-recording'):
                item['videos'].append({'label': label, 'href': href})
            elif kind == 'r-lab':
                lab = next(({'label': a, 'href': b} for key, a, b in DL_LABS if key in label), None)
                if lab and lab not in item['labs']:
                    item['labs'].append(lab)
        lectures.setdefault(dl_section(rid), []).append(item)
    folder = os.path.join(DL_DIR, 'site', 'Assignments')
    os.makedirs(os.path.join(SITE, 'public', 'dl-slides'), exist_ok=True)
    work = []
    for f, label in (('assignment1_optimization.md', 'Assignment 1'), ('assignment2.qmd', 'Assignment 2')):
        shutil.copy(os.path.join(folder, f), os.path.join(SITE, 'public', 'dl-slides'))
        work.append({'label': label, 'href': '/dl-slides/' + f})
    lectures.setdefault('dl-optimization', []).append({'title': 'Assignments', 'sub': '', 'slides': work, 'cheats': [], 'videos': [], 'labs': []})

QUIZZES = [('ml_2025_q1.pdf', 'Quiz 1'), ('ml_2025_q1_TA.pdf', 'Quiz 1 with solutions'), ('ml_2025_q2_TA.pdf', 'Quiz 2 with solutions'), ('ml_2025_quiz3_TA.pdf', 'Quiz 3 with solutions'), ('ml_2025_quiz4_TA_copy.pdf', 'Quiz 4 with solutions')]


def text_of(v):
    return ''.join(v) if isinstance(v, list) else (v or '')


def clean_source(src, kind):
    for url, local in DATA_URLS.items():
        src = src.replace(url, local)
    if kind == 'raw':
        return ''
    if kind == 'markdown' and src.lstrip().startswith('## Sources'):
        return ''
    if kind == 'markdown':
        src = re.sub(r'\A\s*---\n.*?\n---\s*', '', src, flags=re.S)                # publishing front matter
        src = re.sub(r'\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)', '', src)            # badges linking elsewhere
        src = re.sub(r'(?m)^\s*https?://(?!(www\.)?(youtube\.com|youtu\.be))\S+\s*$', '', src)  # bare links
    return scrub(src)


def scrub(s):
    s = re.sub(r'(?m)^#*\s*Authors?:.*$', '', s)  # bylines stay in the references page
    return re.sub(r'/(Users|home)/[A-Za-z0-9_.-]+/', '/home/user/', s).replace('nipunbatra', 'author')


def clean_text(s):
    s = scrub(re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', s))
    s = '\n'.join(line.rsplit('\r', 1)[-1] for line in s.split('\n')).rstrip()  # progress bars rewrite one line
    lines = s.split('\n')
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES] + [f'… {len(lines) - MAX_LINES} more lines']
    return '\n'.join(lines)


def png_size(path):
    with open(path, 'rb') as f:
        h = f.read(24)
    return struct.unpack('>II', h[16:24])


def save_image(data, nb, n):
    name = f'{nb}-{n}.png'
    path = os.path.join(OUT, 'img', name)
    with open(path, 'wb') as f:
        f.write(base64.b64decode(data))
    w, h = png_size(path)
    if w > MAX_W:
        subprocess.run(['sips', '--resampleWidth', str(MAX_W), path], check=True, capture_output=True)
        w, h = png_size(path)
    return {'img': f'/notebooks/img/{name}', 'w': w, 'h': h}


def convert(nb, title, folder=None):
    n = json.load(open(os.path.join(folder or os.path.join(SRC, 'ml-teaching', 'notebooks'), nb + '.ipynb'), encoding='utf8'))
    cells, images = [], 0
    for c in n.get('cells', []):
        src = clean_source(text_of(c.get('source')), c['cell_type'])
        c['source'] = src
        src = src.strip()
        if c['cell_type'] == 'markdown' and src:
            cells.append({'md': src})
        elif c['cell_type'] == 'code' and src:
            out = []
            for o in c.get('outputs', []):
                kind = o.get('output_type')
                if kind == 'stream':
                    t = clean_text(text_of(o.get('text')))
                    if t:
                        out.append({'text': t})
                elif kind in ('display_data', 'execute_result'):
                    d = o.get('data') or {}
                    if 'image/png' in d:
                        images += 1
                        out.append(save_image(text_of(d['image/png']).replace('\n', ''), nb, images))
                    elif 'text/plain' in d:
                        t = clean_text(text_of(d['text/plain']))
                        if t and not re.match(r'<(Figure|matplotlib|IPython|Axes)|^\(?<', t) and 'Javascript' not in t:
                            out.append({'text': t})
                elif kind == 'error':
                    out.append({'text': clean_text(f"{o.get('ename')}: {o.get('evalue')}")})
            cells.append({'code': src, 'out': out})
    with open(os.path.join(OUT, nb + '.json'), 'w', encoding='utf8') as f:
        json.dump({'id': nb, 'title': title, 'cells': cells}, f, ensure_ascii=False, separators=(',', ':'))
    # The downloadable copy has no saved outputs, because the page already shows them.
    for c in n.get('cells', []):
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
    n['cells'] = [c for c in n.get('cells', []) if c['source'].strip()]
    n['metadata'] = {k: v for k, v in n.get('metadata', {}).items() if k in ('kernelspec', 'language_info')}
    with open(os.path.join(OUT, 'files', nb + '.ipynb'), 'w', encoding='utf8') as f:
        json.dump(n, f, ensure_ascii=False, indent=1)
    return images


COPIED = {}


def copy_pdf(src, folder):
    name = os.path.basename(src)
    assert COPIED.setdefault((folder, name), src) == src, f'two files named {name}'
    os.makedirs(os.path.join(SITE, 'public', folder), exist_ok=True)
    shutil.copy(src, os.path.join(SITE, 'public', folder, name))
    return f'/{folder}/{name}'


def local_slide(href):
    m = re.search(r'/ml-teaching/(.*)$', href)
    src = os.path.join(SRC, 'ml-teaching', m.group(1)) if m else os.path.join(SRC, 'downloads', 'slides', href.rsplit('/', 1)[-1])
    return copy_pdf(src, 'slides')


def local_lab(href):
    return next({'label': label, 'href': path} for key, label, path in LOCAL_LABS if key in href.lower())


def schedule():
    s = open(os.path.join(SRC, 'ml2025', 'schedule.qmd'), encoding='utf8').read()
    tidy = lambda v: html.unescape(re.sub('<[^>]+>', '', v)).strip()
    out = {}
    for r in re.split(r'(?=<tr id=")', s)[1:]:
        r = r.split('</tbody>')[0]
        rid = re.match(r'<tr id="([^"]+)"', r).group(1)
        h = re.search(r'<h3[^>]*>(.*?)</h3>', r, re.S)
        if not h or rid not in SECTION_OF_LECTURE:
            continue
        sub = re.search(r'<p class="sub">(.*?)</p>', r, re.S)
        item = {'title': tidy(h.group(1)), 'sub': tidy(sub.group(1)) if sub else '', 'slides': [], 'cheats': [], 'videos': [], 'labs': []}
        for m in re.finditer(r'<a class="r (r-[\w-]+)" href="([^"]+)"[^>]*?aria-label="([^"]*)"', r):
            kind, href, label = m.group(1), html.unescape(m.group(2)), re.sub(r', (lecture \d+|topic.*)$', '', html.unescape(m.group(3)))
            if kind == 'r-cheat':
                item['cheats'].append({'label': label, 'href': '/cheatsheets/' + href.split('/')[-1]})
            elif kind == 'r-short':
                item['videos'].append({'label': label, 'href': href})
            elif kind == 'r-lab' and 'flappy' not in href:  # flappy needs a desktop game window, so it stays in the notebook only
                lab = local_lab(href)
                if lab not in item['labs']:
                    item['labs'].append(lab)
            elif kind in ('r-slides', 'r-notes'):
                item['slides'].append({'label': label, 'href': local_slide(href)})
        out.setdefault(SECTION_OF_LECTURE[rid], []).append(item)
    return out


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, 'img'))
    os.makedirs(os.path.join(OUT, 'files', 'data'))
    for folder in ('slides', 'handouts', 'quizzes', 'dl-slides'):
        shutil.rmtree(os.path.join(SITE, 'public', folder), ignore_errors=True)
    nb_dir = os.path.join(SRC, 'ml-teaching', 'notebooks')
    for f in os.listdir(nb_dir):
        if f.endswith(NOTEBOOK_FILES):
            shutil.copy(os.path.join(nb_dir, f), os.path.join(OUT, 'files'))
    for local in DATA_URLS.values():
        shutil.copy(os.path.join(SRC, 'downloads', local), os.path.join(OUT, 'files', 'data'))
    have = {f[:-6] for f in os.listdir(os.path.join(SRC, 'ml-teaching', 'notebooks')) if f.endswith('.ipynb')}
    listed = [nb for rows in NOTEBOOKS.values() for nb, _ in rows]
    assert len(listed) == len(set(listed)), 'a notebook is listed twice'
    assert set(listed) == have, f'unassigned or missing: {sorted(have ^ set(listed))}'
    images = sum(convert(nb, title) for rows in NOTEBOOKS.values() for nb, title in rows)
    images += sum(convert(nb, title, os.path.join(PROB_DIR, 'notebooks')) for rows in MORE_NOTEBOOKS.values() for nb, title in rows)
    images += sum(convert(nb, title, os.path.join(DL_FILES, 'notebooks')) for rows in DL_NOTEBOOKS.values() for nb, title in rows)

    cheat = os.path.join(SITE, 'public', 'cheatsheets')
    os.makedirs(cheat, exist_ok=True)
    src = os.path.join(SRC, 'ml2025', 'cheatsheets')
    for f in os.listdir(src):
        if f.endswith('.pdf'):
            shutil.copy(os.path.join(src, f), cheat)

    lectures = schedule()
    dl_schedule(lectures)
    for section, rows in MORE_SLIDES.items():
        lectures.setdefault(section, []).append({'title': 'More slides', 'sub': '', 'slides': [{'label': label, 'href': copy_pdf(os.path.join(SRC, 'ml-teaching', p), 'slides')} for p, label in rows], 'cheats': [], 'videos': [], 'labs': []})
    lectures.setdefault('probabilistic-ml', []).append({'title': 'Slides', 'sub': '', 'slides': [{'label': label, 'href': copy_pdf(os.path.join(PROB_DIR, 'slides', f), 'slides')} for f, label in PROB_SLIDES], 'cheats': [], 'videos': [], 'labs': []})
    for section, rows in DL_CHEATS.items():
        cheats = []
        for code, label in rows:
            shutil.copy(os.path.join(DL_DIR, 'slides-pdf', f'{code}-cheatsheet.pdf'), cheat)
            cheats.append({'label': label, 'href': f'/cheatsheets/{code}-cheatsheet.pdf'})
        lectures.setdefault(section, []).append({'title': 'More cheat sheets', 'sub': '', 'slides': [], 'cheats': cheats, 'videos': [], 'labs': []})
    lectures.setdefault('practice', []).append({'title': 'Past quizzes', 'sub': 'Closed-book quizzes from the 2025 offering.', 'slides': [{'label': label, 'href': copy_pdf(os.path.join(SRC, 'ml2025', 'quizzes', f), 'quizzes')} for f, label in QUIZZES], 'cheats': [], 'videos': [], 'labs': []})
    for section, rows in HANDOUTS.items():
        lectures.setdefault(section, []).append({'title': 'Tutorial handouts', 'sub': 'Worked explanations with practice problems.', 'slides': [{'label': label, 'href': copy_pdf(os.path.join(SRC, 'ml-teaching', p), 'handouts')} for p, label in rows], 'cheats': [], 'videos': [], 'labs': []})
    for section, items in lectures.items():
        for item in items:
            for c in item['cheats']:
                assert os.path.exists(os.path.join(SITE, 'public', c['href'][1:])), c['href']
    head = '// Generated by scripts/import-course.py. Do not edit by hand.\n'
    with open(os.path.join(SITE, 'lib', 'resources.mjs'), 'w', encoding='utf8') as f:
        f.write(head + 'export const RESOURCES = ' + json.dumps(lectures, ensure_ascii=False, indent=1) + ';\n')
    # Kept separate because the sidebar needs it on every page, and the full list is large.
    nav = {}
    for table in (NOTEBOOKS, MORE_NOTEBOOKS, DL_NOTEBOOKS):
        for s, rows in table.items():
            nav.setdefault(s, []).extend({'id': nb, 'title': t} for nb, t in rows)
    with open(os.path.join(SITE, 'lib', 'notebooks.mjs'), 'w', encoding='utf8') as f:
        f.write(head + 'const NOTEBOOKS = ' + json.dumps(nav, ensure_ascii=False, separators=(',', ':')) + ';\n')
        f.write('export const WITH_RESOURCES = ' + json.dumps(sorted(lectures)) + ';\n')
        f.write('export const notebooksOf = id => NOTEBOOKS[id] ?? [];\n')
    print('notebooks', len(listed), 'images', images, 'sections', len(lectures))


main()
