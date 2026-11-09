export const metadata = { title: 'Epocher - References' };

const SOURCES = [
  ['ES 335 / ES 678 Machine Learning, Semester I 2026, IIT Gandhinagar', 'Prof. Udit Bhatia, with Prof. Krishna Prasad Miyapuram. Lectures, class notes, tutorials and course documents.'],
  ['ES 335 Machine Learning, Semester I 2025–26, IIT Gandhinagar', 'Prof. Nipun Batra. Slides, cheat sheets, handouts, notebooks, quizzes and one-minute videos.'],
  ['ES 667 Deep Learning, August 2026, IIT Gandhinagar', 'Prof. Nipun Batra. Slides, notes, cheat sheets, notebooks, assignments and lecture recordings.'],
  ['ES 661 Probabilistic Machine Learning, IIT Gandhinagar', 'Prof. Nipun Batra. Slides and notebooks.'],
];

const PAPERS = [
  'Kingma and Ba, Adam: A Method for Stochastic Optimization, 2015',
  'Loshchilov and Hutter, Decoupled Weight Decay Regularization, 2019',
  'Ioffe and Szegedy, Batch Normalization, 2015',
  'He et al., Deep Residual Learning for Image Recognition, 2016',
  'Ronneberger et al., U-Net, 2015',
  'Tan and Le, EfficientNet, 2019',
  'Hochreiter and Schmidhuber, Long Short-Term Memory, 1997',
  'Vaswani et al., Attention Is All You Need, 2017',
  'Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021',
  'Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, 2020',
  'Kingma and Welling, Auto-Encoding Variational Bayes, 2014',
  'Goodfellow et al., Generative Adversarial Nets, 2014',
  'Ho et al., Denoising Diffusion Probabilistic Models, 2020',
  'Song et al., Denoising Diffusion Implicit Models, 2021',
  'Ho and Salimans, Classifier-Free Diffusion Guidance, 2022',
  'Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, 2022',
  'Lipman et al., Flow Matching for Generative Modeling, 2023',
  'Kipf and Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2017',
  'Veličković et al., Graph Attention Networks, 2018',
  'Chen et al., A Simple Framework for Contrastive Learning of Visual Representations, 2020',
  'He et al., Masked Autoencoders Are Scalable Vision Learners, 2022',
  'Williams, Simple Statistical Gradient-Following Algorithms, 1992',
  'Schulman et al., Proximal Policy Optimization Algorithms, 2017',
  'Mnih et al., Human-Level Control through Deep Reinforcement Learning, 2015',
  'van Hasselt et al., Deep Reinforcement Learning with Double Q-Learning, 2016',
  'Lundberg and Lee, A Unified Approach to Interpreting Model Predictions, 2017',
  'Chouldechova, Fair Prediction with Disparate Impact, 2017',
  'Nelder and Wedderburn, Generalized Linear Models, 1972',
  'Koenker and Bassett, Regression Quantiles, 1978',
  'Kalman, A New Approach to Linear Filtering and Prediction Problems, 1960',
  'Rosenbaum and Rubin, The Central Role of the Propensity Score, 1983',
  'Auer et al., Finite-Time Analysis of the Multiarmed Bandit Problem, 2002',
  'Li et al., A Contextual-Bandit Approach to Personalized News Article Recommendation, 2010',
  'Jones et al., Efficient Global Optimization of Expensive Black-Box Functions, 1998',
  'Li et al., Hyperband, 2018',
  'Beck and Teboulle, A Fast Iterative Shrinkage-Thresholding Algorithm, 2009',
  'Burges et al., Learning to Rank using Gradient Descent, 2005',
  'Dwork et al., Calibrating Noise to Sensitivity in Private Data Analysis, 2006',
  'Abadi et al., Deep Learning with Differential Privacy, 2016',
  'McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data, 2017',
  'Ke et al., LightGBM, 2017',
  'Prokhorenkova et al., CatBoost, 2018',
  'Goodfellow et al., Explaining and Harnessing Adversarial Examples, 2015',
  'Madry et al., Towards Deep Learning Models Resistant to Adversarial Attacks, 2018',
  'Gal and Ghahramani, Dropout as a Bayesian Approximation, 2016',
  'Lakshminarayanan et al., Simple and Scalable Predictive Uncertainty Estimation, 2017',
  'Hinton et al., Distilling the Knowledge in a Neural Network, 2015',
  'Dinh et al., Density Estimation using Real NVP, 2017',
  'Vincent, A Connection Between Score Matching and Denoising Autoencoders, 2011',
  'Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, 2021',
  'Song et al., Consistency Models, 2023',
  'Jacot et al., Neural Tangent Kernel, 2018',
  'Frankle and Carbin, The Lottery Ticket Hypothesis, 2019',
  'Zaheer et al., Deep Sets, 2017',
  'Finn et al., Model-Agnostic Meta-Learning, 2017',
  'Kirkpatrick et al., Overcoming Catastrophic Forgetting in Neural Networks, 2017',
  'Chen et al., Neural Ordinary Differential Equations, 2018',
  'Raissi et al., Physics-Informed Neural Networks, 2019',
  'Gorishniy et al., Revisiting Deep Learning Models for Tabular Data, 2021',
  'Haarnoja et al., Soft Actor-Critic, 2018',
  'Kumar et al., Conservative Q-Learning for Offline Reinforcement Learning, 2020',
  'Sutton, Dyna, an Integrated Architecture for Learning, Planning, and Reacting, 1991',
  'Mildenhall et al., NeRF, 2020',
  'Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering, 2023',
  'Bricken et al., Towards Monosemanticity, 2023',
];

export default function References() {
  return (
    <div>
      <h1>References</h1>
      <p>Written and organised by Sumit Jha, from the material below.</p>
      <ul>{SOURCES.map(([what, who]) => <li key={what}><b>{what}.</b> {who}</li>)}</ul>
      <h2>Papers behind the added material</h2>
      <ul>{PAPERS.map(p => <li key={p}>{p}</li>)}</ul>
    </div>
  );
}
