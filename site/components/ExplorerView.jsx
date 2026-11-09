'use client';
import { lazy, Suspense } from 'react';

// Each catalogue row in lib/lectures.mjs names one of these modules. Importing on demand keeps the other explorers out of the page's download.
const LOADERS = {
  L01: () => import('../explorers/l01'),
  Maths: () => import('../explorers/maths'),
  Regression: () => import('../explorers/regression'),
  Evaluation: () => import('../explorers/evaluation'),
  Svm: () => import('../explorers/svm'),
  Trees: () => import('../explorers/trees'),
  Practice: () => import('../explorers/practice'),
  Foundations: () => import('../explorers/foundations'),
  Modelling: () => import('../explorers/modelling'),
  Methods: () => import('../explorers/methods'),
  Tutorials: () => import('../explorers/tutorials'),
  Unsupervised: () => import('../explorers/unsupervised'),
  Clusters: () => import('../explorers/clusters'),
  Reduction: () => import('../explorers/reduction'),
  Anomaly: () => import('../explorers/anomaly'),
  Classical: () => import('../explorers/classical'),
  Neuralnets: () => import('../explorers/neuralnets'),
  Vision: () => import('../explorers/vision'),
  Sequences: () => import('../explorers/sequences'),
  Additions: () => import('../explorers/additions'),
  Modern: () => import('../explorers/modern'),
  Beyond: () => import('../explorers/beyond'),
  Deeplearning: () => import('../explorers/deeplearning'),
  Moreml: () => import('../explorers/moreml'),
  Frontier: () => import('../explorers/frontier'),
  Extensions: () => import('../explorers/extensions'),
  Dlviz: () => import('../explorers/dlviz'),
  Coverage: () => import('../explorers/coverage'),
  Gaps: () => import('../explorers/gaps'),
  Completions: () => import('../explorers/completions'),
};
const loaded = new Map();
function load(module, component) {
  const key = `${module}/${component}`;
  if (!loaded.has(key)) loaded.set(key, lazy(() => LOADERS[module]().then(m => ({ default: m[component] }))));
  return loaded.get(key);
}

export default function ExplorerView({ module, component }) {
  const C = load(module, component);
  return <Suspense fallback={null}><C /></Suspense>;
}
