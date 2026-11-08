// Descriptions for pictures whose captions in the source notes only point at a section. Key: image file name.
export const CAPTIONS = {
  's03_dataset.png': 'Slide 3: the eight emails on two axes, the number of links and the number of capitalised words, marked as spam or not spam.',
  's056_round2_question.png': 'Slide 56: round two of boosting. Runs 5 and 6 now have the largest residuals, and the question batch ≤ 48 wins with the highest split score.',
  's064_boosting_summary.png': 'Slide 64: the whole gradient boosting procedure as a table, from residuals to the question, the leaf values and the update.',
  'r2_multiplier_confidence.png': 'Left: how much each point’s weight is multiplied, against its loss, for three values of β. Right: a learner’s confidence against its average loss, with the six rounds marked.',
  's072_round1_weights.png': 'Slide 72: AdaBoost round one. The ten correct runs are multiplied by 0.447 and the two mistakes by 2.236, so after normalizing each mistake carries weight 0.25.',
  's074_second_stump_answer.png': 'Slide 74: the second stump has weighted error 0.2 and earns a vote of 0.693. Reusing the first stump would score exactly 0.5 and earn no vote.',
  's075_round2_weights.png': 'Slide 75: AdaBoost round two. The four mistakes are doubled and the correct runs halved, before normalizing.',
  's083_what_xgb_changes.png': 'Slide 83: what XGBoost changes from gradient boosting. Slopes and curvatures choose the question, and penalties shrink the leaves.',
  's086_gradients_curvatures.png': 'Slide 86: the settings for round one, each run’s slope and curvature, and the sums for the parent node.',
  's087_root_question.png': 'Slide 87: every candidate root question with the slopes and curvatures of its two groups, and its gain. The rate ≤ 0.065 question wins.',
  's089_leaf_weights.png': 'Slide 89: each leaf’s weight with and without the penalty. Every weight is smaller with λ = 1, by the factor H/(H+λ).',
  's096_xgb_summary.png': 'Slide 96: the whole XGBoost procedure as a table, from slopes and curvatures to the gain, the leaf weights and the update.',
  's098_bagging_vs_boosting.png': 'Slide 98: bagging and boosting compared on how trees are grown, how deep they are, how they are combined and what they mainly reduce.',
  's099_three_boosters.png': 'Slide 99: gradient boosting, AdaBoost and XGBoost side by side, on what each tree fits, the loss reduced and how the question is chosen.',
  's100_predictions_A_B.png': 'Slide 100: every method’s prediction for the two new runs A and B, each with its score or vote.',
};
