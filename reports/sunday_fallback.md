# Why PCA, t-SNE, and UMAP can disagree without proving robustness

This is the registered fallback topic, not a publish-ready post. PCA preserves maximum
linear variance, t-SNE emphasizes local neighbourhood probabilities in a jointly fitted
sample, and UMAP builds a neighbourhood graph under chosen hyperparameters. Each can
show a different two-dimensional story even when the original 512-dimensional evidence
is unchanged. A projection is an explanation aid; robustness must be tested with aligned
features, valid attacks, quantitative metrics, uncertainty, and explicit claim limits.
