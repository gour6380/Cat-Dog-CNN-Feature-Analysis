# Third-party data and software

## Oxford-IIIT Pet

The experiment uses the Oxford-IIIT Pet dataset from the University of Oxford Visual Geometry
Group. The dataset is obtained separately from its
[official page](https://www.robots.ox.ac.uk/~vgg/data/pets/) and is licensed by its authors under
CC BY-SA 4.0. Dataset photographs, real calibration patches, and photo-containing
activation/Grad-CAM overlays remain ignored local artifacts, as do annotations and
extracted files. They are not distributed in the repository. A later separate
decision to share any dataset-derived photograph requires attribution and a license
review; the project MIT license does not relicense those images.

## Model initialization and dependencies

The experiment downloads Torchvision's `ResNet18_Weights.IMAGENET1K_V1` tensors into ignored local
storage. Those weights are not redistributed here. PyTorch, Torchvision, scikit-learn, UMAP, and
the other packages in `requirements.txt` remain subject to their respective upstream licenses.
The project MIT license applies only to original project code and documentation.

## Visualization method attribution

The low/mid/high feature walkthrough is inspired by
[Lee et al., 2009](https://ai.stanford.edu/~ang/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf),
not a reproduction of their convolutional deep belief network. The implementation
also draws method ideas from [Zeiler and Fergus](https://arxiv.org/abs/1311.2901),
[Grad-CAM](https://arxiv.org/abs/1610.02391),
[saliency-map sanity checks](https://arxiv.org/abs/1810.03292), and
[Distill's Feature Visualization](https://distill.pub/2017/feature-visualization/).
No figure from those works is copied into the repository. Synthetic images generated
by this study are labeled as optimized stimuli, not original dataset photographs.
