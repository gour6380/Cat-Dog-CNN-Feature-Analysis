# Third-party data and software

## Oxford-IIIT Pet

The experiment uses the Oxford-IIIT Pet dataset from the University of Oxford Visual Geometry
Group. The dataset is obtained separately from its
[official page](https://www.robots.ox.ac.uk/~vgg/data/pets/) and is licensed by its authors under
CC BY-SA 4.0. No dataset photograph, annotation archive, or extracted dataset file is distributed
in this repository.

## Model initialization and dependencies

The experiment downloads Torchvision's `ResNet18_Weights.IMAGENET1K_V1` tensors into ignored local
storage. Those weights are not redistributed here. PyTorch, Torchvision, scikit-learn, UMAP, and
the other packages in `requirements.txt` remain subject to their respective upstream licenses.
The project MIT license applies only to original project code and documentation.
