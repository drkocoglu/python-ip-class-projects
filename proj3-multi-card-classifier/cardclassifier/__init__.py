"""cardclassifier: from-scratch playing-card rank & suit classification.

Python port of a MATLAB image-processing course project by Yildirim Kocoglu
and Farshad Bolouri. All image processing is implemented with NumPy only; the
single ML component is the rank SVM (scikit-learn), as permitted.
"""

from . import (
    augment,
    binarize,
    edges,
    enhance,
    filters,
    hog,
    io_utils,
    morphology,
    pipeline,
    rank_classifier,
    regionprops,
    rotation,
    segmentation,
    shape_features,
    suit_classifier,
    training,
)

__all__ = [
    "augment", "binarize", "edges", "enhance", "filters", "hog", "io_utils",
    "morphology", "pipeline", "rank_classifier", "regionprops", "rotation",
    "segmentation", "shape_features", "suit_classifier", "training",
]
__version__ = "2.0.0"
