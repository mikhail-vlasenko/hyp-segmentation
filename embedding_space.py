"""
Embedding space. Largely copied from the
Hyperbolic Image Segmentation (Atigh et al., 2022) author implementation:
https://github.com/MinaGhadimiAtigh/HyperbolicImageSegmentation

Please consider citing this work:
@article{ghadimiatigh2022hyperbolic,
  title={Hyperbolic Image Segmentation},
  author={GhadimiAtigh, Mina and Schoep, Julian and Acar, Erman and van Noord, Nanne and Mettes, Pascal},
  journal={arXiv preprint arXiv:2203.05898},
  year={2022}
}
"""

import logging
from abc import ABC, abstractmethod

from torch import nn
import torch

from hyperbolic_layers import hyp_mlr_torch


class AbstractEmbeddingSpace(nn.Module, ABC):
    """
    General purpose base class for implementing embedding spaces.
    Softmax function turns into hierarchical softmax only when the 'tree' has an hierarchal structure defined.
    """

    def __init__(self, offsets, normals, curvature, train: bool = True, prototype_path: str = ''):
        super(AbstractEmbeddingSpace, self).__init__()

    def run_log_torch(self, embeddings, offsets, normals, curvature=None):
        """ Calculates (joint) probabilities for incoming embeddings. Assumes embeddings are already on manifold. """
        logits = self.logits_torch(embeddings=embeddings, offsets=offsets, normals=normals, curvature=curvature)
        assert not torch.isnan(logits).any()
        return logits

    @abstractmethod
    def logits_torch(self, embeddings, offsets, normals, curvature):
        """ Returns logits to pass to (hierarchical) softmax function."""
        pass


class HyperbolicEmbeddingSpace(AbstractEmbeddingSpace):
    def __init__(self, offsets, normals, curvature, train: bool = True, prototype_path: str = ''):
        super().__init__(offsets, normals, curvature, train=train, prototype_path=prototype_path)

    def logits_torch(self, embeddings, offsets, normals, curvature):
        return hyp_mlr_torch(
            embeddings,
            c=curvature,
            P_mlr=offsets,
            A_mlr=normals
        )


def EmbeddingSpace(offsets, normals, curvature, train: bool = True, prototype_path: str = ''):
    """Returns correct embedding space obj based on config."""
    base = HyperbolicEmbeddingSpace


    class EmbeddingSpace(base):
        def __init__(self,offsets, normals, curvature, train: bool = True, prototype_path: str = ''):
            super().__init__( offsets, normals, curvature, train=train, prototype_path=prototype_path)

    return EmbeddingSpace(offsets, normals, curvature, train=train, prototype_path=prototype_path)
