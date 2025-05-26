from dataclasses import dataclass, field
from typing import Union, Tuple, Dict

import torch
from torch import nn
import math

import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath
from transformers import SegformerDecodeHead

from embedding_space import EmbeddingSpace
from hyperbolic_layers import fast_dist
from prototypes import create_prototypes
from utils import num_labels_for_granularity


@dataclass
class HeadKwargs:
    primary_granularity: str = None
    dim: int = None
    hyperbolic: bool = False
    curvature: float = 0.1
    max_class_sep: bool = False
    tau: float = 0.1
    embeddings_paths: Dict[str, str] = field(default_factory=dict)
    independent_heads: bool = False


@dataclass
class HeadReturnType:
    logits: Dict[str, torch.Tensor] = field(default_factory=dict)
    repr: torch.Tensor = None


class HyperbolicSegformerDecodeHead(SegformerDecodeHead):
    def __init__(self):
        raise NotImplementedError

    def forward(
            self, encoder_hidden_states: torch.FloatTensor, eval_mode: bool = False
    ) -> HeadReturnType:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.dim_reduce(hidden_states)

        # print(f"{hidden_states.shape=}")  # batch_size, 256, height/4, width/4

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        result = HeadReturnType()
        if self.hyperbolic:
            if self.max_class_sep:
                hidden_states = self.classifier(hidden_states)
            ball = gt.PoincareBall(c=self.curvature)
            output = hidden_states.permute(0, 2, 3, 1)
            rep = ball.expmap0(output)
            if self.max_class_sep:
                rep_shape = rep.shape
                rep = rep.view(-1, rep.shape[-1])
                for key, value in self.prototypes.items():
                    result.logits[key] = self.prototypes_logits(rep, rep_shape, value)
            else:
                embedding_space = EmbeddingSpace(self.offsets, self.normals, self.curvature)
                result.logits[self.primary_granularity] = embedding_space.run_log_torch(rep, self.offsets, self.normals, self.curvature)
            for key in result.logits:
                result.logits[key] = result.logits[key].permute(0, 3, 1, 2)
        else:
            primary_logits = self.classifier(hidden_states)
            if self.max_class_sep:
                # logits are of shape (batch, num_classes - 1, h, w)
                # prototypes are of shape (num_classes, num_classes - 1)
                # we want (batch, num_classes, h, w) on output
                for key, value in self.prototypes.items():
                    result.logits[key] = torch.einsum("bchw,nc->bnhw", primary_logits, value)
            else:
                result.logits[self.primary_granularity] = primary_logits
            rep = hidden_states
        result.repr = rep
        return result

    def __post_init__(self, args: HeadKwargs):
        if args.dim is None:
            self.dim = self.config.decoder_hidden_size
        else:
            self.dim = args.dim

        self.max_class_sep = args.max_class_sep
        self.tau = args.tau
        self.hyperbolic = args.hyperbolic
        self.primary_granularity = args.primary_granularity
        self.num_classes = num_labels_for_granularity(self.primary_granularity)

        self.prototypes = {}
        for key, value in args.embeddings_paths.items():
            # independent_heads ensures there is at most one prototype set for each head
            if not args.independent_heads or key == self.primary_granularity:
                # value = value.replace("/home/mvlasenko/hyperbolic/hyp-segmentation/h_embeds/",
                #               "hierarchy_embeddings/hierarchies/hierarchy_embeddings/experiments/spin_dataset/spin_hierarchy_part-first/")
                self.prototypes[key] = torch.load(value, weights_only=False).embeddings.weight.tensor

        if self.max_class_sep:
            if len(self.prototypes) > 0:
                # these should be already in the ball
                norm = torch.norm(self.prototypes[self.primary_granularity], dim=1, p=2)
                assert norm.max() < 1.0, f"Embeddings are not in the ball, max norm is {norm.max()}"
            else:
                prototypes = create_prototypes(self.num_classes)
                prototypes = torch.from_numpy(prototypes).float()
                if self.hyperbolic:
                    prototypes = prototypes * 0.95  # downscale to have prototypes in the ball, not on the boundary
                self.prototypes[self.primary_granularity] = prototypes

            self.num_classes = self.prototypes[self.primary_granularity].shape[1]  # make the decoder compress to the right channel dimension

            for key, value in self.prototypes.items():
                if self.hyperbolic:
                    value = value.unsqueeze(1)
                self.prototypes[key] = torch.nn.Parameter(value, requires_grad=False)

            self.prototypes = nn.ParameterDict(self.prototypes)

        self.curvature = args.curvature
        self.ball = gt.PoincareBall(c=self.curvature)

        if self.hyperbolic and not self.max_class_sep:
            normals_ = torch.randn(self.num_classes, self.dim) * 1e-5
            normals_ = pmath.expmap0(normals_, k=self.ball.k)
            self.normals = gt.ManifoldParameter(normals_, manifold=self.ball, requires_grad=True)

            offsets_ = torch.zeros(self.num_classes, self.dim)
            offsets_ = pmath.expmap0(offsets_, k=self.ball.k)
            self.offsets = gt.ManifoldParameter(offsets_, manifold=self.ball, requires_grad=True)

            self.normals.requires_grad_()
            self.offsets.requires_grad_()

        self.dim_reduce = nn.Identity()
        if self.dim != self.config.decoder_hidden_size:
            # do not include non-linearity as we only need to reduce the representation dimension
            self.dim_reduce = nn.Sequential(
                nn.Conv2d(self.config.decoder_hidden_size, self.dim, kernel_size=1),
                # nn.BatchNorm2d(self.dim),
            )

        self.classifier = nn.Conv2d(self.dim, self.num_classes, kernel_size=1)

    @classmethod
    def from_segformer_decode_head(
            cls, segformer_decode_head: SegformerDecodeHead, args: HeadKwargs
    ) -> "HyperbolicSegformerDecodeHead":
        segformer_decode_head.__class__ = cls

        segformer_decode_head.__post_init__(args)

        return segformer_decode_head

    def prototypes_logits(self, rep: torch.Tensor, rep_shape, prototypes) -> torch.Tensor:
        distances = fast_dist(rep, prototypes, self.ball.k).T
        return (-1 * distances * self.tau).reshape(*rep_shape[:-1], prototypes.shape[0])
