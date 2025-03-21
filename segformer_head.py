import torch
from torch import nn
import math

import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath
from transformers import SegformerDecodeHead

from embedding_space import EmbeddingSpace
from hyperbolic_layers import fast_dist
from prototypes import create_prototypes


class HyperbolicSegformerDecodeHead(SegformerDecodeHead):
    def __init__(self):
        raise NotImplementedError

    def forward(self, encoder_hidden_states: torch.FloatTensor, return_repr: bool = False) -> torch.Tensor:
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
        if self.hyperbolic:
            if self.max_class_sep:
                hidden_states = self.classifier(hidden_states)
            ball = gt.PoincareBall(c=self.curvature)
            output = hidden_states.permute(0, 2, 3, 1)
            rep = ball.expmap0(output)
            if self.max_class_sep:
                rep_shape = rep.shape
                rep = rep.view(-1, rep.shape[-1])
                distances = fast_dist(rep, self.prototypes, self.ball.k).T
                logits = (-1 * distances * self.tau).reshape(*rep_shape[:-1], self.prototypes.shape[0])
            else:
                self.embedding_space = EmbeddingSpace(self.offsets, self.normals, self.curvature)
                logits = self.embedding_space.run_log_torch(rep, self.offsets, self.normals, self.curvature)
            logits = logits.permute(0, 3, 1, 2)
        else:
            logits = self.classifier(hidden_states)
            if self.max_class_sep:
                # logits are of shape (batch, num_classes - 1, h, w)
                # prototypes are of shape (num_classes, num_classes - 1)
                # we want (batch, num_classes, h, w) on output
                logits = torch.einsum("bchw,nc->bnhw", logits, self.prototypes)
            rep = hidden_states

        if return_repr:
            return logits, rep
        return logits

    def __post_init__(self, num_classes, dim, hyperbolic, curvature, max_class_sep, tau):
        if dim is None:
            self.dim = self.config.decoder_hidden_size
        else:
            self.dim = dim
        
        self.max_class_sep = max_class_sep
        self.tau = tau
        if self.max_class_sep:
            prototypes = create_prototypes(num_classes)
            prototypes = torch.from_numpy(prototypes).float()
            # change num_classes to num_classes - 1 as the network should now output that dimension
            num_classes = prototypes.shape[1]
            if hyperbolic:
                prototypes = prototypes * 0.95  # downscale to have prototypes in the ball, not on the boundary
                prototypes = prototypes.unsqueeze(1)
            self.prototypes = torch.nn.Parameter(prototypes, requires_grad=False)

        self.hyperbolic = hyperbolic

        self.curvature = curvature
        self.ball = gt.PoincareBall(c=self.curvature)

        if self.hyperbolic and not self.max_class_sep:
            normals_ = torch.randn(num_classes, self.dim) * 1e-5
            normals_ = pmath.expmap0(normals_, k=self.ball.k)
            self.normals = gt.ManifoldParameter(normals_, manifold=self.ball, requires_grad=True)

            offsets_ = torch.zeros(num_classes, self.dim)
            offsets_ = pmath.expmap0(offsets_, k=self.ball.k)
            self.offsets = gt.ManifoldParameter(offsets_, manifold=self.ball, requires_grad=True)

            self.normals.requires_grad_()
            self.offsets.requires_grad_()

        self.dim_reduce = nn.Identity()
        if self.dim != self.config.decoder_hidden_size:
            self.dim_reduce = nn.Sequential(
                nn.Conv2d(self.config.decoder_hidden_size, self.dim, kernel_size=1),
                nn.BatchNorm2d(self.dim),
                # nn.ReLU(),  # leads to a failing assert on NaNs in hyperbolic almost immediately
            )

        self.classifier = nn.Conv2d(self.dim, num_classes, kernel_size=1)
        self.num_classes = num_classes


    @classmethod
    def from_segformer_decode_head(
            cls, segformer_decode_head: SegformerDecodeHead,
            num_classes, dim, hyperbolic, curvature, max_class_sep, tau
    ) -> "HyperbolicSegformerDecodeHead":
        segformer_decode_head.__class__ = cls

        segformer_decode_head.__post_init__(num_classes, dim, hyperbolic, curvature, max_class_sep, tau)

        return segformer_decode_head
