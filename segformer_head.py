import torch
from torch import nn
import math

import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath
from transformers import SegformerDecodeHead

from embedding_space import EmbeddingSpace


class HyperbolicSegformerDecodeHead(SegformerDecodeHead):
    def __init__(self):
        raise NotImplementedError

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        self.embedding_space = EmbeddingSpace(self.offsets, self.normals, self.curvature)
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

        # print(f"{hidden_states.shape=}")  # batch_size, 256, height/4, width/4

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        # logits = self.classifier(hidden_states)
        # do the hyperbolic stuff here
        ball = gt.PoincareBall(c=1.0)
        output = hidden_states.permute((0, 2, 3, 1))
        out_proj = ball.expmap0(output)  # seems to just map the vector to the ball
        output = self.embedding_space.run_log_torch(out_proj, self.offsets, self.normals, 1.0)
        logits = output.permute((0, 3, 1, 2))

        return logits

    def __post_init__(self, num_classes):
        self.dim = 256
        self.ball = gt.PoincareBall(c=1.0)
        normals_ = torch.randn(num_classes, self.dim) * 1e-5
        normals_ = pmath.expmap0(normals_, k=self.ball.k)
        self.normals = gt.ManifoldParameter(normals_, manifold=self.ball, requires_grad=True)

        offsets_ = torch.zeros(num_classes, self.dim)
        offsets_ = pmath.expmap0(offsets_, k=self.ball.k)
        self.offsets = gt.ManifoldParameter(offsets_, manifold=self.ball, requires_grad=True)

        self.curvature = torch.tensor(1.0)

        self.normals.requires_grad_()
        self.offsets.requires_grad_()

    @classmethod
    def from_segformer_decode_head(cls, segformer_decode_head: SegformerDecodeHead, num_classes) -> "HyperbolicSegformerDecodeHead":
        segformer_decode_head.__class__ = cls

        segformer_decode_head.__post_init__(num_classes)

        return segformer_decode_head
