import torch
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors.tangent_tensor import TangentTensor
import time


manifold = PoincareBall(c=Curvature(1.0))
output = torch.randn(35000, 200, device="cuda", requires_grad=True)
prototypes = torch.randn(201, 200, device="cuda", requires_grad=False)

output_hyperbolic = manifold.expmap(TangentTensor(data=output, man_dim=-1, manifold=manifold))
prototypes = manifold.expmap(TangentTensor(data=prototypes, man_dim=-1, manifold=manifold))

# does not work
# distances = manifold.dist(output_hyperbolic, prototypes).T

distances = []
start = time.time()
for i in range(prototypes.shape[0]):
    distances.append(manifold.dist(output_hyperbolic, prototypes[i].unsqueeze(0)))
distances = torch.stack(distances).T
print(time.time() - start)

print(distances.shape)
