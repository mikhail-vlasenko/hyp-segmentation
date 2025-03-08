import time

import geoopt as gt
import torch
from scipy.linalg import pinvh

ball = gt.PoincareBall(c=1.)

reps = torch.randn(35000, 200, device="cuda", requires_grad=True)
prototypes = torch.randn(201, 200, device="cuda", requires_grad=True)

start = time.time()
reps = ball.expmap0(reps)
print(time.time() - start)

prototypes = ball.expmap0(prototypes)

distances = []
start = time.time()
for i in range(prototypes.shape[0]):
    distances.append(ball.dist(reps, prototypes[i].unsqueeze(0)))
distances = torch.stack(distances).T
print(time.time() - start)
print(distances.shape)
print(distances.device)



