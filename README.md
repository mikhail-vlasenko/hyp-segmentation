# Hyperbolic Part-Whole Image Segmentation

Official code for **"Hyperbolic Part-Whole Image Segmentation"**, accepted at AISTATS 2026.

## Overview

We introduce a hyperbolic prototypical segmentation framework that simultaneously segments images at **object**, **part**, and **subpart** granularity levels within a unified embedding space. By embedding class prototypes on the Poincare ball, the model captures the natural hierarchy between classes, enabling:

- **State-of-the-art** part and subpart segmentation on SubPartImageNet (SPIN), surpassing multi-billion-parameter models with only ~73M parameters
- **Zero-shot generalization** via hierarchy-aware prototypes
- **Cross-level transfer** from subpart-level supervision to object-level predictions without object-level labels

### Key Results on SPIN

| Model | Object mIoU | Part mIoU | Subpart mIoU | # Params |
|---|---|---|---|---|
| GLaMM-FT | 0.911 | 0.608 | 0.246 | 3x7B |
| HALLUMI | 0.893 | 0.582 | 0.185 | 7B |
| **Ours** | 0.895 | **0.677** | **0.268** | **73M** |


## Citation

```bibtex
@inproceedings{vlasenko2026hyperbolic,
  title={Hyperbolic Part-Whole Image Segmentation},
  author={Vlasenko, Mikhail and Ghadimi Atigh, Mina and Mettes, Pascal},
  booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026}
}
```
