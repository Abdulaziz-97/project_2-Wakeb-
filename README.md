Windowed absolute positional embeddings are **absolute position embeddings (APEs)** used in a model that does *window-based attention*, where the position encoding is defined per location **inside a window** and then reused (tiled) across all windows. [arxiv](http://arxiv.org/abs/2311.05613)

## First: what “absolute positional embeddings” are
An absolute positional embedding is a learned vector (or table) tied to each absolute token/patch location and **added to the token embeddings** so attention can use spatial/order information. [arxiv](http://arxiv.org/abs/2311.05613)
This is the standard ViT-style “abs pos embed” idea: attention itself has no spatial bias, so the embedding injects that bias. [arxiv](http://arxiv.org/abs/2311.05613)

## What “windowed” changes
With window attention, tokens attend only within a local \(W \times W\) window, but the attention weights (QKV projections) are shared across windows. [arxiv](http://arxiv.org/abs/2311.05613)
So a common design is to make the absolute embedding *window-aware*: a \(W \times W\) position table that represents “top-left of a window, center of a window, …” and is reused for every window location in the image (tiling). [arxiv](http://arxiv.org/abs/2311.05613)

## Why people do this (the bug it fixes)
A key issue reported in “Window Attention is Bugged” is that **naively interpolating** global absolute position embeddings when you switch to / use window attention at different resolutions can be wrong and hurt performance. [arxiv](http://arxiv.org/abs/2311.05613)
Their fix is essentially: ensure each window uses the same position embedding pattern by **tiling the pretrained embeddings per window** (their “absolute window position embedding / absolute win” strategy). [arxiv](http://arxiv.org/abs/2311.05613)

## Relation to Swin/implementations
In Swin-like models, you may see a config switch like `use_absolute_embeddings` (optional), because many windowed architectures instead rely on relative position bias; but absolute embeddings can still be used. [huggingface](https://huggingface.co/docs/transformers/en/model_doc/swin)
When absolute embeddings are used with windows, the “windowed absolute” interpretation is: positions are encoded consistently within each window, rather than as one big global grid that you later interpolate. [arxiv](http://arxiv.org/abs/2311.05613)

If you tell me which context you saw it in (Swin? ViTDet? Hiera? a specific repo), I can map the exact tensor shapes (e.g., \((1, W^2, d)\) tiled vs \((1, H\!\times\!W, d)\) interpolated).
