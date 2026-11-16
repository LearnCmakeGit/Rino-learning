# Day13 note: latent dimension, width, and representation geometry

## Setup
Function family: `u(x)=sin(2π f x + φ)`, with `f ∈ [0.5,3]`, `φ ∈ [-π,π]`. Fixed 16 input sensors, 24 random query positions per function, 512 training samples. `torch.manual_seed(0)`, Adam (`lr=1e-3`), 800 epochs, MSE loss. The exercise uses an encoder–latent–query decoder with Tanh layers.

## Observed results (single run per configuration)

| hidden_dim | latent_dim | training MSE (epoch 799) | sample 0 sensor reconstruction MSE |
|---:|---:|---:|---:|
| 8 | 4 | 0.240467 | 0.410378 |
| 64 | 2 | 0.062785 | 0.056688 |
| 64 | 3 | **0.002244** | **0.003372** |
| 64 | 4 | 0.004023 | 0.010401 |

Before correcting the Adam learning rate from `0.1` to `0.001`, training fluctuated around a much higher error; an earlier printout also erroneously divided the accumulated sample-weighted loss by the number of batches rather than samples.

## Takeaways
1. At fixed `latent_dim=4`, increasing `hidden_dim` from 8 to 64 greatly improved fitting. Hidden-layer capacity and latent bottleneck are different factors.
2. The current function family varies through only **two parameters** (`f`, `φ`); the shared `sin` rule can be stored in decoder weights rather than one separate latent coordinate.
3. **Intrinsic dimension ≠ easiest Euclidean latent dimension.** Phase is periodic; an intuitive continuous representation is `z ≈ (f, cos φ, sin φ)`: three coordinates but still two independent degrees of freedom. This is a *hypothesis* about why latent 3 works well, not a measurement of the learned latent variables.
4. In these controlled runs, latent 3 fitted much better than latent 2, and even reached a lower training error than latent 4. This does **not** prove 3 is mathematically minimal or inherently superior to 4. Different latent sizes change initialization/optimization, and all numbers are single-run **training** errors, not unseen-function generalization tests.

## Possible follow-up
Evaluate fresh frequencies/phases and plot learned 3D `z` against `f`, `sin φ`, and `cos φ` to test the geometry conjecture.
