# Learning-rate stability for gradient descent

## 1. Quadratic example from Day 05

Consider the linear model

\[
\hat{y}=XW^{T}+b,
\]

with mean-squared-error loss

\[
L(\theta)=\frac{1}{N}\lVert A\theta-y\rVert_2^2,
\tag{1}
\]

where

\[
\theta=\begin{bmatrix}w\\b\end{bmatrix}.
\tag{2}
\]

For the Day 05 data,

\[
A=
\begin{bmatrix}
0 & 1\\
1 & 1\\
2 & 1\\
3 & 1
\end{bmatrix},
\qquad N=4.
\tag{3}
\]

The gradient is

\[
\nabla L(\theta)=\frac{2}{N}A^T(A\theta-y),
\tag{4}
\]

and therefore the Hessian is the constant matrix

\[
H=\nabla^2L(\theta)=\frac{2}{N}A^TA.
\tag{5}
\]

For this example,

\[
H=
\begin{bmatrix}
7 & 3\\
3 & 2
\end{bmatrix}.
\tag{6}
\]

Gradient descent updates the parameters as

\[
\theta_{k+1}=\theta_k-\alpha\nabla L(\theta_k),
\tag{7}
\]

where \(\alpha\) is the learning rate.

Let \(\theta^*\) be the minimizer and define the error

\[
e_k=\theta_k-\theta^*.
\tag{8}
\]

For a quadratic loss,

\[
e_{k+1}=(I-\alpha H)e_k.
\tag{9}
\]

After diagonalizing \(H\), each eigen-direction evolves independently as

\[
e_{k+1}^{(i)}=(1-\alpha\lambda_i)e_k^{(i)}.
\tag{10}
\]

Convergence therefore requires

\[
\left|1-\alpha\lambda_i\right|<1
\quad \text{for every } i.
\tag{11}
\]

For a positive-definite Hessian this gives

\[
0<\alpha<\frac{2}{\lambda_{\max}}.
\tag{12}
\]

For the Day 05 problem,

\[
\lambda_{\max}\approx 8.405,
\tag{13}
\]

so the stability limit is approximately

\[
\alpha_{\mathrm{crit}}\approx \frac{2}{8.405}\approx 0.238.
\tag{14}
\]

This agrees with the numerical experiment: \(\alpha=0.23\) converges, while \(\alpha=0.25\) diverges.

For a positive-definite quadratic, the fixed learning rate that minimizes the worst-case asymptotic contraction factor is

\[
\alpha_{\mathrm{opt}}=
\frac{2}{\lambda_{\max}+\lambda_{\min}}.
\tag{15}
\]

This result is exact only for the quadratic, deterministic, full-batch case.

---

## 2. What changes for a general neural network?

For a nonlinear model with parameters \(\theta\in\mathbb{R}^{P}\), the loss is

\[
L=L(\theta),
\tag{16}
\]

and the Hessian is

\[
H(\theta)=\nabla_{\theta}^{2}L(\theta).
\tag{17}
\]

If \(P\) is in the millions or billions, the full Hessian would contain \(P^2\) entries, so explicitly forming it is impossible in practice.

The local quadratic approximation is

\[
L(\theta+\Delta\theta)
\approx
L(\theta)
+
\nabla L(\theta)^T\Delta\theta
+
\frac{1}{2}\Delta\theta^T H(\theta)\Delta\theta.
\tag{18}
\]

If the loss is locally smooth and the Hessian is positive in the relevant directions, the same intuition remains useful:

\[
\alpha\lesssim\frac{2}{\lambda_{\max}(H)}.
\tag{19}
\]

However, unlike the Day 05 quadratic example:

- \(H(\theta)\) changes during training;
- neural-network losses are generally nonconvex, so \(H\) may have negative eigenvalues;
- mini-batch gradients are noisy;
- adaptive optimizers rescale different parameter directions differently;
- the useful learning rate may therefore vary substantially over training.

Thus Eq. (19) is best viewed as a local stability guide, not as a globally exact learning-rate formula.

---

## 3. How can the largest curvature be estimated without forming the Hessian?

The key observation is that many iterative eigenvalue algorithms do not require the matrix itself. They only require matrix-vector products.

For the Hessian, this means computing

\[
v\mapsto Hv.
\tag{20}
\]

Automatic differentiation can compute a Hessian-vector product (HVP) without forming \(H\). One identity is

\[
Hv
=
\nabla_{\theta}
\left(
\nabla_{\theta}L(\theta)^Tv
\right).
\tag{21}
\]

Once HVPs are available, the dominant eigenvalue can be estimated by power iteration:

\[
v_{k+1}
=
\frac{Hv_k}{\lVert Hv_k\rVert_2},
\tag{22}
\]

with Rayleigh-quotient estimate

\[
\lambda_{\max}
\approx
\frac{v_k^T Hv_k}{v_k^Tv_k}.
\tag{23}
\]

Lanczos iteration can estimate the extreme eigenvalues more efficiently and accurately using the same HVP interface.

Therefore, even for a billion-parameter model, one does not need to store a billion-by-billion Hessian. One only needs repeated HVPs, whose memory cost can be comparable to a small number of gradient evaluations, although the computation is still expensive.

---

## 4. Why is this not normally used to choose every training learning rate?

In large-scale training, repeatedly estimating \(\lambda_{\max}(H)\) would add substantial cost. More importantly, the local curvature changes during optimization and the mini-batch gradient is stochastic.

Practical training therefore usually combines empirical and adaptive mechanisms such as

\[
\text{learning-rate warmup},
\quad
\text{decay schedules},
\quad
\text{momentum},
\quad
\text{Adam/AdamW},
\quad
\text{gradient clipping}.
\tag{24}
\]

A learning-rate range test is another practical approach: increase \(\alpha\) over a short trial run, observe where the loss begins to decrease rapidly and where it becomes unstable, and choose a value below the instability region.

Hence there are two complementary viewpoints:

1. **Curvature viewpoint:** estimate the local spectral scale through HVPs, power iteration, or Lanczos.
2. **Practical training viewpoint:** select a stable base learning rate experimentally, then control it using the optimizer and a schedule.

The simple Day 05 example is useful because it exposes the exact mathematical origin of the learning-rate stability limit before these complications are introduced.
