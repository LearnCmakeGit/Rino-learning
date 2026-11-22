# Course Handoff — Python / NumPy / PyTorch Foundations

**Updated:** 2026-11-21  
**Repository:** `LearnCmakeGit/Rino-learning`  
**Status:** Day01–Day13 lessons/practice exist; **Day14 final assessment has not yet been completed**. This is a living handoff note, not a completion certificate.

## Purpose and working agreement

This repository is the record of a roughly **two-week foundations course**. The immediate goal is practical fluency in Python, NumPy, PyTorch, model training/inference, and debugging—not to force every exercise toward the RINO research project. The learner already has strong C++/numerical PDE/HPC knowledge but is building Python/PyTorch idioms from first principles.

Preferred learning style:
- The learner writes their own `*_prac.py` implementations; review the actual GitHub version and explain errors and underlying concepts before changing files.
- Make tensor shapes, broadcasting, gradient tracking, train/eval mode, and numerical quantities explicit.
- Avoid overextending the two-week foundations course. Preserve substantive math and experimental observations in notes when useful.
- Do not silently overwrite learner practice code. Keep the separate Neural Operator learning track and original RINO research discussion distinct.

## Course trajectory recorded so far

| Segment | Focus | Evidence / notes |
|---|---|---|
| Day01–Day03 | Python basics, NumPy matrix operations, PyTorch tensors | `day01/`–`day03/` |
| Day04–Day06 | Autograd, manual gradient descent, `nn.Module`, optimizers | `day04/`–`day06/` |
| Day07–Day09 | Dataset/DataLoader, train-vs-infer, checkpoints and workflow | `day07/`–`day09/` |
| Day10–Day11 | MLP, nonlinear function approximation | `day10/`–`day11/` |
| Day12 | Autoencoder / latent representation and reconstruction | `day12/model_prac.py`, `day12/train_prac.py` |
| Day13 | Function encoder + continuous query decoder; varying `hidden_dim` and `latent_dim`; inference diagnostics | `day13/model_prac.py`, `day13/train_prac.py`, `day13/infer_prac.py`, `day13/latent_dimension_notes.md` |
| **Day14** | **Independent final integration exercise** | **Not started / not yet assessed** |

These are topic-level progress markers, **not** a claim that every skill has been independently tested.

## Day13: verified implementation and experiments

Practice model in `day13/model_prac.py`:
- Default encoder `[B,16] -> [B,3]` (`hidden_dim=64`, `latent_dim=3`).
- Query decoder takes repeated latent and coordinates `[B,Q,1]`, outputs `[B,Q,1]`.
- `z.unsqueeze(1).expand(-1,Q,-1)` then `torch.cat(...,dim=-1)` handles arbitrary query count **but not arbitrary number/location of input sensors**.

Practice training in `day13/train_prac.py`: random family `u(x)=sin(2π f x + φ)`, `f∈[0.5,3]`, `φ∈[-π,π]`; 512 training functions, 16 fixed sensors, 24 random queries per function; `torch.manual_seed(0)`, Adam learning rate `1e-3`, 800 epochs. Training loss normalized by sample count.

Single-run final **training** MSE at epoch 799:
- `hidden_dim=8, latent_dim=4`: **0.240467**
- `hidden_dim=64, latent_dim=2`: **0.062785**
- `hidden_dim=64, latent_dim=3`: **0.002244**
- `hidden_dim=64, latent_dim=4`: **0.004023**

Inference test in `day13/infer_prac.py`: one sine `f=1.5`, `φ=3`, 41 new query positions. Reported **MSE 0.0007257518**, **RMSE 0.02694**, **max absolute error 0.05613**. This is a useful new-function example, **not a statistically comprehensive held-out test**. The practice file correctly constructs `model = model_prac()`, loads `day13.pt`, calls `model.eval()`, and invokes `model(input_u,query_x)` under `torch.no_grad()`.

Detailed numeric results, convergence graphs, percentages, and the phase-topology interpretation are retained in **[day13/latent_dimension_notes.md](day13/latent_dimension_notes.md)**. Key caution: 2 underlying parameters (`f,φ`) do not imply that a 2-coordinate continuous Euclidean latent is easiest; `(f,cosφ,sinφ)` is an illustrative 3D representation, **not proof of actual learned latent geometry** or minimal dimension. Single seeds and training loss cannot settle generalization or architecture superiority.

Relevant prior note: **[notes/learning_rate_stability.md](notes/learning_rate_stability.md)**.

## Current understanding versus remaining checks

Demonstrated through written code and discussion:
- Tensor shape manipulation, `unsqueeze`, `expand`, concatenation, and guarding against silent broadcasting.
- Model class versus instance (`model_prac` vs `model`), `forward()`, checkpoint loading, `eval()`, and `no_grad()`.
- Error interpretation: MSE vs RMSE vs maximum absolute error, aggregate metrics vs isolated pointwise discrepancies.
- Capacity (`hidden_dim`) vs bottleneck (`latent_dim`); fixed-seed reproducibility and limits of single runs.

Still worth assessing independently:
- Writing an unfamiliar training/inference pipeline **without copying** a prior script.
- Explicit train/validation/test distinction, data leakage checks, performance on unseen samples, and overfitting.
- Correct tensor shapes and broadcasting across batch/point axes.
- Saving/reloading and checking that predictions match after reload.
- Diagnose a deliberately seeded bug, with explanation instead of trial-and-error.

## Day14 — proposed final integration assessment (pending)

Complete a small **new** supervised PyTorch learning exercise, not just another sine query decoder. Required: generate separate train/test datasets; define an `nn.Module`; create a `DataLoader`; train with Adam; switch to `eval()`/`no_grad()`; save/reload weights; report MSE/RMSE/max error on train and unseen test data; write a short postmortem on failure modes and tensor shapes.

Do not mark the course complete until the learner attempts the task and its results are reviewed. Add assessment outcomes and remaining weak spots to this file **after** Day14.

## Next conversation: Neural Operator foundations (separate track)

**Start a fresh conversation** once Day14 and this handoff have been finalized. Have the assistant read this file and the actual GitHub sources, especially `day13/`, instead of relying on long-chat memory. A separate short course can then cover (in an adjustable order): function-to-function mappings; simple solution operators; DeepONet and FNO concepts and minimal implementation; resolution and sampling; then cross-resolution diagnostics.

**Keep this separate from RINO paper research.** A query-decoder function autoencoder is a useful building block but is not itself evidence of a learned nontrivial PDE solution operator. Variable-resolution input and latent consistency are future, optional topics, not prerequisites to closing the two-week foundation course.

### Suggested opening message for the next chat

> 我开始 Neural Operator 基础课程。先读 GitHub 仓库 `LearnCmakeGit/Rino-learning` 的 `COURSE_HANDOFF.md`、Day13 练习代码以及后续更新的 Day14 验收记录。按短课程模式逐步学习，重点补基础并亲手写练习代码，不要默认把内容绑定到 RINO 论文研究。先根据交接记录提出课程路线。
