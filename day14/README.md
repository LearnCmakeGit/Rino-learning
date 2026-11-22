# Day14 — Final integration assessment (independent practice)

**Purpose:** Finish the two-week **Python / NumPy / PyTorch foundations** course. This is an assessment of training, testing, shape discipline, saving/reloading, and debugging—not a Neural Operator or RINO assignment.

**Status:** Assigned; not yet completed or graded.

## New task: Learn a nonlinear scalar map

Construct a PyTorch regression model for

```python
def target_fn(x):
    return torch.sin(2.0 * x) + 0.3 * x**2
```

with input `x` of shape `[B,1]` and target `y` of shape `[B,1]`. This is intentionally **different from Day13**: there is no latent variable or query decoder, and you should build the pipeline yourself.

### Dataset specification

- Set `torch.manual_seed(0)` for reproducibility.
- Training: 512 independently sampled inputs, `x_train = -2 + 4 * torch.rand(512,1)`.
- Validation: 128 independently sampled inputs in the same interval.
- Test: 256 independently sampled inputs in the same interval, generated separately. Do **not** train or tune on the test set.
- Compute targets using `target_fn`; do not add noise at first.
- Ensure all tensors use shape `[N,1]`, not `[N]`, to avoid silent broadcasting.

### Implementation (write this yourself)

Create three practice files:

1. `day14/model_prac.py`: define an `nn.Module` with at least two `nn.Linear` layers and a nonlinear activation. It must map `[B,1] → [B,1]`. You choose the width and depth; explain the choice briefly.
2. `day14/train_prac.py`: generate data, build `TensorDataset` and shuffled `DataLoader` for training, optimize with Adam and MSE, measure **train and validation MSE per epoch**. Make a reasonable stopping rule (fixed epochs or validation-based early stopping). Save `model.state_dict()` under `day14/`. Print enough metrics to see whether the model overfits or underfits.
3. `day14/infer_prac.py`: reconstruct the same model class, load saved weights using `map_location="cpu", weights_only=True`, switch to `eval()`, infer under `torch.no_grad()`, and evaluate on a separately generated test set.

**Important reproducibility detail:** Independently producing the same test set from a script requires a dedicated generator/seed or a shared data function; avoid depending accidentally on how many random numbers training consumed. A simple choice is `test_generator = torch.Generator().manual_seed(2026)` and `torch.rand(256,1,generator=test_generator)`. Keep the seed and definition of the held-out test set explicit.

### Required output

- Report the training, validation, and test **MSE**; report test **RMSE** and **max absolute error**.
- Assert `prediction.shape == target.shape` **before computing loss**.
- Print five pairs of `(x, prediction, truth)` from the test set.
- Reload the weights into a new model and verify its predictions match the original model on the same inputs (e.g., `torch.allclose`); if testing in separate scripts, document how you compared them.
- In a brief note, explain `train()` vs `eval()`, `zero_grad/backward/step`, and the exact input/output tensor shapes.

### Optional: one debugging exercise

Deliberately change your target from `[B,1]` to `[B]`, observe how broadcasting affects the loss shape, then restore the correct target. Explain why shape assertion is necessary. This is optional and should not be used in final training results.

## Acceptance checklist

- [ ] Model implemented independently in `model_prac.py`
- [ ] Distinct train/validation/test datasets, no test leakage
- [ ] Correct minibatch training loop and MSE metrics
- [ ] Shape checks and correct evaluation mode / no-grad
- [ ] Checkpoint save + reload and matching predictions
- [ ] Test MSE, RMSE, max error, and five sample predictions
- [ ] Short explanation of numerical/model behavior
- [ ] GitHub code reviewed and handoff note updated **after** acceptance

## Suggested workflow

1. Build the three `*_prac.py` files independently; use the earlier Day06–Day13 scripts only for targeted API reminders rather than copying whole solutions.
2. Run locally or in Colab and commit the files.
3. Ask for a GitHub code review. We will verify correctness and interpret the results, then update [COURSE_HANDOFF.md](../COURSE_HANDOFF.md).
4. Once complete, start a fresh conversation for the separate Neural Operator course.

**No solutions are included intentionally.** The goal is to test what can be reconstructed without a template.
