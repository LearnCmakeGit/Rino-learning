# Day13: Query at arbitrary coordinates

Day12 used a fixed query grid. Day13 uses randomly drawn query coordinates during training and predicts at new coordinates during inference.

- Input: 16 samples on fixed sensor points, shape `[batch, 16]`
- Latent representation: `z`, shape `[batch, 8]`
- Query points: shape `[batch, num_queries, 1]`
- Output and target: `[batch, num_queries, 1]`

Run from repo root:

```bash
python day13/train.py
python day13/infer.py
```

## Practice

1. In `infer.py`, change 41 query points to 101. Which tensor shapes change? Does the model require retraining?
2. Replace `1.25` with `2.5`; compare inference MSE. The same network was trained across multiple functions.
3. Write `train_prac.py` and `model_prac.py` yourself. Add `assert prediction.shape == target.shape` before loss.
4. Compare `model(u_input, query)` with calling `model.encode(u_input)` once and `model.decode(z, query)` for multiple query grids.

The query coordinates can change without changing the input sensor grid, but the encoder still assumes **16 fixed input sensor values**. Supporting arbitrary input discretizations is a later step.
