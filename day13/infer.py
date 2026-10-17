"""Day13: Reload the model and query ONE input function at new coordinates."""
from pathlib import Path

import torch
from model import QueryFunctionModel


def main():
    model = QueryFunctionModel()
    path = Path(__file__).resolve().parent / "query_model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()

    sensors = torch.linspace(0.0, 1.0, 16).view(1, 16)
    u_input = torch.sin(2 * torch.pi * 1.25 * sensors + 0.3)

    # 41 query positions: not the same count as the 16 input sensors.
    query = torch.linspace(0.0, 1.0, 41).view(1, 41, 1)
    truth = torch.sin(2 * torch.pi * 1.25 * query + 0.3)

    with torch.no_grad():
        prediction, z = model(u_input, query)
        mse = torch.mean((prediction - truth) ** 2).item()

    print("u_input shape:", tuple(u_input.shape))
    print("latent shape:", tuple(z.shape))
    print("query shape:", tuple(query.shape))
    print("output shape:", tuple(prediction.shape))
    print(f"inference MSE: {mse:.6f}")
    print("x      true      predicted")
    for idx in (0, 10, 20, 30, 40):
        print(f"{query[0, idx, 0]:.2f}  {truth[0, idx, 0]:+.5f}  {prediction[0, idx, 0]:+.5f}")


if __name__ == "__main__":
    main()
