"""Day13: Train with random query coordinates, test on a denser grid."""
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import QueryFunctionModel


def evaluate_function(frequency, phase, coordinates):
    # frequency, phase: [samples, 1]; coordinates: [samples, queries]
    return torch.sin(2.0 * torch.pi * frequency * coordinates + phase)


def make_data(num_samples=512, num_input_points=16, num_queries=24):
    # Input sensor positions stay fixed, but query positions vary by sample.
    x_input = torch.linspace(0.0, 1.0, num_input_points)
    frequency = 0.5 + 2.5 * torch.rand(num_samples, 1)
    phase = -torch.pi + 2.0 * torch.pi * torch.rand(num_samples, 1)

    u_input = evaluate_function(frequency, phase, x_input.unsqueeze(0))
    x_query = torch.rand(num_samples, num_queries, 1)
    u_target = evaluate_function(frequency, phase, x_query[..., 0]).unsqueeze(-1)

    return u_input, x_query, u_target


def main():
    torch.manual_seed(0)
    u_input, x_query, u_target = make_data()
    loader = DataLoader(TensorDataset(u_input, x_query, u_target),
                        batch_size=32, shuffle=True)

    model = QueryFunctionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(300):
        model.train()
        total_loss = 0.0
        for batch_input, batch_query, batch_target in loader:
            prediction, z = model(batch_input, batch_query)
            assert prediction.shape == batch_target.shape, (
                prediction.shape, batch_target.shape
            )
            loss = loss_fn(prediction, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_input.shape[0]

        if epoch % 50 == 0 or epoch == 299:
            print(f"epoch={epoch:3d} mean_mse={total_loss / len(loader.dataset):.6f}")

    model.eval()
    with torch.no_grad():
        sample = u_input[0:1]
        # These 101 coordinates differ from training-time random query points.
        new_x = torch.linspace(0.0, 1.0, 101).view(1, 101, 1)
        prediction, z = model(sample, new_x)

        # At a training sensor position, compare with the corresponding input.
        sensor_query = torch.linspace(0.0, 1.0, 16).view(1, 16, 1)
        sensor_prediction, _ = model(sample, sensor_query)
        sensor_mse = loss_fn(sensor_prediction[..., 0], sample).item()

    print("input:", tuple(sample.shape))
    print("z:", tuple(z.shape))
    print("new queries:", tuple(new_x.shape))
    print("prediction:", tuple(prediction.shape))
    print(f"sensor reconstruction MSE: {sensor_mse:.6f}")
    print("first five (x, prediction):")
    for x, y in zip(new_x[0, :5, 0], prediction[0, :5, 0]):
        print(f"  {x.item():.3f}  {y.item():+.5f}")

    path = Path(__file__).resolve().parent / "query_model.pt"
    torch.save(model.state_dict(), path)
    print("checkpoint:", path)


if __name__ == "__main__":
    main()
