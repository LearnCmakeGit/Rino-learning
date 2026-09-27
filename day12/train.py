from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import QueryDecoderAutoEncoder


def make_dataset(num_samples=512, num_input_points=16, num_query_points=32):
    x_input = torch.linspace(0.0, 1.0, num_input_points)
    x_query = torch.linspace(0.0, 1.0, num_query_points).view(num_query_points, 1)

    input_samples = []
    query_coordinates = []
    query_values = []

    for i in range(num_samples):
        frequency = 0.5 + 2.5 * (i / num_samples)
        phase = 0.8 * torch.sin(torch.tensor(float(i)))

        u_input = torch.sin(2.0 * torch.pi * frequency * x_input + phase)
        u_query = torch.sin(2.0 * torch.pi * frequency * x_query[:, 0] + phase)

        input_samples.append(u_input)
        query_coordinates.append(x_query)
        query_values.append(u_query.view(num_query_points, 1))

    U_input = torch.stack(input_samples)
    X_query = torch.stack(query_coordinates)
    U_query = torch.stack(query_values)

    return U_input, X_query, U_query


def main():
    torch.manual_seed(0)

    U_input, X_query, U_query = make_dataset()

    dataset = TensorDataset(U_input, X_query, U_query)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = QueryDecoderAutoEncoder(
        num_input_points=16,
        latent_dim=4,
        hidden_dim=32,
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    model.train()

    num_epochs = 800

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for u_input, x_query, u_query in loader:
            u_pred, z = model(u_input, x_query)
            loss = loss_fn(u_pred, u_query)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"epoch={epoch:4d}  loss={epoch_loss:.6f}")

    model.eval()

    with torch.no_grad():
        sample_u = U_input[0:1]
        sample_xq = X_query[0:1]
        sample_true = U_query[0:1]

        sample_pred, z = model(sample_u, sample_xq)

    print("\nShapes:")
    print("sample_u shape      =", tuple(sample_u.shape))
    print("latent z shape      =", tuple(z.shape))
    print("sample_xq shape     =", tuple(sample_xq.shape))
    print("sample_pred shape   =", tuple(sample_pred.shape))

    print("\nFirst five query points:")
    for i in range(5):
        xq = sample_xq[0, i, 0].item()
        yt = sample_true[0, i, 0].item()
        yp = sample_pred[0, i, 0].item()
        print(f"x={xq:.3f}  true={yt:+.5f}  pred={yp:+.5f}")

    print("\nlatent z =", z.tolist())

    checkpoint = Path(__file__).resolve().parent / "query_decoder.pt"
    torch.save(model.state_dict(), checkpoint)
    print("\nSaved checkpoint:", checkpoint)


if __name__ == "__main__":
    main()
