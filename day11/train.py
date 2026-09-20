from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import AutoEncoder1D


def make_dataset(num_samples=256, num_points=16):
    x = torch.linspace(0.0, 1.0, num_points)

    samples = []
    for i in range(num_samples):
        a = 0.5 + 2.5 * (i / (num_samples - 1))
        phase = 0.5 * torch.sin(torch.tensor(float(i)))
        u = torch.sin(2.0 * torch.pi * a * x + phase)
        samples.append(u)

    X = torch.stack(samples)
    return X


def main():
    torch.manual_seed(0)

    X = make_dataset()
    dataset = TensorDataset(X, X)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AutoEncoder1D(input_dim=16, latent_dim=3)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()

    for epoch in range(500):
        epoch_loss = 0.0

        for X_batch, Y_batch in loader:
            X_recon = model(X_batch)
            loss = loss_fn(X_recon, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 100 == 0 or epoch == 499:
            print(f"epoch={epoch:3d} loss={epoch_loss:.6f}")

    print("\nModel parameter shapes:")
    for name, param in model.named_parameters():
        print(f"{name:25s} {tuple(param.shape)}")

    model.eval()
    with torch.no_grad():
        sample = X[0:1]
        z = model.encode(sample)
        recon = model.decode(z)

    print("\nOne sample:")
    print("input shape =", tuple(sample.shape))
    print("latent z shape =", tuple(z.shape))
    print("reconstructed shape =", tuple(recon.shape))
    print("latent z =", z)
    print("sample reconstruction MSE =", torch.mean((recon - sample) ** 2).item())

    checkpoint = Path(__file__).resolve().parent / "autoencoder.pt"
    torch.save(model.state_dict(), checkpoint)
    print("\nSaved checkpoint:", checkpoint)


if __name__ == "__main__":
    main()
