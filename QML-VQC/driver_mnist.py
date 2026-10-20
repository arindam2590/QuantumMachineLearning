"""Quantum ML Project - MNIST classification using a hybrid ANN + VQC model.

This file is a separate user/driver program. The existing VQC_framework and
config_quantum_circuit.yaml are reused without modification.

Architecture
------------
MNIST image (1 x 28 x 28)
    -> Flatten
    -> Linear(784, 128) + ReLU
    -> Linear(128, 32) + ReLU
    -> Linear(32, n_qubits) + Tanh
    -> scale to quantum angles [-pi, pi]
    -> VQC(n_qubits -> n_qubits expectation values)
    -> Linear(n_qubits, 32) + ReLU
    -> Linear(32, 10)
    -> digit logits (0..9)

The VQC itself is still defined entirely by config_quantum_circuit.yaml and
uses PennyLane default.qubit through VQC_framework.

Important device note
---------------------
PennyLane default.qubit runs the quantum simulation on CPU. On Apple Silicon,
repeated MPS <-> CPU transfers during hybrid ANN + VQC training may cause
Metal command-buffer crashes. Therefore, if "mps" is requested, this driver
uses CPU for the complete hybrid model.
"""

import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import yaml

from VQC_framework import VQC
from hybrid_mnist_vqc import HybridMNISTVQC


DEFAULT_CONFIG = "config_quantum_circuit.yaml"


def parse_args():
    """Read the primary QML training arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=str, default="./data")

    # Quantum simulation is slower than an ordinary ANN. Small defaults make
    # the example practical. Use 0 to use the complete MNIST split.
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=500)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-output",
        type=str,
        default="model/mnist_vqc_model.pt",
    )
    parser.add_argument(
        "--circuit-output",
        type=str,
        default="circuit/mnist_vqc_circuit.png",
    )

    return parser.parse_args()


def resolve_training_device(requested):
    """Resolve a stable device for hybrid ANN + VQC training.

    default.qubit is CPU based. For Apple MPS, the repeated MPS <-> CPU
    transfers through the differentiable VQC can cause Metal runtime crashes.
    Therefore MPS falls back to CPU in this driver.
    """
    requested = requested.lower()

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"

        print(
            "Warning: CUDA requested but is not available. "
            "Falling back to CPU."
        )
        return "cpu"

    if requested == "mps":
        print(
            "Warning: MPS requested, but PennyLane default.qubit is CPU based.\n"
            "Using CPU for the complete hybrid ANN + VQC model to avoid "
            "Metal/MPS runtime crashes."
        )
        return "cpu"

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            print(
                "Info: Apple MPS is available, but this example uses CPU "
                "because the VQC uses PennyLane default.qubit."
            )
            return "cpu"

        return "cpu"

    raise ValueError(
        "Unsupported device. Choose from: auto, cpu, cuda, mps."
    )


def load_quantum_circuit_config(path):
    """Load only the quantum-circuit section from the existing YAML file."""
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "quantum_circuit" not in config:
        raise KeyError(
            "The configuration file must contain a top-level "
            "'quantum_circuit:' section."
        )

    return config["quantum_circuit"]


def set_seed(seed):
    """Set simple reproducibility seeds for this example."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_dataset(dataset, max_samples, seed):
    """Optionally use a deterministic subset; 0 means the full dataset."""
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator
    )[:max_samples]

    return Subset(dataset, indices.tolist())


def create_data_loaders(args):
    """Download MNIST and create train/test DataLoaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset = limit_dataset(
        train_dataset,
        args.train_samples,
        args.seed,
    )

    test_dataset = limit_dataset(
        test_dataset,
        args.test_samples,
        args.seed + 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


def train(model, loader, optimizer, criterion, device):
    """Train the hybrid network for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return (
        total_loss / total,
        100.0 * correct / total,
    )


def evaluate(model, loader, criterion, device):
    """Evaluate the classifier."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return (
        total_loss / total,
        100.0 * correct / total,
    )


def show_sample_predictions(model, loader, device, count=10):
    """Print a few true/predicted digit labels."""
    model.eval()

    images, labels = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu()

    labels = labels.cpu()
    n = min(count, labels.numel())

    print("\nSample predictions")
    print("------------------")

    for index in range(n):
        print(
            f"Sample {index + 1:2d}: "
            f"true={labels[index].item()}  "
            f"predicted={predictions[index].item()}"
        )


def save_model(model, args, circuit_config, resolved_device):
    """Save the trained hybrid model and basic experiment information."""
    output_path = Path(args.model_output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "quantum_circuit_config": circuit_config,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "requested_device": args.device,
            "resolved_device": str(resolved_device),
        },
        output_path,
    )

    print(f"\nModel saved to: {output_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    circuit_config = load_quantum_circuit_config(
        args.config
    )

    training_device = resolve_training_device(
        args.device
    )

    print(f"Epochs          : {args.epochs}")
    print(f"Learning rate   : {args.lr}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Device request  : {args.device}")
    print(f"Resolved device : {training_device}")
    print(f"Train samples   : {args.train_samples or 'all'}")
    print(f"Test samples    : {args.test_samples or 'all'}")

    # VQC and ANN use the same stable training device.
    vqc = VQC(
        circuit_config,
        device=training_device
    )

    device = vqc.device

    print("\n" + vqc.summary())

    train_loader, test_loader = create_data_loaders(
        args
    )

    model = HybridMNISTVQC(vqc).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    print("\nHybrid network")
    print("--------------")
    print(model)

    print("\nTraining")
    print("--------")

    for epoch in range(
        1,
        args.epochs + 1
    ):
        train_loss, train_acc = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f} | "
            f"train acc={train_acc:6.2f}% | "
            f"test loss={test_loss:.4f} | "
            f"test acc={test_acc:6.2f}%"
        )

    show_sample_predictions(
        model,
        test_loader,
        device
    )

    save_model(
        model,
        args,
        circuit_config,
        device
    )

    # Visualize the quantum circuit using one encoded MNIST image.
    images, _ = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        encoded = model.image_encoder(
            images[:1]
        )

        quantum_input = (
            torch.pi * encoded
        ).squeeze(0)

    circuit_path = Path(
        args.circuit_output
    )

    circuit_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vqc.visualize(
        quantum_input,
        save_path=str(circuit_path),
        show=False,
    )

    print(
        f"Circuit visualization saved to: "
        f"{circuit_path}"
    )


if __name__ == "__main__":
    main()
