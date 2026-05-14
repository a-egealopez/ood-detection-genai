from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, TensorDataset


@dataclass
class DataLoaderSpec:
    id_name: str
    ood_name: str
    input_dim: int
    is_image: bool


def _to_tensor_dataset(
    x: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> TensorDataset:
    x = (x - mean) / std
    return TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long(),
    )


def _build_mnist_dataset(cfg: DictConfig) -> tuple[dict, DataLoaderSpec]:
    n_train = cfg.data.get("n_train_samples", None)
    n_eval = cfg.data.get("n_eval_samples", None)
    root = cfg.data.root

    norm = transforms.Normalize((0.5,), (0.5,))
    flat = transforms.Lambda(lambda x: x.view(1, -1))

    id_tf = transforms.Compose([transforms.ToTensor(), norm, flat])
    ood_tf = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            norm,
            flat,
        ]
    )

    id_train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=id_tf)
    id_test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=id_tf)
    ood_test = torchvision.datasets.SVHN(root=root, split="test", download=True, transform=ood_tf)

    def first_n(ds, n):
        if n is None:
            return ds
        n = min(n, len(ds))
        return Subset(ds, list(range(n)))

    datasets = {
        "train": first_n(id_train, n_train),
        "id_eval": first_n(id_test, n_eval),
        "ood_eval": first_n(ood_test, n_eval),
    }

    return datasets, DataLoaderSpec(
        id_name="MNIST",
        ood_name="SVHN",
        input_dim=784,
        is_image=True,
    )


def _build_sicap_dataset(cfg: DictConfig) -> tuple[dict, DataLoaderSpec]:
    root = Path(cfg.data.features_dir)

    if not root.exists():
        root = Path("data/data_sicap/features_UNI")
    if not root.exists():
        raise FileNotFoundError(f"features_dir not found: {root}")

    x_train = np.load(root / "X_train.npy").astype(np.float32)
    y_train = np.load(root / "y_train.npy").squeeze().astype(np.int64)
    x_test = np.load(root / "X_test.npy").astype(np.float32)
    y_test = np.load(root / "y_test.npy").squeeze().astype(np.int64)

    train_classes = cfg.data.get("train_classes", [1, 2])
    test_classes = cfg.data.get("test_classes", [3, 4])

    train_mask = np.isin(y_train, train_classes)
    id_mask = np.isin(y_test, train_classes)
    ood_mask = np.isin(y_test, test_classes)

    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test_id, y_test_id = x_test[id_mask], y_test[id_mask]
    x_test_ood, y_test_ood = x_test[ood_mask], y_test[ood_mask]

    if len(x_train) == 0 or len(x_test_id) == 0 or len(x_test_ood) == 0:
        raise ValueError("Class split produced an empty dataset; check train/test classes.")

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    n_train = cfg.data.get("n_train_samples", None)
    n_eval = cfg.data.get("n_eval_samples", None)

    def first_n(x, y, n):
        if n is None:
            return _to_tensor_dataset(x, y, mean, std)
        n = min(n, len(x))
        return _to_tensor_dataset(x[:n], y[:n], mean, std)

    datasets = {
        "train": first_n(x_train, y_train, n_train),
        "id_eval": first_n(x_test_id, y_test_id, n_eval),
        "ood_eval": first_n(x_test_ood, y_test_ood, n_eval),
    }

    return datasets, DataLoaderSpec(
        id_name="sicap", ood_name="sicap_ood", input_dim=int(x_train.shape[1]), is_image=False
    )


_DATASET_BUILDERS = {
    "mnist": _build_mnist_dataset,
    "sicap": _build_sicap_dataset,
}


def build_dataloaders(cfg: DictConfig) -> dict:
    dataset_name = str(cfg.data.get("dataset", "mnist")).lower()

    if dataset_name not in _DATASET_BUILDERS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    datasets, meta = _DATASET_BUILDERS[dataset_name](cfg)

    cfg.data.input_dim = meta.input_dim
    cfg.data.id_name = meta.id_name
    cfg.data.ood_name = meta.ood_name
    cfg.data.is_image = meta.is_image

    loader_kwargs = {
        "batch_size": int(cfg.training.batch_size),
        "num_workers": int(cfg.data.get("num_workers", 0)),
        "pin_memory": bool(cfg.data.get("pin_memory", False)),
    }

    return {
        split: DataLoader(datasets[split], shuffle=(split == "train"), **loader_kwargs)
        for split in ["train", "id_eval", "ood_eval"]
    }
