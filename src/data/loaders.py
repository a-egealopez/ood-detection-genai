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
    img_mean: np.ndarray | None = None
    img_std: np.ndarray | None = None


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
    flat_mode = bool(cfg.data.get("flat", True))

    norm = transforms.Normalize((0.5,), (0.5,))
    flat_tf = transforms.Lambda(lambda x: x.view(-1))

    if flat_mode:
        id_tf = transforms.Compose([transforms.ToTensor(), norm, flat_tf])
        ood_tf = transforms.Compose(
            [transforms.Grayscale(1), transforms.Resize((28, 28)), transforms.ToTensor(), norm, flat_tf]
        )
    else:
        id_tf = transforms.Compose([transforms.ToTensor(), norm])
        ood_tf = transforms.Compose(
            [transforms.Grayscale(1), transforms.Resize((28, 28)), transforms.ToTensor(), norm]
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

    id_classes  = cfg.data.get("id_classes",  [1, 2])
    ood_classes = cfg.data.get("ood_classes", [3, 4])

    train_mask = np.isin(y_train, id_classes)
    id_mask    = np.isin(y_test,  id_classes)
    ood_mask   = np.isin(y_test,  ood_classes)

    x_train, y_train       = x_train[train_mask], y_train[train_mask]
    x_test_id, y_test_id   = x_test[id_mask],     y_test[id_mask]
    x_test_ood, y_test_ood = x_test[ood_mask],    y_test[ood_mask]

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


def _build_toy_dataset(cfg: DictConfig) -> tuple[dict, DataLoaderSpec]:
    from sklearn.datasets import make_moons

    dataset_name = str(cfg.data.dataset)
    n_samples = int(cfg.data.get("n_samples", 600))
    noise = float(cfg.data.get("noise", 0.05))
    seed = int(cfg.get("seed", 42))

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    X = X.astype(np.float32)
    X_id, y_id = X[y == 0], y[y == 0]
    X_ood, y_ood = X[y == 1], y[y == 1]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_id))
    X_id, y_id = X_id[idx], y_id[idx]

    n_train = int(len(X_id) * float(cfg.data.get("train_frac", 0.7)))
    X_train, y_train = X_id[:n_train], y_id[:n_train]
    X_id_eval, y_id_eval = X_id[n_train:], y_id[n_train:]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    datasets = {
        "train": _to_tensor_dataset(X_train, y_train, mean, std),
        "id_eval": _to_tensor_dataset(X_id_eval, y_id_eval, mean, std),
        "ood_eval": _to_tensor_dataset(X_ood, y_ood, mean, std),
    }
    return datasets, DataLoaderSpec(
        id_name=f"{dataset_name}_id",
        ood_name=f"{dataset_name}_ood",
        input_dim=2,
        is_image=False,
    )


def _build_blobs_dataset(cfg: DictConfig) -> tuple[dict, DataLoaderSpec]:
    from sklearn.datasets import make_blobs

    n_samples = int(cfg.data.get("n_samples", 2000))
    centers = cfg.data.get("centers", [[-2.0, 0.0], [2.0, 0.0]])
    cluster_std = float(cfg.data.get("cluster_std", 0.5))
    seed = int(cfg.get("seed", 42))

    X, y = make_blobs(
        n_samples=n_samples,
        centers=list(centers),
        cluster_std=cluster_std,
        random_state=seed,
    )

    X = X.astype(np.float32)
    X_id, y_id = X[y == 0], y[y == 0]
    X_ood, y_ood = X[y == 1], y[y == 1]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_id))
    X_id, y_id = X_id[idx], y_id[idx]

    n_train = int(len(X_id) * float(cfg.data.get("train_frac", 0.7)))
    X_train, y_train = X_id[:n_train], y_id[:n_train]
    X_id_eval, y_id_eval = X_id[n_train:], y_id[n_train:]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    datasets = {
        "train": _to_tensor_dataset(X_train, y_train, mean, std),
        "id_eval": _to_tensor_dataset(X_id_eval, y_id_eval, mean, std),
        "ood_eval": _to_tensor_dataset(X_ood, y_ood, mean, std),
    }
    return datasets, DataLoaderSpec(
        id_name="blobs_id",
        ood_name="blobs_ood",
        input_dim=2,
        is_image=False,
    )


def _build_pathmnist_dataset(cfg: DictConfig) -> tuple[dict, DataLoaderSpec]:
    try:
        from medmnist import PathMNIST
    except ImportError:
        raise ImportError("medmnist is required: pip install medmnist") from None

    root = str(cfg.data.get("root", "data/"))
    n_train = cfg.data.get("n_train_samples", None)
    n_eval = cfg.data.get("n_eval_samples", None)
    flat_mode = bool(cfg.data.get("flat", True))

    if cfg.data.get("id_classes") is not None:
        id_classes = list(cfg.data.id_classes)
        ood_classes = list(cfg.data.ood_classes)
    else:
        id_classes = [int(cfg.data.get("norm_label", 6))]
        ood_classes = [int(cfg.data.get("tum_label", 8))]

    size = int(cfg.data.get("img_size", 28))
    train_ds = PathMNIST(split="train", download=True, root=root, size=size)
    test_ds  = PathMNIST(split="test",  download=True, root=root, size=size)

    def to_chw(imgs: np.ndarray) -> np.ndarray:
        return imgs.astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0

    x_train_all = to_chw(train_ds.imgs)
    y_train_all = train_ds.labels.squeeze()
    x_test_all  = to_chw(test_ds.imgs)
    y_test_all  = test_ds.labels.squeeze()

    train_mask = np.isin(y_train_all, id_classes)
    id_mask    = np.isin(y_test_all,  id_classes)
    ood_mask   = np.isin(y_test_all,  ood_classes)

    x_train, y_train = x_train_all[train_mask], y_train_all[train_mask]
    x_id,    y_id    = x_test_all[id_mask],     y_test_all[id_mask]
    x_ood,   y_ood   = x_test_all[ood_mask],    y_test_all[ood_mask]

    if len(x_train) == 0 or len(x_id) == 0 or len(x_ood) == 0:
        raise ValueError("PathMNIST class filtering produced an empty split.")

    if flat_mode:
        x_train = x_train.reshape(len(x_train), -1)
        x_id    = x_id.reshape(len(x_id), -1)
        x_ood   = x_ood.reshape(len(x_ood), -1)

    _mean = np.zeros((1,) + x_train.shape[1:], dtype=np.float32)
    _std  = np.ones((1,) + x_train.shape[1:], dtype=np.float32)
    _denorm_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    _denorm_std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def first_n(x, y, n):
        if n is None:
            return x, y
        n = min(n, len(x))
        return x[:n], y[:n]

    x_train, y_train = first_n(x_train, y_train, n_train)
    x_id,    y_id    = first_n(x_id,    y_id,    n_eval)
    x_ood,   y_ood   = first_n(x_ood,   y_ood,   n_eval)

    id_tag  = "+".join(str(c) for c in id_classes)
    ood_tag = "+".join(str(c) for c in ood_classes)
    datasets = {
        "train":    _to_tensor_dataset(x_train, y_train, _mean, _std),
        "id_eval":  _to_tensor_dataset(x_id,    y_id,    _mean, _std),
        "ood_eval": _to_tensor_dataset(x_ood,   y_ood,   _mean, _std),
    }
    return datasets, DataLoaderSpec(
        id_name=f"PathMNIST-ID{id_tag}",
        ood_name=f"PathMNIST-OOD{ood_tag}",
        input_dim=3 * size * size,
        is_image=True,
        img_mean=_denorm_mean,
        img_std=_denorm_std,
    )


_DATASET_BUILDERS = {
    "mnist": _build_mnist_dataset,
    "sicap_c1": _build_sicap_dataset,
    "sicap_c12": _build_sicap_dataset,
    "moons": _build_toy_dataset,
    "blobs": _build_blobs_dataset,
    "pathmnist": _build_pathmnist_dataset,
    "pathmnist_c1": _build_pathmnist_dataset,
    "pathmnist_c2": _build_pathmnist_dataset,
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
    if meta.img_mean is not None:
        cfg.data.img_mean = meta.img_mean.tolist()
        cfg.data.img_std = meta.img_std.tolist()

    loader_kwargs = {
        "batch_size": int(cfg.training.batch_size),
        "num_workers": int(cfg.data.get("num_workers", 0)),
        "pin_memory": bool(cfg.data.get("pin_memory", False)),
    }

    return {
        split: DataLoader(datasets[split], shuffle=(split == "train"), **loader_kwargs)
        for split in ["train", "id_eval", "ood_eval"]
    }
