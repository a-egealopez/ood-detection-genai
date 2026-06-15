"""
Validación rápida: ¿escalar la resolución de PathMNIST invierte la asimetría
de complejidad entre tejido sano (ID) y tumoral (OOD)?

Métricas sin entrenar nada:
  - Entropía de Shannon (skimage)
  - Varianza local (complejidad de textura local)
  - Magnitud media del gradiente (detalle de bordes)

Resoluciones: 28, 64, 128, 224
Clases analizadas según ambos escenarios del proyecto:
  - c1: ID=[6], OOD=[7,8]
  - c2: ID=[0,3,4,5,6], OOD=[7,8]
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from medmnist import PathMNIST
from scipy import ndimage
from skimage.filters import sobel
from skimage.measure import shannon_entropy

# ── Configuración ──────────────────────────────────────────────────────────────
DATA_ROOT = "data/"
OUT_DIR = Path("figures/entropy_resolution")
RESOLUTIONS = [28, 64, 128, 224]
N_SAMPLES = 500  # muestras por clase (balance velocidad/precisión)
SEED = 42

CLASS_NAMES = {
    0: "Adipose",
    1: "Background",
    2: "Debris",
    3: "Lymphocytes",
    4: "Mucus",
    5: "Smooth muscle",
    6: "Normal mucosa",  # ID principal
    7: "Cancer stroma",  # OOD
    8: "Adenocarcinoma",  # OOD
}

ID_CLASSES = [0, 3, 4, 5, 6]
OOD_CLASSES = [7, 8]
ALL_CLASSES = ID_CLASSES + OOD_CLASSES

OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

# ── Métricas de complejidad ────────────────────────────────────────────────────


def img_to_gray(img_chw: np.ndarray) -> np.ndarray:
    """RGB [0,255] uint8 CHW → gray [0,1] float HW."""
    img = img_chw.astype(np.float32) / 255.0
    if img.ndim == 3:
        return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    return img[0]


def complexity_metrics(img_chw: np.ndarray) -> dict:
    gray = img_to_gray(img_chw)

    # 1. Entropía de Shannon (bins=256 sobre [0,1])
    entropy = shannon_entropy(gray)

    # 2. Varianza local media (ventana 5×5)
    local_mean = ndimage.uniform_filter(gray, size=5)
    local_sq = ndimage.uniform_filter(gray**2, size=5)
    local_var = np.mean(np.maximum(local_sq - local_mean**2, 0))

    # 3. Magnitud media del gradiente Sobel
    grad_mag = np.mean(sobel(gray))

    return {"entropy": entropy, "local_var": local_var, "grad_mag": grad_mag}


# ── Carga de datos y cómputo de métricas ──────────────────────────────────────


def load_class_images(ds, class_id: int, n: int) -> np.ndarray:
    labels = ds.labels.squeeze()
    idx = np.where(labels == class_id)[0]
    idx = rng.choice(idx, size=min(n, len(idx)), replace=False)
    # ds.imgs: (N, H, W, C) uint8
    return ds.imgs[idx].transpose(0, 3, 1, 2)  # → (n, C, H, W)


print("Cargando datos y calculando métricas...")
print(
    f"{'Res':>5}  {'Clase':<22}  {'Entropía':>10}  {'Var local':>10}  {'Grad mag':>10}  {'Tipo':>6}"
)
print("-" * 72)

# results[res][class_id] = {metric: array_of_values}
results = {}

for res in RESOLUTIONS:
    results[res] = {}
    ds = PathMNIST(split="train", size=res, download=True, root=DATA_ROOT)

    for cls in ALL_CLASSES:
        imgs = load_class_images(ds, cls, N_SAMPLES)
        metrics_list = [complexity_metrics(img) for img in imgs]

        results[res][cls] = {
            k: np.array([m[k] for m in metrics_list]) for k in ("entropy", "local_var", "grad_mag")
        }

        tag = "OOD" if cls in OOD_CLASSES else "ID"
        ent_m = results[res][cls]["entropy"].mean()
        var_m = results[res][cls]["local_var"].mean()
        grd_m = results[res][cls]["grad_mag"].mean()
        print(
            f"{res:>5}  {CLASS_NAMES[cls]:<22}  {ent_m:>10.4f}  {var_m:>10.6f}  {grd_m:>10.6f}  {tag:>6}"
        )

    print()


# ── Tabla resumen: ratio OOD/ID por resolución ────────────────────────────────

print("\n── RATIO COMPLEJIDAD  OOD / ID (>1 = OOD más complejo, bueno para detección) ──")
print(
    f"{'Res':>5}  {'Métrica':<12}  {'ID mean':>9}  {'OOD mean':>9}  {'Ratio':>7}  {'Dirección':>12}"
)
print("-" * 62)

ratio_table = {}  # ratio_table[res][metric] = ratio
for res in RESOLUTIONS:
    ratio_table[res] = {}
    for metric in ("entropy", "local_var", "grad_mag"):
        id_vals = np.concatenate([results[res][c][metric] for c in ID_CLASSES])
        ood_vals = np.concatenate([results[res][c][metric] for c in OOD_CLASSES])
        id_mean = id_vals.mean()
        ood_mean = ood_vals.mean()
        ratio = ood_mean / (id_mean + 1e-9)
        ratio_table[res][metric] = ratio
        direction = "OOD>ID ✓" if ratio > 1.0 else "ID>OOD ✗"
        print(
            f"{res:>5}  {metric:<12}  {id_mean:>9.5f}  {ood_mean:>9.5f}  {ratio:>7.3f}  {direction:>12}"
        )
    print()


# ── Figura 1: Evolución del ratio por resolución ──────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
metrics_labels = {
    "entropy": "Entropía de Shannon",
    "local_var": "Varianza local media",
    "grad_mag": "Magnitud gradiente Sobel",
}

for ax, metric in zip(axes, metrics_labels, strict=False):
    ratios = [ratio_table[r][metric] for r in RESOLUTIONS]
    colors = ["#e74c3c" if r < 1.0 else "#27ae60" for r in ratios]
    bars = ax.bar([str(r) for r in RESOLUTIONS], ratios, color=colors, edgecolor="k", linewidth=0.8)
    ax.axhline(
        1.0, color="black", linestyle="--", linewidth=1.2, label="ratio=1 (igual complejidad)"
    )
    ax.set_xlabel("Resolución (px)", fontsize=11)
    ax.set_ylabel("Ratio OOD/ID", fontsize=11)
    ax.set_title(metrics_labels[metric], fontsize=11, fontweight="bold")
    for bar, r in zip(bars, ratios, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{r:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(ratios) * 1.2 + 0.1)

fig.suptitle(
    "PathMNIST: ¿Aumentar resolución invierte la asimetría de complejidad?\n"
    "Rojo = ID más complejo (inversión activa), Verde = OOD más complejo (detección correcta)",
    fontsize=11,
)
plt.tight_layout()
fig.savefig(OUT_DIR / "ratio_by_resolution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Figura 1 guardada: {OUT_DIR / 'ratio_by_resolution.png'}")


# ── Figura 2: Distribuciones de entropía por clase y resolución ───────────────

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(len(RESOLUTIONS), len(ALL_CLASSES), figure=fig, hspace=0.5, wspace=0.3)

cmap_id = plt.cm.Blues
cmap_ood = plt.cm.Reds

for i, res in enumerate(RESOLUTIONS):
    for j, cls in enumerate(ALL_CLASSES):
        ax = fig.add_subplot(gs[i, j])
        vals = results[res][cls]["entropy"]
        color = "#e74c3c" if cls in OOD_CLASSES else "#2980b9"
        ax.hist(vals, bins=20, color=color, alpha=0.75, edgecolor="k", linewidth=0.4)
        ax.axvline(vals.mean(), color="black", linewidth=1.5, linestyle="--")
        ax.set_title(
            f"{CLASS_NAMES[cls][:12]}\n({res}px) μ={vals.mean():.2f}", fontsize=7, fontweight="bold"
        )
        ax.tick_params(labelsize=6)
        ax.set_yticks([])
        if i < len(RESOLUTIONS) - 1:
            ax.set_xticks([])

fig.suptitle(
    "Distribución de Entropía de Shannon por clase y resolución\n"
    "Azul = ID · Rojo = OOD · línea = media",
    fontsize=12,
    fontweight="bold",
)
fig.savefig(OUT_DIR / "entropy_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Figura 2 guardada: {OUT_DIR / 'entropy_distributions.png'}")


# ── Figura 3: Separabilidad cuantitativa (d' de Cohen) por resolución ────────


def cohen_d(a, b):
    """Efecto de separabilidad entre dos grupos (positivo = b > a)."""
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2 + 1e-9)
    return (b.mean() - a.mean()) / pooled_std


print("\n── d' DE COHEN  (OOD vs ID) — positivo = OOD más complejo ──")
print(f"{'Res':>5}  {'entropy':>10}  {'local_var':>10}  {'grad_mag':>10}")
print("-" * 42)

cohen_data = {m: [] for m in ("entropy", "local_var", "grad_mag")}
for res in RESOLUTIONS:
    row = f"{res:>5}"
    for metric in ("entropy", "local_var", "grad_mag"):
        id_vals = np.concatenate([results[res][c][metric] for c in ID_CLASSES])
        ood_vals = np.concatenate([results[res][c][metric] for c in OOD_CLASSES])
        d = cohen_d(id_vals, ood_vals)
        cohen_data[metric].append(d)
        row += f"  {d:>+10.3f}"
    print(row)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(RESOLUTIONS))
w = 0.25
for k, (metric, label) in enumerate(metrics_labels.items()):
    vals = cohen_data[metric]
    colors = ["#27ae60" if v > 0 else "#e74c3c" for v in vals]
    bars = ax.bar(
        x + k * w, vals, w, label=label, color=colors, alpha=0.85, edgecolor="k", linewidth=0.7
    )

ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_xticks(x + w)
ax.set_xticklabels([f"{r}px" for r in RESOLUTIONS], fontsize=11)
ax.set_ylabel("d' de Cohen (OOD vs ID)", fontsize=11)
ax.set_title(
    "Separabilidad OOD/ID por resolución\n"
    "Verde = OOD más complejo (detectable) · Rojo = inversión activa",
    fontsize=11,
    fontweight="bold",
)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "cohen_d_separability.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigura 3 guardada: {OUT_DIR / 'cohen_d_separability.png'}")


# ── Veredicto final ────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("VEREDICTO")
print("=" * 72)
for res in RESOLUTIONS:
    ratios = [ratio_table[res][m] for m in ("entropy", "local_var", "grad_mag")]
    n_inverted = sum(r < 1.0 for r in ratios)
    status = "INVERSIÓN ACTIVA" if n_inverted >= 2 else "Asimetría correcta"
    print(f"  {res:>3}px → ratios {[f'{r:.3f}' for r in ratios]}  →  {status}")
print()
print("Resolución óptima para romper la inversión:")
best_res = max(
    RESOLUTIONS, key=lambda r: min(ratio_table[r][m] for m in ("entropy", "local_var", "grad_mag"))
)
print(f"  → {best_res}px (mayor ratio mínimo en todas las métricas)")
print("=" * 72)
