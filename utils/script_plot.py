import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


# === Parser dei log ===
def parse_log_file(filepath):
    """
    Estrae solo Loss e Accuracy dai blocchi di metriche nel log.
    Restituisce {model_name: {"Loss": float, "Accuracy": float}}
    """
    results = {}
    with open(filepath, "r") as f:
        text = f.read()

    # Prendi solo la parte dopo "=== Final Results ==="
    if "=== Final Results ===" in text:
        text = text.split("=== Final Results ===")[-1]

    # Trova righe del tipo: Model: { ... }
    matches = re.findall(r"(\w+):\s*({.*?})", text, re.S)
    for model_name, metrics_str in matches:
        try:
            # Rimuovi il campo 'Confusion Matrix': array([...])
            metrics_str = re.sub(
                r",\s*'Confusion Matrix':\s*array\(\[.*?\]\)", "", metrics_str, flags=re.S
            )

            metrics = eval(metrics_str)  # ora non c'è più array()
            # Prendi solo Loss e Accuracy
            filtered = {k: v for k, v in metrics.items() if k in ["Loss", "Accuracy"]}
            results[model_name] = filtered
        except Exception as e:
            print(f"Errore parsing {model_name} in {filepath}: {e}")

    return results


# === Aggregatore ===
def aggregate_results(log_files):
    """
    Aggrega solo Loss e Accuracy da più log.
    {model_name: {"Loss": [..], "Accuracy": [..]}}
    """
    aggregated = {}
    for file in log_files:
        run_results = parse_log_file(file)
        for model, metrics in run_results.items():
            if model not in aggregated:
                aggregated[model] = {"Loss": [], "Accuracy": []}
            for k in ["Loss", "Accuracy"]:
                if k in metrics and isinstance(metrics[k], (int, float)):
                    aggregated[model][k].append(metrics[k])
    return aggregated


# === Plot ===
def plot_loss_accuracy(aggregated):
    """
    Plot con 2 subplot (Loss sopra, Accuracy sotto).
    Mostra media ± std su più run.
    """
    models = list(aggregated.keys())

    # Loss
    loss_means = [np.mean(aggregated[m]["Loss"]) for m in models]
    loss_stds = [np.std(aggregated[m]["Loss"]) for m in models]

    # Accuracy
    acc_means = [np.mean(aggregated[m]["Accuracy"]) for m in models]
    acc_stds = [np.std(aggregated[m]["Accuracy"]) for m in models]

    # Colori
    colors = []
    cmap = get_cmap("Blues", len(models))
    for i, m in enumerate(models):
        if "Teacher" in m:
            colors.append("red")
        else:
            colors.append(cmap(i))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Loss ---
    axs[0].bar(models, loss_means, yerr=loss_stds, capsize=5, color=colors)
    axs[0].set_title("Loss per modello")
    axs[0].set_ylabel("Loss")
    axs[0].grid(axis="y", linestyle="--", alpha=0.6)

    # --- Accuracy ---
    axs[1].bar(models, acc_means, yerr=acc_stds, capsize=5, color=colors)
    axs[1].set_title("Accuracy per modello")
    axs[1].set_ylabel("Accuracy")
    axs[1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


# === Main ===
if __name__ == "__main__":
    log_dir = "./logs"
    log_files = [os.path.join(log_dir, f"main{i}.log") for i in range(1, 7)]

    aggregated = aggregate_results(log_files)
    plot_loss_accuracy(aggregated)
