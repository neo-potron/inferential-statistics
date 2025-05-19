import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, t


# ========== PART 1 : Estimation de la moyenne par échantillonnage ==========

def simulate_sample_means(n, M):
    """
    Simule M échantillons de taille n et retourne leurs moyennes.

    Args:
        n (int): Taille de chaque échantillon.
        M (int): Nombre d'échantillons simulés.

    Returns:
        np.ndarray: Tableau des moyennes d'échantillon.
    """
    samples = np.random.randn(M, n)
    return samples.mean(axis=1)


def plot_histograms_fixed_n(n, Ms):
    """
    Trace des histogrammes des moyennes d'échantillon pour un n fixé et différents M.

    Args:
        n (int): Taille de chaque échantillon.
        Ms (list[int]): Liste du nombre d’échantillons à tester.
    """
    sigma = 1
    theo_std = sigma / np.sqrt(n)
    x = np.linspace(-1, 1, 200)

    plt.figure(figsize=(15, 5))
    for i, M in enumerate(Ms):
        means = simulate_sample_means(n, M)
        plt.subplot(1, 3, i + 1)
        plt.hist(means, bins=30, density=True, alpha=0.7, label="Histogramme")
        plt.plot(x, norm.pdf(x, 0, theo_std), 'r-', lw=2, label="Densité théorique")
        plt.title(f"M={M}, n={n}\nMoyenne={means.mean():.3f}, Var={means.var():.3f}")
        plt.xlabel("Moyenne"); plt.ylabel("Densité"); plt.legend()
    plt.tight_layout()
    plt.show()


def plot_histograms_varying_n(M, ns):
    """
    Trace des histogrammes pour M fixé et différents n.

    Args:
        M (int): Nombre d’échantillons simulés.
        ns (list[int]): Liste de tailles d’échantillons.
    """
    colors = ['blue', 'orange', 'green', 'red']
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2], wspace=0.3, hspace=0.3)

    # Histogrammes séparés
    for i, n in enumerate(ns):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        means = simulate_sample_means(n, M)
        ax.hist(means, bins=20, density=True, alpha=0.7, color=colors[i])
        ax.set_title(f"n={n}, Var estimée: {means.var():.4f}")
        ax.set_xlabel("Moyenne"); ax.set_ylabel("Densité")

    # Histogrammes superposés
    ax_superpose = fig.add_subplot(gs[:, 2])
    for n, color in zip(ns, colors):
        means = simulate_sample_means(n, M)
        ax_superpose.hist(means, bins=30, density=True, alpha=0.5, color=color, label=f'n={n}')
    ax_superpose.set_title("Histogrammes superposés")
    ax_superpose.set_xlabel("Moyenne"); ax_superpose.set_ylabel("Densité")
    ax_superpose.legend()
    plt.show()


# ========== PART 2 : Intervalles de confiance ==========

def plot_normal_critical_regions(alpha=0.05):
    """
    Affiche les zones critiques sur la courbe d’une loi normale centrée réduite.

    Args:
        alpha (float): Niveau de signification.
    """
    x = np.linspace(-5, 5, 1000)
    y = norm.pdf(x)
    xcrit_low = norm.ppf(alpha / 2)
    xcrit_high = norm.ppf(1 - alpha / 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x, y)
    axes[0].set_title("Densité de la loi normale N(0,1)")
    axes[0].grid()

    axes[1].plot(x, y)
    axes[1].fill_between(x[x < xcrit_low], y[x < xcrit_low], 0, color='red', alpha=0.5)
    axes[1].fill_between(x[x > xcrit_high], y[x > xcrit_high], 0, color='red', alpha=0.5)
    axes[1].set_title(f"Zones critiques (α = {alpha})")
    axes[1].grid()
    plt.show()


def estimate_confidence_interval_normal(n, T=10000, alpha=0.05):
    """
    Évalue la fréquence de couverture d'un IC basé sur la loi normale.

    Args:
        n (int): Taille de l’échantillon.
        T (int): Nombre de répétitions.
        alpha (float): Niveau de confiance.

    Returns:
        float: Taux d'erreur empirique.
    """
    xcrit = norm.ppf(1 - alpha / 2)
    contain_true_mu = 0

    for _ in range(T):
        samples = np.random.randn(n)
        mu = samples.mean()
        sigma = samples.std()
        lower = mu - xcrit * sigma / np.sqrt(n)
        upper = mu + xcrit * sigma / np.sqrt(n)
        if lower <= 0 <= upper:
            contain_true_mu += 1

    error_rate = 1 - contain_true_mu / T
    return error_rate


def estimate_confidence_interval_student(n, T=10000, alpha=0.05):
    """
    Évalue la fréquence de couverture d'un IC basé sur la loi de Student.

    Args:
        n (int): Taille de l’échantillon.
        T (int): Nombre de répétitions.
        alpha (float): Niveau de confiance.

    Returns:
        float: Taux d'erreur empirique.
    """
    t_crit = t.ppf(1 - alpha / 2, df=n - 1)
    contain_true_mu = 0

    for _ in range(T):
        samples = np.random.randn(n)
        mu = samples.mean()
        sigma = samples.std(ddof=1)
        lower = mu - t_crit * sigma / np.sqrt(n)
        upper = mu + t_crit * sigma / np.sqrt(n)
        if lower <= 0 <= upper:
            contain_true_mu += 1

    error_rate = 1 - contain_true_mu / T
    return error_rate
