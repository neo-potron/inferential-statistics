import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, chi2, t, levene, ttest_ind

# Chargement des données
data = np.loadtxt("dataA.txt")  # Adapter le chemin si nécessaire
peres = data[:, 0]
meres = data[:, 1]



# ========== PART 1 : Statistiques descriptives + Visualisation ==========

def afficher_histogrammes(pères, mères):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramme
    axs[0].hist(pères, bins=20, alpha=0.6, color='blue', label='Pères', density=True)
    axs[0].hist(mères, bins=20, alpha=0.6, color='green', label='Mères', density=True)
    axs[0].set_title("Histogramme")
    axs[0].set_xlabel("Taille (pouces)")
    axs[0].set_ylabel("Densité")
    axs[0].legend()
    axs[0].grid(True)

    # Histogramme cumulé
    axs[1].hist(pères, bins=20, cumulative=True, alpha=0.6, color='blue', label='Pères', density=True)
    axs[1].hist(mères, bins=20, cumulative=True, alpha=0.6, color='green', label='Mères', density=True)
    axs[1].set_title("Histogramme cumulé")
    axs[1].set_xlabel("Taille (pouces)")
    axs[1].set_ylabel("Densité cumulée")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def afficher_boxplots(pères, mères):
    plt.boxplot([pères, mères], labels=['Pères', 'Mères'])
    plt.title("Boîtes à moustaches")
    plt.ylabel("Taille (pouces)")
    plt.grid(True)
    plt.show()

def afficher_statistiques(data, nom):
    print(f"\n=== Statistiques descriptives : {nom} ===")
    print(f"Moyenne : {np.mean(data):.2f}")
    print(f"Variance (non biaisée) : {np.var(data, ddof=1):.2f}")
    print(f"Écart-type (non biaisé) : {np.std(data, ddof=1):.2f}")
    print(f"Variance (biaisée) : {np.var(data, ddof=0):.2f}")
    print(f"Écart-type (biaisé) : {np.std(data, ddof=0):.2f}")
    print(f"Asymétrie (non biaisée) : {skew(data, bias=False):.2f}")
    print(f"Aplatissement (non biaisé) : {kurtosis(data, bias=False):.2f}")

# Exécution
afficher_histogrammes(peres, meres)
afficher_boxplots(peres, meres)
afficher_statistiques(peres, "Pères")
afficher_statistiques(meres, "Mères")




# ========== PART 2 : Test de conformité à la loi normale (χ² + Droite de Henri) ==========

def merge_bins(O, E, bins):
    """Fusionne les classes si Ei < 5 en extrémités"""
    O, E, bins = O.copy(), E.copy(), bins.copy()
    while len(E) > 1 and E[0] < 5:
        E[1] += E[0]; O[1] += O[0]
        E, O, bins = E[1:], O[1:], bins[1:]
    while len(E) > 1 and E[-1] < 5:
        E[-2] += E[-1]; O[-2] += O[-1]
        E, O, bins = E[:-1], O[:-1], bins[:-1]
    return O, E, bins

def chi2_test_gaussienne(data, nbins=10, nom=""):
    """Test du χ² d'ajustement à une loi normale"""
    N, mean, std = len(data), np.mean(data), np.std(data, ddof=1)
    bins = np.linspace(min(data), max(data), nbins + 1)
    O, _ = np.histogram(data, bins=bins)
    E = [N * (norm.cdf(bins[i+1], mean, std) - norm.cdf(bins[i], mean, std)) for i in range(nbins)]
    O, E, bins = merge_bins(np.array(O), np.array(E), bins)
    chi_stat = np.sum((O - E) ** 2 / E)
    ddl = len(O) - 3  # -1 (somme = N) -2 (moyenne, std estimés)
    p_value = 1 - chi2.cdf(chi_stat, ddl)

    print(f"\n=== Test χ² : {nom} ===")
    print(f"Statistique : {chi_stat:.2f}, ddl : {ddl}, p-value : {p_value:.4f}")
    conclusion = "On rejette H0 (non gaussien)" if p_value < 0.05 else "On ne rejette pas H0 (gaussien)"
    print("→", conclusion)

    centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]
    return centers, O, E, width, data

def plot_hist(ax, centers, O, E, width, nom):
    ax.bar(centers, O, width=width, alpha=0.6, label="Observé", color="blue", edgecolor="black")
    ax.plot(centers, E, 'ro--', label="Attendu (Gaussienne)")
    ax.set_title(f"Test χ² : {nom}")
    ax.legend(); ax.set_xlabel("Taille"); ax.set_ylabel("Effectif")
    ax.grid(True)

def plot_henri(ax, data, nom):
    n = len(data)
    sorted_data = np.sort(data)
    proba = (np.arange(1, n + 1) - 0.5) / n
    theor = norm.ppf(proba)
    ax.plot(theor, sorted_data, 'o', label="Données")
    slope, intercept = np.polyfit(theor, sorted_data, 1)
    ax.plot(theor, slope * theor + intercept, 'r--', label="Droite de Henri")
    ax.set_title(f"Droite de Henri : {nom}")
    ax.legend(); ax.set_xlabel("Quantiles théoriques"); ax.set_ylabel("Quantiles empiriques")
    ax.grid(True)

# Résultats
results_peres = chi2_test_gaussienne(peres, nom="Pères")
results_meres = chi2_test_gaussienne(meres, nom="Mères")

# Affichage
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plot_hist(axs[0, 0], *results_peres[:4], nom="Pères")
plot_henri(axs[1, 0], results_peres[4], nom="Pères")
plot_hist(axs[0, 1], *results_meres[:4], nom="Mères")
plot_henri(axs[1, 1], results_meres[4], nom="Mères")
plt.tight_layout()
plt.show()
