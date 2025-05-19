import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, norm

def test_chi2_independence(table: np.ndarray, variable_name: str) -> None:
    """
    Effectue un test du chi² d'indépendance sur un tableau de contingence.

    Args:
        table (np.ndarray): Tableau de contingence (catégories en ligne, groupes en colonne).
        variable_name (str): Nom de la variable étudiée (pour affichage).

    Prints:
        Résultat du test du chi² et tableau des effectifs attendus sous H0.
    """
    print(f"Test du khi² d'indépendance pour {variable_name} :")
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"  Statistique χ² = {chi2:.3f}, ddl = {dof}, p-value = {p:.4f}")
    if p < 0.05:
        print("  → Rejet de H0 : les distributions sont différentes entre groupes.\n")
    else:
        print("  → Pas de rejet de H0 : les distributions sont similaires entre groupes.\n")
    print("  Table attendue sous H0 :\n", expected, "\n")

def estimer_moments_ldl(somme_variations: float, somme_carres: float, n: int) -> tuple:
    """
    Calcule la moyenne, la variance et l'écart-type de la variation du taux LDL.

    Args:
        somme_variations (float): Somme des variations de LDL.
        somme_carres (float): Somme des carrés des écarts à la moyenne.
        n (int): Effectif.

    Returns:
        tuple: Moyenne, variance et écart-type.
    """
    mean = somme_variations / n
    variance = somme_carres / (n - 1)
    std = np.sqrt(variance)
    return mean, variance, std

def afficher_boxplots_ldl(data_traitement, data_placebo):
    """
    Affiche des boxplots de la variation du taux LDL pour les deux groupes.

    Args:
        data_traitement (np.ndarray): Données simulées pour le groupe traitement.
        data_placebo (np.ndarray): Données simulées pour le groupe placebo.
    """
    plt.figure(figsize=(6, 4))
    plt.boxplot([data_traitement, data_placebo], labels=['Traitement', 'Placebo'])
    plt.ylabel("Variation du taux LDL (g/l)")
    plt.title("Distribution des variations du taux LDL par groupe")
    plt.grid(True)
    plt.show()

def tracer_barres_comparatives(obs, exp, labels, title, axe):
    """
    Trace un graphique en barres comparant les valeurs observées et attendues.

    Args:
        obs (np.ndarray): Effectifs observés.
        exp (np.ndarray): Effectifs attendus.
        labels (list): Libellés des catégories.
        title (str): Titre du graphique.
        axe (matplotlib.axes.Axes): Axe sur lequel tracer.
    """
    x = np.arange(len(labels))
    width = 0.35

    axe.bar(x - width/2, obs[:, 0], width, label='Traitement (observé)', color='blue', alpha=0.7)
    axe.bar(x + width/2, obs[:, 1], width, label='Placebo (observé)', color='orange', alpha=0.7)
    axe.plot(x - width/2, exp[:, 0], 'o--', color='black', label='Traitement (attendu)')
    axe.plot(x + width/2, exp[:, 1], 'o--', color='red', label='Placebo (attendu)')
    axe.set_title(title)
    axe.set_xticks(x)
    axe.set_xticklabels(labels)
    axe.set_ylabel("Effectifs")
    axe.grid(True)

def intervalle_confiance_diff_moyennes(m1, v1, n1, m2, v2, n2, alpha=0.05):
    """
    Calcule un intervalle de confiance et réalise un test unilatéral gauche
    pour la différence entre deux moyennes indépendantes.

    Returns:
        tuple: (delta, IC bas, IC haut, Z, p-value)
    """
    delta = m1 - m2
    se = np.sqrt(v1 / n1 + v2 / n2)
    z = delta / se
    p = norm.cdf(z)  # Test unilatéral gauche
    z_crit = norm.ppf(1 - alpha / 2)
    ci_lower = delta - z_crit * se
    ci_upper = delta + z_crit * se
    return delta, ci_lower, ci_upper, z, p

# Données de contingence (sexe et âge)
sexe_data = np.array([[428, 502], [399, 390]])
age_data = np.array([[162, 210], [403, 423], [262, 259]])

# Tests du khi² d'indépendance
test_chi2_independence(sexe_data, "le sexe")
test_chi2_independence(age_data, "l'âge")

# Données de variation LDL
n_traitement, S_traitement, SC_traitement = 827, -528.4477, 76.1925
n_placebo, S_placebo, SC_placebo = 892, -116.0837, 35.8899

mean_t, var_t, std_t = estimer_moments_ldl(S_traitement, SC_traitement, n_traitement)
mean_p, var_p, std_p = estimer_moments_ldl(S_placebo, SC_placebo, n_placebo)

print("Estimations ponctuelles de la variation du taux LDL :")
print(f"Traitement : moyenne = {mean_t:.4f} g/l, variance = {var_t:.6f}, écart-type = {std_t:.4f} g/l")
print(f"Placebo   : moyenne = {mean_p:.4f} g/l, variance = {var_p:.6f}, écart-type = {std_p:.4f} g/l\n")

# Boxplots à partir de simulations
np.random.seed(0)
sim_t = np.random.normal(loc=mean_t, scale=std_t, size=n_traitement)
sim_p = np.random.normal(loc=mean_p, scale=std_p, size=n_placebo)
afficher_boxplots_ldl(sim_t, sim_p)

# Graphiques barres observé vs attendu
obs_sexe = sexe_data
obs_age = age_data
att_sexe = np.array([[447.42, 482.58], [379.58, 409.42]])
att_age = np.array([[178.97, 193.03], [397.38, 428.62], [250.65, 270.35]])
labels_sexe = ['Femmes', 'Hommes']
labels_age = ['25-44 ans', '45-64 ans', '> 65 ans']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
tracer_barres_comparatives(obs_sexe, att_sexe, labels_sexe, "Répartition par sexe", axes[0])
tracer_barres_comparatives(obs_age, att_age, labels_age, "Répartition par âge", axes[1])
axes[1].legend()
plt.tight_layout()
plt.show()

# Calcul de l'IC et du test
delta, ci_low, ci_up, z_stat, p_val = intervalle_confiance_diff_moyennes(
    mean_t, var_t, n_traitement, mean_p, var_p, n_placebo
)

print(f"Estimation ponctuelle de la différence des moyennes : {delta:.4f} g/l")
print(f"Intervalle de confiance à 95% : [{ci_low:.4f}, {ci_up:.4f}] g/l")
print(f"Statistique de test Z : {z_stat:.4f}")
print(f"p-value : {p_val:.8f}")
if p_val < 0.05:
    print("→ Le test est significatif : le traitement est plus efficace que le placebo.")
else:
    print("→ Le test n'est pas significatif : on ne peut pas conclure que le traitement est plus efficace.")
