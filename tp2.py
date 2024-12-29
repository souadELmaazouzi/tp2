import numpy as np
import matplotlib.pyplot as plt

# Fonction coût
def C(x):
    return 400 * x + 500 * np.sqrt((x - 10)**2 + 9)

# Méthode du gradient projeté
def gradient_projected(learning_rate=0.01, tol=1e-6, max_iter=100):
    x = 5  # Initialisation
    cost_values = [C(x)]
    x_values = [x]
    for _ in range(max_iter):
        grad = 400 + 500 * (x - 10) / np.sqrt((x - 10)**2 + 9)
        x_new = x - learning_rate * grad
        # Projeter x dans le domaine admissible
        x_new = max(x_new, 3)
        if abs(x_new - x) < tol:
            break
        x = x_new
        x_values.append(x)
        cost_values.append(C(x))
    return x_values, cost_values

# Méthode de pénalisation
def penalization_method(learning_rate=0.01, lambda_param=100, tol=1e-6, max_iter=100):
    x = 5  # Initialisation
    cost_values = [C(x)]
    x_values = [x]
    for _ in range(max_iter):
        cost_penalized = C(x) + lambda_param * max(0, 3 - x)**2
        grad = 400 + 500 * (x - 10) / np.sqrt((x - 10)**2 + 9)
        grad_penalized = grad + 2 * lambda_param * (3 - x) * (x < 3)
        x_new = x - learning_rate * grad_penalized
        if abs(x_new - x) < tol:
            break
        x = x_new
        x_values.append(x)
        cost_values.append(C(x))
    return x_values, cost_values

# Algorithme d'Uzawa
def uzawa_algorithm(learning_rate=0.01, rho=0.1, tol=1e-6, max_iter=100):
    x = 5  # Initialisation
    lambda_param = 0
    cost_values = [C(x)]
    x_values = [x]
    lambda_values = [lambda_param]
    for _ in range(max_iter):
        grad = 400 + 500 * (x - 10) / np.sqrt((x - 10)**2 + 9)
        x_new = x - learning_rate * grad
        lambda_new = lambda_param + rho * (3 - x_new)
        # Respecter la contrainte x >= 3
        x_new = max(x_new, 3)
        if abs(x_new - x) < tol and abs(lambda_new - lambda_param) < tol:
            break
        x = x_new
        lambda_param = lambda_new
        x_values.append(x)
        lambda_values.append(lambda_param)
        cost_values.append(C(x))
    return x_values, cost_values, lambda_values

# Créer une figure avec trois sous-graphiques (sans la fonction de coût)
fig, ax = plt.subplots(3, 1, figsize=(10, 12))

# Méthode du gradient projeté
x_proj, cost_proj = gradient_projected()
ax[0].plot(x_proj, cost_proj, label='Méthode du gradient projeté', color='blue', marker='o')
ax[0].set_title("Méthode du gradient projeté")
ax[0].set_xlabel("x (Position de P)")
ax[0].set_ylabel("Coût total")
ax[0].grid(True)
ax[0].legend()

# Méthode de pénalisation
x_pen, cost_pen = penalization_method()
ax[1].plot(x_pen, cost_pen, label='Méthode de pénalisation', color='red', marker='x')
ax[1].set_title("Méthode de pénalisation")
ax[1].set_xlabel("x (Position de P)")
ax[1].set_ylabel("Coût total")
ax[1].grid(True)
ax[1].legend()

# Algorithme d'Uzawa
x_uzawa, cost_uzawa, lambda_uzawa = uzawa_algorithm()
ax[2].plot(x_uzawa, cost_uzawa, label='Algorithme d\'Uzawa', color='green', marker='s')
ax[2].set_title("Algorithme d'Uzawa")
ax[2].set_xlabel("x (Position de P)")
ax[2].set_ylabel("Coût total")
ax[2].grid(True)
ax[2].legend()

# Afficher les graphiques
plt.tight_layout()
plt.show()

# Affichage des résultats d'optimisation
final_x_proj = x_proj[-1]
final_cost_proj = cost_proj[-1]
final_x_pen = x_pen[-1]
final_cost_pen = cost_pen[-1]
final_x_uzawa = x_uzawa[-1]
final_cost_uzawa = cost_uzawa[-1]

# Affichage des résultats pour chaque méthode
print(f"Optimisation (gradient projeté) : x = {final_x_proj}, coût = {final_cost_proj}")
print(f"Optimisation (méthode de pénalisation) : x = {final_x_pen}, coût = {final_cost_pen}")
print(f"Optimisation (algorithme d'Uzawa) : x = {final_x_uzawa}, coût = {final_cost_uzawa}")
