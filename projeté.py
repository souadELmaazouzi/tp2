import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fonction coût
def cout(x, y):
    return 500 * x - 400 * y + 4000

# Gradient de la fonction coût
def gradient():
    return np.array([500, -400])

# Fonction de projection sur les contraintes
def project(x, y):
    y = np.clip(y, 3, 10)  # Contrainte sur y
    x = np.sqrt(9 + y**2)  # Contrainte sur x
    return x, y

# Méthode du Gradient Projeté
def gradient_projete(alpha=0.01, tol=1e-6, max_iter=1000):
    y = 5  # Valeur initiale de y
    x = np.sqrt(9 + y**2)  # Calcul initial de x à partir de y
    trajectory = []  # Liste pour stocker la trajectoire
    for _ in range(max_iter):
        grad = gradient()  # Gradient de la fonction coût
        x_new, y_new = project(x - alpha * grad[0], y - alpha * grad[1])  # Mise à jour et projection
        trajectory.append((x_new, y_new, cout(x_new, y_new)))  # Ajouter la position et le coût
        if np.linalg.norm([x_new - x, y_new - y]) < tol:  # Condition de convergence
            break
        x, y = x_new, y_new  # Mise à jour des positions
    return x, y, cout(x, y), trajectory

# Visualisation 3D de la trajectoire
def plot_3d(trajectory):
    y_vals = np.linspace(3, 10, 100)  # Plage de valeurs de y
    x_vals = np.sqrt(9 + y_vals**2)  # Calcul de x pour chaque y
    X, Y = np.meshgrid(x_vals, y_vals)  # Grille pour x et y
    Z = cout(X, Y)  # Calcul du coût pour chaque point de la grille

    # Création du graphique
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)  # Surface du coût

    # Trajectoire du Gradient Projeté
    traj_x, traj_y, traj_z = zip(*trajectory)
    ax.plot(traj_x, traj_y, traj_z, color='r', marker='o', label="Trajectoire")  # Trajectoire
    ax.set_xlabel('x ')
    ax.set_ylabel('y ')
    ax.set_zlabel('Coût')
    ax.set_title('Gradient Projeté - Visualisation 3D')
    plt.legend()
    plt.show()

# Exécution de la méthode du Gradient Projeté
x_opt, y_opt, cost_min, trajectory = gradient_projete()
print(f"Gradient Projeté - Position optimale: P(x={x_opt:.2f}, y={y_opt:.2f}), Coût minimal: {cost_min:.2f} Dhs")

# Visualisation de la trajectoire
plot_3d(trajectory)
