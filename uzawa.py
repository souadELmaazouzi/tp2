import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fonction coût
def cout(x, y):
    return 500 * x - 400 * y + 4000

# Gradient de la fonction coût
def gradient():
    return np.array([500, -400])

# Fonction de mise à jour des multiplicateurs de Lagrange (Uzawa) avec pénalisation plus forte
def update_lambda(lambda_val, g, factor=1):
    # Mise à jour plus rapide des multiplicateurs de Lagrange
    return lambda_val + factor * g

# Algorithme d'Uzawa avec pénalisation renforcée
def uzawa_algorithm(alpha=0.01, tol=1e-6, max_iter=1000):
    # Initialisation des variables
    x = 50  # Valeur initiale de x
    y = 5   # Valeur initiale de y
    lambda_1 = lambda_2 = lambda_3 = lambda_4 = 0  # Multiplicateurs de Lagrange initialisés à 0
    trajectory = []  # Liste pour stocker la trajectoire
    penalty_factor = 10  # Facteur de pénalisation plus élevé
    
    for _ in range(max_iter):
        # Calcul du gradient de la fonction coût
        grad = gradient()
        
        # Calcul des violations des contraintes
        g1 = max(0, x - 32)
        g2 = max(0, 109 - x)
        g3 = max(0, y - 3)
        g4 = max(0, 10 - y)
        
        # Mise à jour des multiplicateurs de Lagrange avec pénalisation
        lambda_1 = update_lambda(lambda_1, g1, penalty_factor)
        lambda_2 = update_lambda(lambda_2, g2, penalty_factor)
        lambda_3 = update_lambda(lambda_3, g3, penalty_factor)
        lambda_4 = update_lambda(lambda_4, g4, penalty_factor)
        
        # Mise à jour des variables primales (x, y) en fonction du gradient et des multiplicateurs
        x_new = x - alpha * (grad[0] + lambda_1 - lambda_2)
        y_new = y - alpha * (grad[1] + lambda_3 - lambda_4)
        
        # Appliquer les contraintes directement sur x et y pour éviter les violations
        x_new = np.clip(x_new, 32, 109)
        y_new = np.clip(y_new, 3, 10)
        
        trajectory.append((x_new, y_new, cout(x_new, y_new)))  # Ajouter à la trajectoire
        
        # Condition d'arrêt
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break
        
        # Mise à jour des variables
        x, y = x_new, y_new
    
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

    # Trajectoire de l'algorithme d'Uzawa
    traj_x, traj_y, traj_z = zip(*trajectory)
    ax.plot(traj_x, traj_y, traj_z, color='r', marker='o', label="Trajectoire")  # Trajectoire
    ax.set_xlabel('x ')
    ax.set_ylabel('y ')
    ax.set_zlabel('Coût')
    ax.set_title('Algorithme d\'Uzawa - Visualisation 3D')
    plt.legend()
    plt.show()

# Exécution de l'algorithme d'Uzawa
x_opt, y_opt, cost_min, trajectory = uzawa_algorithm()
print(f"Uzawa - Position optimale: P(x={x_opt:.2f}, y={y_opt:.2f}), Coût minimal: {cost_min:.2f} Dhs")

# Visualisation de la trajectoire
plot_3d(trajectory)
