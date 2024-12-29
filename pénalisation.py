import numpy as np
import matplotlib.pyplot as plt

# Fonction coût
def cout(x, y):
    return 500 * x - 400 * y + 4000

# Fonction de gradient de la fonction coût
def gradient(x, y):
    return np.array([500, -400])

# Fonction de pénalisation extérieure
def penalisation(x, y, lambda_pen):
    g = max(0, 3 - y) + max(0, y - 10) + max(0, 32 - x) + max(0, x - 109)
    return g

# Mise à jour avec pénalisation extérieure
def mise_a_jour(x, y, alpha, lambda_pen):
    grad = gradient(x, y)
    
    # Mise à jour de x et y avec pénalisation
    x_new = x - alpha * (grad[0] + lambda_pen * (2 * max(0, x - 109) - 2 * max(0, 32 - x)))
    y_new = y - alpha * (grad[1] + lambda_pen * (2 * max(0, y - 10) - 2 * max(0, 3 - y)))
    
    # Limiter x et y dans les bornes [32, 109] et [3, 10]
    x_new = np.clip(x_new, 32, 109)
    y_new = np.clip(y_new, 3, 10)
    
    return x_new, y_new

# Fonction de pénalisation extérieure
def penalisation_exterieure(alpha=0.01, lambda_pen=1, tol=1e-6, max_iter=1000):
    # Initialisation de x et y
    x = 50
    y = 5
    
    trajectory = []  # Liste pour enregistrer la trajectoire
    for _ in range(max_iter):
        x_new, y_new = mise_a_jour(x, y, alpha, lambda_pen)
        
        # Enregistrer la trajectoire et le coût
        trajectory.append((x_new, y_new, cout(x_new, y_new)))
        
        # Critère d'arrêt
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            break
        
        # Mise à jour des valeurs de x et y
        x, y = x_new, y_new
        
        # Augmenter la pénalité lambda lentement
        lambda_pen *= 1.05  # Augmentation plus lente de la pénalité
        
    return x, y, cout(x, y), trajectory

# Visualisation de la trajectoire
def plot_trajectory(trajectory):
    traj_x, traj_y, traj_z = zip(*trajectory)
    plt.plot(traj_x, traj_y, label="Trajectoire")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectoire de la méthode de pénalisation extérieure")
    plt.legend()
    plt.show()

# Exécution
x_opt, y_opt, cost_min, trajectory = penalisation_exterieure()
print(f"Pénalisation Extérieure - Position optimale: P(x={x_opt:.2f}, y={y_opt:.2f}), Coût minimal: {cost_min:.2f} Dhs")
plot_trajectory(trajectory)
