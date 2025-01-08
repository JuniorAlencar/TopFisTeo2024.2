import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função para gerar as posições das partículas
def direct_disks_box(N, sigma):
    condition = False
    while not condition:
        L = [(random.uniform(sigma, 1.0 - sigma), random.uniform(sigma, 1.0 - sigma))]
        for k in range(1, N):
            a = (random.uniform(sigma, 1.0 - sigma), random.uniform(sigma, 1.0 - sigma))
            min_dist = min(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) for b in L)
            if min_dist < 2.0 * sigma:
                condition = False
                break
            else:
                L.append(a)
                condition = True
    return L

# Parâmetros
N = 4
colors = ['r', 'b', 'g', 'orange']
sigma = 0.2
n_runs = 8

# Configuração da figura
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

circles = []

# Inicializa os círculos (partículas)
for color in colors:
    circle = plt.Circle((0, 0), radius=sigma, fc=color, animated=True)
    ax.add_patch(circle)
    circles.append(circle)

# Atualização do frame
def update(frame):
    pos = direct_disks_box(N, sigma)
    for circle, (x, y) in zip(circles, pos):
        circle.set_center((x, y))
    return circles

# Animação
ani = FuncAnimation(fig, update, frames=n_runs, blit=True)

# Salvar o GIF
ani.save('direct_disks_box.gif', writer='imagemagick', fps=2)

# Mostrar a animação (opcional)
plt.show()

