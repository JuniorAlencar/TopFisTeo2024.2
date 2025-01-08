import math
import random
import matplotlib.pyplot as plt
import numpy as np

def wall_time(pos_a, vel_a, sigma):
    if vel_a > 0.0:
        del_t = (1.0 - sigma - pos_a) / vel_a
    elif vel_a < 0.0:
        del_t = (pos_a - sigma) / abs(vel_a)
    else:
        del_t = float('inf')
    return del_t

def pair_time(pos_a, vel_a, pos_b, vel_b, sigma):
    del_x = [pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]]
    del_x_sq = del_x[0] ** 2 + del_x[1] ** 2
    del_v = [vel_b[0] - vel_a[0], vel_b[1] - vel_a[1]]
    del_v_sq = del_v[0] ** 2 + del_v[1] ** 2
    scal = del_v[0] * del_x[0] + del_v[1] * del_x[1]
    Upsilon = scal ** 2 - del_v_sq * ( del_x_sq - 4.0 * sigma **2)
    if Upsilon > 0.0 and scal < 0.0:
        del_t = - (scal + math.sqrt(Upsilon)) / del_v_sq
    else:
        del_t = float('inf')
    return del_t

# Definir o número de partículas
N = 4
# Define o raio das partículas
sigma = 0.1

# Inicializar posições aleatórias para N partículas
pos = []
while len(pos) < N:
    x = random.uniform(sigma, 1.0 - sigma)
    y = random.uniform(sigma, 1.0 - sigma)
    pos.append([x, y])

# Inicializar velocidades aleatórias para N partículas
vel = []
for i in range(N):
    vx = random.uniform(-1, 1)
    vy = random.uniform(-1, 1)
    vel.append([vx, vy])

# Atualizar a lista 'singles' para colisões com as paredes
singles = []
for k in range(N):
    singles.append((k, 0))  # Componente x
    singles.append((k, 1))  # Componente y

# Atualizar a lista 'pairs' para colisões entre partículas
pairs = []
for i in range(N):
    for j in range(i+1, N):
        pairs.append((i, j))

t = 0.0
n_events = 10000  # Aumentado para melhorar a média temporal

# Lista para armazenar as posições y ao longo do tempo
y_positions_over_time = []

for event in range(n_events):
    wall_times = [wall_time(pos[k][l], vel[k][l], sigma) for k, l  in singles]
    pair_times = [pair_time(pos[k], vel[k], pos[l], vel[l], sigma) for k, l in pairs]
    next_event = min(wall_times + pair_times)
    t += next_event
    for k, l in singles:
        pos[k][l] += vel[k][l] * next_event
    if min(wall_times) < min(pair_times):
        collision_disk, direction = singles[wall_times.index(next_event)]
        vel[collision_disk][direction] *= -1.0
    else:
        a, b = pairs[pair_times.index(next_event)]
        del_x = [pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]]
        abs_x = math.sqrt(del_x[0] ** 2 + del_x[1] ** 2)
        e_perp = [c / abs_x for c in del_x]
        del_v = [vel[b][0] - vel[a][0], vel[b][1] - vel[a][1]]
        scal = del_v[0] * e_perp[0] + del_v[1] * e_perp[1]
        for k in range(2):
            vel[a][k] += e_perp[k] * scal
            vel[b][k] -= e_perp[k] * scal
    # Armazenar as posições y das partículas para a média temporal
    y_positions_over_time.extend([pos[k][1] for k in range(N)])
    # print 'event', event
    # print 'time', t
    # print 'pos', pos
    # print 'vel', vel

# Calcular a densidade ao longo do eixo y usando a média temporal
num_bins = 50
y_bins = np.linspace(0, 1, num_bins + 1)
density, _ = np.histogram(y_positions_over_time, bins=y_bins, density=True)

# Plotar a densidade ao longo do eixo y
y_centers = (y_bins[:-1] + y_bins[1:]) / 2
plt.plot(y_centers, density)
plt.xlabel('Posição y')
plt.ylabel('Densidade')
plt.title('Densidade de discos ao longo do eixo y para N = %.0f' % N)
plt.savefig("./img/density_y_%.0f.jpg" % N, dpi=300)

