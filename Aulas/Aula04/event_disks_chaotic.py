import os
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

colors = ['r', 'g', 'b', 'orange']

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
    Upsilon = scal ** 2 - del_v_sq * (del_x_sq - 4.0 * sigma ** 2)
    if Upsilon > 0.0 and scal < 0.0:
        del_t = - (scal + math.sqrt(Upsilon)) / del_v_sq
    else:
        del_t = float('inf')
    return del_t

def min_arg(l):
    return min(zip(l, range(len(l))))

def compute_next_event(pos, vel):
    wall_times = [wall_time(pos[k][l], vel[k][l], sigma) for k, l in singles]
    pair_times = [pair_time(pos[k], vel[k], pos[l], vel[l], sigma) for k, l in pairs]
    return min_arg(wall_times + pair_times)

def compute_new_velocities(pos, vel, next_event_arg):
    if next_event_arg < len(singles):
        collision_disk, direction = singles[next_event_arg]
        vel[collision_disk][direction] *= -1.0
    else:
        a, b = pairs[next_event_arg - len(singles)]
        del_x = [pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]]
        abs_x = math.sqrt(del_x[0] ** 2 + del_x[1] ** 2)
        e_perp = [c / abs_x for c in del_x]
        del_v = [vel[b][0] - vel[a][0], vel[b][1] - vel[a][1]]
        scal = del_v[0] * e_perp[0] + del_v[1] * e_perp[1]
        for k in range(2):
            vel[a][k] += e_perp[k] * scal
            vel[b][k] -= e_perp[k] * scal

def round_pos_vel(pos, vel, precision):
    # Arredonda posições e velocidades para a precisão especificada
    pos = [[round(coord, precision) for coord in p] for p in pos]
    vel = [[round(coord, precision) for coord in v] for v in vel]
    return pos, vel

# Configurações iniciais
pos1 = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
vel1 = [[0.21, 0.12], [0.71, 0.18], [-0.23, -0.79], [0.78, 0.1177]]

pos2 = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
vel2 = [[0.21, 0.12], [0.71, 0.18], [-0.23, -0.79], [0.78, 0.1177]]

singles = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
sigma = 0.15
t1 = 0.0
t2 = 0.0
dt = 0.02     # dt=0 corresponde à animação de evento a evento
n_steps = 400
arrow_scale = 0.2
precision = 4  # Número de casas decimais para arredondar na segunda simulação

next_event1, next_event_arg1 = compute_next_event(pos1, vel1)
next_event2, next_event_arg2 = compute_next_event(pos2, vel2)

# Criação da figura e dos eixos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10)

# Configuração dos eixos
for ax in [ax1, ax2]:
    ax.axis([0, 1, 0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

# Criação dos círculos e das setas de velocidade para ambas as simulações
particles1 = []
velocity_lines1 = []
particles2 = []
velocity_lines2 = []

for (x, y), (dx, dy), c in zip(pos1, vel1, colors):
    # Primeira simulação
    circle1 = plt.Circle((x, y), radius=sigma, fc=c)
    ax1.add_patch(circle1)
    particles1.append(circle1)
    dx_scaled = dx * arrow_scale
    dy_scaled = dy * arrow_scale
    line1, = ax1.plot([x, x + dx_scaled], [y, y + dy_scaled], 'k-')
    velocity_lines1.append(line1)
    
    # Segunda simulação
    circle2 = plt.Circle((x, y), radius=sigma, fc=c)
    ax2.add_patch(circle2)
    particles2.append(circle2)
    line2, = ax2.plot([x, x + dx_scaled], [y, y + dy_scaled], 'k-')
    velocity_lines2.append(line2)

# Títulos
ax1.set_title('Simulação com Precisão Padrão')
ax2.set_title('Simulação com Precisão Reduzida')

# Textos para o tempo
time_text1 = ax1.text(0.95, 1.03, '', ha='center', transform=ax1.transAxes)
time_text2 = ax2.text(0.95, 1.03, '', ha='center', transform=ax2.transAxes)

def update(frame):
    global pos1, vel1, t1, next_event1, next_event_arg1
    global pos2, vel2, t2, next_event2, next_event_arg2

    # Primeira simulação (precisão padrão)
    if dt:
        next_t1 = t1 + dt
    else:
        next_t1 = t1 + next_event
    while t1 + next_event1 <= next_t1:
        t1 += next_event1
        for k, l in singles:
            pos1[k][l] += vel1[k][l] * next_event1
        compute_new_velocities(pos1, vel1, next_event_arg1)
        next_event1, next_event_arg1 = compute_next_event(pos1, vel1)
    remain_t1 = next_t1 - t1
    for k, l in singles:
        pos1[k][l] += vel1[k][l] * remain_t1
    t1 += remain_t1
    next_event1 -= remain_t1

    # Atualiza partículas e velocidades para a primeira simulação
    for i, ((x, y), (dx, dy)) in enumerate(zip(pos1, vel1)):
        particles1[i].center = (x, y)
        dx_scaled = dx * arrow_scale
        dy_scaled = dy * arrow_scale
        velocity_lines1[i].set_data([x, x + dx_scaled], [y, y + dy_scaled])

    time_text1.set_text('t = %.2f' % t1)

    # Segunda simulação (precisão reduzida)
    if dt:
        next_t2 = t2 + dt
    else:
        next_t2 = t2 + next_event
    while t2 + next_event2 <= next_t2:
        t2 += next_event2
        for k, l in singles:
            pos2[k][l] += vel2[k][l] * next_event2
        # Arredonda posições e velocidades para simular precisão reduzida
        pos2, vel2 = round_pos_vel(pos2, vel2, precision)
        compute_new_velocities(pos2, vel2, next_event_arg2)
        # Arredonda novamente
        pos2, vel2 = round_pos_vel(pos2, vel2, precision)
        next_event2, next_event_arg2 = compute_next_event(pos2, vel2)
    remain_t2 = next_t2 - t2
    for k, l in singles:
        pos2[k][l] += vel2[k][l] * remain_t2
    t2 += remain_t2
    next_event2 -= remain_t2

    # Arredonda posições e velocidades
    pos2, vel2 = round_pos_vel(pos2, vel2, precision)

    # Atualiza partículas e velocidades para a segunda simulação
    for i, ((x, y), (dx, dy)) in enumerate(zip(pos2, vel2)):
        particles2[i].center = (x, y)
        dx_scaled = dx * arrow_scale
        dy_scaled = dy * arrow_scale
        velocity_lines2[i].set_data([x, x + dx_scaled], [y, y + dy_scaled])

    time_text2.set_text('t = %.2f' % t2)

    return particles1 + velocity_lines1 + [time_text1] + particles2 + velocity_lines2 + [time_text2]

# Cria a animação
anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

# Salva a animação como GIF
anim.save('./img/event_difference.gif', writer='pillow', fps=50)

