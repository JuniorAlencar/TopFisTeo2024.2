
import matplotlib.pyplot as plt
import random
import numpy as np

# Considerando k = kb = T = 1
k = 1.0  # Constante de força
kb = 1.0  # Constante de Boltzmann
T = 1.0  # Temperatura
beta = 1 / (kb * T) 

# Energia de Boltzmann
def E(x):
    return k * x ** 2 / 2

# Distribuição de Boltzmann (não-normalizada)
def P(x):
    return np.exp((-k * x ** 2) / kb * T)

# Distribuição proposta (uniforme) no intervalo [-a, a]
#a = 1
#a = 3
def g(a, x):
    return 1 / (2 * a) if -a <= x <= a else 0

# Rejection Sampling

# Devemos gerar um número aleatório uniforme [0, 1], se  u <= P(x')/M*g(x'), devemos aceitar x'. Onde x' é obtido de forma uniforme no intervalo [-a, a] (x_prop)
def rejection_sampling(a, n_samples):
    samples = []
    
    # Valor de M que faz com que Mg(x) >= P(x) para todos x no intervalo de interesse [-a, a]
    M = max([P(y) / g(a, y) for y in np.linspace(-a, a, 1000)])
    while len(samples) < n_samples:
        # Amostra da proposta
        x_prop = random.uniform(-a, a)
        u = random.uniform(0, 1)
        
        # Verificar aceitação
        if u <= P(x_prop) / (M * g(a, x_prop)):
            samples.append(x_prop)
    return np.array(samples)