
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
    return np.exp((-beta * k *x ** 2) / 2)

# Distribuição proposta (uniforme) no intervalo [-a, a]
#a = 1
#a = 3
# def g(a, x):
#     return 1 / (2 * a) if -a <= x <= a else 0
def g(a, x):
    return 1 / (2*a) if -a <= x <= a else 0

def P_teorica(x):
    const = ((beta * k) / (2 * np.pi))**0.5
    return const * np.exp(-(beta * k * x**2)/2)

# Rejection Sampling

# Devemos gerar um número aleatório uniforme [0, 1], se  u <= P(x')/M*g(x'), devemos aceitar x'. Onde x' é obtido de forma uniforme no intervalo [-a, a] (x_prop)
# def rejection_sampling(n_samples, a):
#     """
#     Função irá filtrar um número n_samples desejados, dentro do intervalo [-a, a], seguindo o critério proposto no rejection_samples, onde:
#     x_proposto é aceito, se, um número u gerado de forma uniforme for tal que u <= P(x_proposto)/(M * g(x_proposto)), com M sendo uma constante
#     de tal forma que Mg(x) > P(x), para que a função g(x) englobe a distribuição de interesse.
    
#     args:
#         n_samples (int): número de amostras desejados
#         a (int): intervalo simétrico de intereresse
#     """
    
    
#     samples = []
    
#     # Valor de M que faz com que Mg(x) >= P(x) para todos x no intervalo de interesse [-a, a]
#     M = max([P(y) / g(a, y) for y in np.linspace(-a, a, 1000)])
    
#     while len(samples) < n_samples:
#         # Amostra da proposta
#         x_prop = random.uniform(-a, a)
#         u = random.uniform(0, 1)
        
#         # Verificar aceitação
#         if u <= P(x_prop) / (M * g(a, x_prop)):
#             samples.append(x_prop)
    
#     return samples


def rejection_sampling(n_samples, a, M):
    """
    Função irá filtrar um número n_samples desejados, dentro do intervalo [-a, a], seguindo o critério proposto no rejection_samples, onde:
    x_proposto é aceito, se, um número u gerado de forma uniforme for tal que u <= P(x_proposto)/(M * g(x_proposto)), com M sendo uma constante
    de tal forma que Mg(x) > P(x), para que a função g(x) englobe a distribuição de interesse.
    
    args:
        n_samples (int): número de amostras desejados
        a (int): intervalo simétrico de intereresse
    """
    
    
    samples = []
    
    # Valor de M que faz com que Mg(x) >= P(x) para todos x no intervalo de interesse [-a, a]
    #M = max([P(y) / g(a, y) for y in np.linspace(-a, a, 1000) if g(a, y) != 0])
    
    while len(samples) < n_samples:
        # Amostra da proposta
        x_prop = random.uniform(-a, a)
        u = random.uniform(0, 1)
        
        # Verificar aceitação
        if u <= P(x_prop) / (M * g(a, x_prop)) and g(a, x_prop) != 0:
            samples.append(x_prop)
    
    return samples

# Função para calcular a distribuição (Pk) e os valores de k
def distribution_data(samples, a, m):
    # Construir os bins e calcular as distribuições
    """
    Calcula a distribuição normalizada a partir dos dados propostos

    args:
        samples (list): valores aceitos pelo método de rejection_samples.
        a (float): valor utilizado no intervalo simétrico que iremos analisar [-a, a].
        m (int): número de pontos utilizados na discretização no eixo x.
    
    """
    #m = 100 # Número de pontos utilizados na discretização do eixo x
    x = np.linspace(-a, a, m) 
    bins = [(float(x[i]), float(x[i+1])) for i in range(m-1)]
    count_values = [0 for i in range(m - 1)]

    # Contando valores de samples dentro dos intervalos de bins
    for sample in samples:
        for i, (start, end) in enumerate(bins):
            if start <= sample < end:  # Verifica se a amostra está no intervalo
                count_values[i] += 1   # Se o valor estiver dentro do intervalo, atualiza a contagem
                break  # Sai do loop assim que o intervalo correto for encontrado

    # Calculando largura dos bins (assumimos largura constante)
    bin_width = bins[0][1] - bins[0][0]

    # Como possuímos bins igualmente espaçados (step), devemos normalizar considerando a largura dos bins
    total_count = sum(count_values)
    Pk_data = [(count / (total_count * bin_width)) for count in count_values]  # Densidade
    k_data = [(bin[1] + bin[0]) / 2 for bin in bins]  # Centros dos bins

    # Filtrando para remover zeros
    filtered_k_data = [k for k, pk in zip(k_data, Pk_data) if pk > 0]
    filtered_Pk_data = [pk for pk in Pk_data if pk > 0]

    return filtered_k_data, filtered_Pk_data