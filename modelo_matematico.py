import numpy as np


# ============================================================
# MODELO MATEMATICO DA VIGA
# ============================================================
# Este modulo concentra apenas a parte fisica e de pre-processamento
# do problema:
#   1. define os materiais e os parametros geometricos da viga;
#   2. calcula a frequencia natural do primeiro modo;
#   3. simula o sinal de vibracao livre amortecida;
#   4. extrai as features que depois alimentam o classificador.
#
# Em outras palavras: daqui saem os dados de entrada do perceptron.


# Cada material tem:
#   E     -> modulo de elasticidade [Pa]
#   rho   -> densidade [kg/m^3]
#   zeta  -> razao de amortecimento
#   label -> classe numerica usada no treinamento
MATERIAIS = {
    "A\u00e7o": {"E": 200e9, "rho": 7850, "zeta": 0.010, "label": 0},
    "Alum\u00ednio": {"E": 70e9, "rho": 2700, "zeta": 0.015, "label": 1},
    "Tit\u00e2nio": {"E": 110e9, "rho": 4500, "zeta": 0.008, "label": 2},
    "Fibra de Carbono": {"E": 150e9, "rho": 1600, "zeta": 0.020, "label": 3},
}


# Parametros geometricos fixos da viga engastada.
L = 1.00
b = 0.050
h = 0.050
A_sec = b * h
I_sec = b * h**3 / 12


# Constante modal do primeiro modo de uma viga engastada-livre.
BETA_MODO_1 = 1.875104068711961


# Atalhos usados pelo restante do projeto.
N_CLASSES = len(MATERIAIS)
NOMES = list(MATERIAIS.keys())


def frequencia_natural(E, rho):
    """
    Calcula a frequencia angular natural do primeiro modo de flexao.

    Formula usada:
        omega_1 = beta_1^2 * sqrt(EI / (rho * A * L^4))

    Papel no projeto:
    - traduz as propriedades do material em uma frequencia propria;
    - e chamada por `simular_vibracao`;
    - tambem e usada em `main.py` para rotular os graficos de exemplo.

    Parametros
    ----------
    E : float
        Modulo de elasticidade do material.
    rho : float
        Densidade do material.

    Retorna
    -------
    float
        Frequencia angular natural em rad/s.
    """
    rigidez_flexao = E * I_sec
    massa_linear = rho * A_sec
    return (BETA_MODO_1**2) * np.sqrt(rigidez_flexao / (massa_linear * L**4))


def resposta_modal_livre(omega_n, zeta, t, q0=1.0, dq0=0.0):
    """
    Gera a resposta amortecida de um sistema de um grau de liberdade.

    Aqui usamos a solucao analitica de vibracao livre subamortecida:
    o sinal oscila com frequencia amortecida e sua amplitude decai no
    tempo por causa do termo exponencial.

    Papel no projeto:
    - e o nucleo temporal da simulacao;
    - recebe uma frequencia natural ja calculada;
    - e chamada por `simular_vibracao`.

    Parametros
    ----------
    omega_n : float
        Frequencia angular natural.
    zeta : float
        Razao de amortecimento.
    t : array
        Vetor de tempo.
    q0 : float
        Deslocamento inicial.
    dq0 : float
        Velocidade inicial.

    Retorna
    -------
    np.ndarray
        Sinal temporal da resposta amortecida.
    """
    zeta_eff = np.clip(zeta, 0.0, 0.999999)
    omega_d = omega_n * np.sqrt(max(1.0 - zeta_eff**2, 1e-12))
    envelope = np.exp(-zeta_eff * omega_n * t)
    termo_seno = (dq0 + zeta_eff * omega_n * q0) / omega_d
    return envelope * (q0 * np.cos(omega_d * t) + termo_seno * np.sin(omega_d * t))


def simular_vibracao(E, rho, zeta, t, A0=1.0):
    """
    Simula o deslocamento da viga usando apenas o primeiro modo.

    Esta e a funcao principal da parte fisica. Ela:
    1. calcula a frequencia natural do material;
    2. usa essa frequencia para montar o sinal amortecido no tempo.

    Papel no projeto:
    - e chamada por `gerar_dataset` para criar os sinais sinteticos;
    - e chamada por `plotar_sinais_exemplo` em `main.py`.

    Parametros
    ----------
    E, rho, zeta : float
        Propriedades fisicas do material.
    t : array
        Vetor de tempo.
    A0 : float
        Amplitude inicial da resposta.

    Retorna
    -------
    np.ndarray
        Sinal de vibracao livre amortecida.
    """
    omega_n = frequencia_natural(E, rho)
    return resposta_modal_livre(omega_n, zeta, t, q0=A0, dq0=0.0)


def extrair_features(sinal, t):
    """
    Extrai as features que o perceptron vai usar como entrada.

    Features produzidas:
    - RMS: energia media do sinal;
    - pico: maior valor absoluto;
    - fator de crista: pico / RMS;
    - frequencia dominante: pico principal do espectro;
    - taxa de decaimento: compara energia da primeira e segunda metade;
    - energia espectral: intensidade maxima no dominio da frequencia.

    Papel no projeto:
    - converte um sinal bruto em um vetor numerico compacto;
    - e chamada por `gerar_dataset` logo apos a simulacao.

    Parametros
    ----------
    sinal : array
        Serie temporal simulada.
    t : array
        Vetor de tempo associado ao sinal.

    Retorna
    -------
    np.ndarray
        Vetor com 6 features.
    """
    dt = t[1] - t[0]
    n_amostras = len(sinal)

    # RMS resume o nivel medio de energia do sinal.
    rms = np.sqrt(np.mean(sinal**2))

    # Pico mede a maior excursao absoluta.
    pico = np.max(np.abs(sinal))

    # Fator de crista relaciona pico e energia media.
    fator_crista = pico / (rms + 1e-12)

    # O espectro e usado para identificar a frequencia dominante.
    espectro = np.abs(np.fft.rfft(sinal))
    freqs = np.fft.rfftfreq(n_amostras, d=dt)
    freq_dom = freqs[np.argmax(espectro)]

    # A razao entre comeco e fim ajuda a capturar o amortecimento.
    meio = n_amostras // 2
    rms_ini = np.sqrt(np.mean(sinal[:meio] ** 2)) + 1e-12
    rms_fim = np.sqrt(np.mean(sinal[meio:] ** 2)) + 1e-12
    taxa_decaimento = rms_ini / rms_fim

    # Energia espectral maxima reforca a assinatura no dominio da frequencia.
    energia_espectral = np.max(espectro) ** 2 / n_amostras

    return np.array(
        [rms, pico, fator_crista, freq_dom, taxa_decaimento, energia_espectral]
    )


def gerar_dataset(n_por_classe=250, seed=42):
    """
    Gera o dataset sintetico completo a partir do modelo fisico.

    Fluxo interno:
    1. percorre cada material;
    2. perturba levemente E, rho, zeta e A0 para gerar variabilidade;
    3. simula o sinal temporal;
    4. extrai as features;
    5. monta X e y.

    Papel no projeto:
    - e a porta de entrada dos dados para o treinamento;
    - e chamada diretamente por `main.py`.

    Parametros
    ----------
    n_por_classe : int
        Numero de amostras sinteticas por material.
    seed : int
        Semente aleatoria para reprodutibilidade.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray]
        X com shape (n_total, 6) e y com os rotulos das classes.
    """
    np.random.seed(seed)
    t = np.linspace(0, 1.0, 1000)

    X_lista = []
    y_lista = []

    for props in MATERIAIS.values():
        for _ in range(n_por_classe):
            # Pequenas variacoes deixam o dataset menos artificial.
            E_v = props["E"] * np.random.uniform(0.95, 1.05)
            rho_v = props["rho"] * np.random.uniform(0.95, 1.05)
            zeta_v = props["zeta"] * np.random.uniform(0.90, 1.10)
            A0_v = np.random.uniform(0.8, 1.2)

            sinal = simular_vibracao(E_v, rho_v, zeta_v, t, A0=A0_v)
            features = extrair_features(sinal, t)

            X_lista.append(features)
            y_lista.append(props["label"])

    return np.array(X_lista), np.array(y_lista)


__all__ = [
    "MATERIAIS",
    "N_CLASSES",
    "NOMES",
    "extrair_features",
    "frequencia_natural",
    "gerar_dataset",
    "simular_vibracao",
]
