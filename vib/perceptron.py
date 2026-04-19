import numpy as np


# ============================================================
# MODELO LINEAR DE CLASSIFICACAO
# ============================================================
# Este modulo reune apenas a parte de aprendizado:
#   1. normalizacao das features;
#   2. funcoes matematicas do modelo linear;
#   3. treinamento um-contra-todos;
#   4. previsao e metricas de avaliacao.
#
# Embora o arquivo use o nome "perceptron", a formulacao adotada aqui
# e a de um classificador linear com sigmoide e custo softplus.


def normalizar(X_tr, X_te):
    """
    Aplica normalizacao z-score com base apenas no conjunto de treino.

    Formula:
        x_norm = (x - media) / desvio

    Papel no projeto:
    - evita que features em escalas muito diferentes dominem o treino;
    - e chamada por `main.py` antes de treinar e antes de prever.

    Parametros
    ----------
    X_tr : np.ndarray
        Features do conjunto de treino.
    X_te : np.ndarray
        Features do conjunto de teste.

    Retorna
    -------
    tuple
        Dados normalizados e os parametros `media` e `desvio`.
    """
    media = X_tr.mean(axis=0)
    desvio = X_tr.std(axis=0) + 1e-12
    return (X_tr - media) / desvio, (X_te - media) / desvio, media, desvio


def sigmoid(t):
    """
    Calcula a sigmoide de forma numericamente estavel.

    A sigmoide transforma uma ativacao linear em um valor entre 0 e 1.
    Ela nao e usada diretamente para prever classes no `main.py`, mas
    e fundamental para calcular o gradiente do custo.

    Papel no projeto:
    - e chamada por `gradiente`.
    """
    return np.where(t >= 0, 1.0 / (1.0 + np.exp(-t)), np.exp(t) / (1.0 + np.exp(t)))


def softplus_estavel(t):
    """
    Calcula a funcao softplus de forma estavel.

    Formula:
        softplus(t) = log(1 + exp(t))

    Ela e usada como bloco do custo total. A versao estavel evita
    overflow para valores muito grandes de `t`.

    Papel no projeto:
    - e chamada por `custo_total`.
    """
    return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0)


def epsilon(r):
    """
    Converte o rotulo binario em um fator de sinal para o custo.

    Convencao usada:
    - r = 1  -> epsilon = -1
    - r = 0  -> epsilon = +1

    Isso permite escrever o custo binario com uma unica expressao.

    Papel no projeto:
    - e chamada por `custo_total` e `gradiente`.
    """
    return np.where(r == 1, -1.0, 1.0)


def custo_total(w0, w, X, r):
    """
    Avalia o custo total do modelo em um conjunto de amostras.

    Passos:
    1. calcula a ativacao linear P = w0 + X @ w;
    2. aplica a codificacao `epsilon(r)`;
    3. soma a softplus em todas as amostras.

    Papel no projeto:
    - mede a qualidade dos pesos atuais;
    - e usado para iniciar o historico de treino;
    - tambem e usado no backtracking dentro de `treinar_perceptron`.
    """
    P = w0 + X @ w
    eps = epsilon(r)
    return np.sum(softplus_estavel(eps * P))


def gradiente(w0, w, X, r):
    """
    Calcula o gradiente do custo em relacao ao vies e aos pesos.

    Esse gradiente aponta a direcao de maior aumento do custo. Como o
    treino faz descida de gradiente, o algoritmo anda na direcao oposta.

    Papel no projeto:
    - e chamado a cada iteracao de `treinar_perceptron`.

    Retorna
    -------
    tuple
        Gradiente do vies e gradiente do vetor de pesos.
    """
    P = w0 + X @ w
    eps = epsilon(r)
    s = sigmoid(eps * P)

    grad_w0 = np.sum(eps * s)
    grad_w = X.T @ (eps * s)
    return grad_w0, grad_w


def _criar_historico_detalhado():
    """
    Inicializa a estrutura usada para guardar o estado do treino.

    Cada lista recebe um valor por estado visitado pelo algoritmo:
    - estado inicial;
    - estado apos cada iteracao aceita.
    """
    return {
        "epocas": [],
        "custos": [],
        "w0": [],
        "w": [],
        "gradiente": [],
        "grad_w0": [],
        "grad_w": [],
        "norma_gradiente": [],
    }


def _registrar_estado_treino(historico, epoca, w0, w, X, r, custo=None):
    """
    Registra custo, pesos e gradiente de um estado especifico do treino.

    O gradiente e recalculado no proprio ponto salvo, o que deixa o
    historico consistente para analise e visualizacao.
    """
    if custo is None:
        custo = custo_total(w0, w, X, r)

    grad_w0, grad_w = gradiente(w0, w, X, r)
    norma_gradiente = np.sqrt(grad_w0**2 + np.sum(grad_w**2))

    historico["epocas"].append(int(epoca))
    historico["custos"].append(float(custo))
    historico["w0"].append(float(w0))
    historico["w"].append(np.array(w, copy=True))
    historico["gradiente"].append(np.concatenate(([grad_w0], grad_w)))
    historico["grad_w0"].append(float(grad_w0))
    historico["grad_w"].append(np.array(grad_w, copy=True))
    historico["norma_gradiente"].append(float(norma_gradiente))


def _historico_em_arrays(historico):
    """Converte todas as listas de um historico de treino para arrays numpy."""
    return {chave: np.asarray(valores) for chave, valores in historico.items()}


def treinar_perceptron(X_tr, r_tr, delta=1e-3, n_iter=150, nome=""):
    """
    Treina um classificador binario por descida de gradiente.

    Fluxo do treino:
    1. inicializa pesos zerados;
    2. calcula o gradiente do custo;
    3. propoe um novo passo nos pesos;
    4. verifica se o custo caiu;
    5. se nao caiu, reduz `delta` ate encontrar um passo valido.

    O nome "perceptron" foi mantido por consistencia com o projeto,
    mas a implementacao usa custo suave e gradiente.

    Papel no projeto:
    - e chamada por `treinar_multiclasse` uma vez para cada classe.

    Parametros
    ----------
    X_tr : np.ndarray
        Features normalizadas do treino.
    r_tr : np.ndarray
        Rotulos binarios da classe atual.
    delta : float
        Passo inicial da descida de gradiente.
    n_iter : int
        Numero maximo de iteracoes.
    nome : str
        Nome da classe, usado apenas na impressao do log.

    Retorna
    -------
    tuple
        Vies treinado, vetor de pesos, historico simples do custo e
        historico detalhado com pesos, gradientes e norma por epoca.
    """
    _, n_features = X_tr.shape
    w0 = 0.0
    w = np.zeros(n_features)

    # O historico permite acompanhar se a otimizacao esta melhorando.
    hist = [custo_total(w0, w, X_tr, r_tr)]
    historico_detalhado = _criar_historico_detalhado()
    _registrar_estado_treino(
        historico_detalhado, epoca=0, w0=w0, w=w, X=X_tr, r=r_tr, custo=hist[0]
    )

    for epoca in range(1, n_iter + 1):
        gw0, gw = gradiente(w0, w, X_tr, r_tr)

        # Propoe um novo conjunto de pesos.
        w0_new = w0 - delta * gw0
        w_new = w - delta * gw
        c_new = custo_total(w0_new, w_new, X_tr, r_tr)

        # Se o custo piorar, o passo e reduzido ate ficar aceitavel.
        while c_new > hist[-1] and delta > 1e-14:
            delta *= 0.5
            w0_new = w0 - delta * gw0
            w_new = w - delta * gw
            c_new = custo_total(w0_new, w_new, X_tr, r_tr)

        w0, w = w0_new, w_new
        hist.append(c_new)
        _registrar_estado_treino(
            historico_detalhado, epoca=epoca, w0=w0, w=w, X=X_tr, r=r_tr, custo=c_new
        )

    print(f"  [{nome:<18}] Custo inicial: {hist[0]:>10.1f}  -> final: {hist[-1]:>8.1f}")
    return w0, w, hist, _historico_em_arrays(historico_detalhado)


def treinar_multiclasse(X_tr, y_tr, nomes=None, delta=1e-3, n_iter=150):
    """
    Treina um modelo binario para cada classe no esquema um-contra-todos.

    Exemplo:
    - para a classe 0, as amostras da classe 0 viram alvo e as demais nao;
    - depois o processo se repete para a classe 1, classe 2 etc.

    Papel no projeto:
    - e a funcao de treino chamada por `main.py`;
    - organiza os varios classificadores binarios em um unico pacote.

    Retorna
    -------
    tuple
        Lista de modelos, lista dos historicos simples de custo e lista
        dos historicos detalhados de cada classificador.
    """
    modelos = []
    historicos = []
    historicos_detalhados = []
    n_classes = int(np.max(y_tr)) + 1

    print("\n=== TREINAMENTO - UM-CONTRA-TODOS ===")

    for c in range(n_classes):
        nome = nomes[c] if nomes is not None else f"Classe {c}"

        # Cria o rotulo binario para a classe atual.
        r_bin = (y_tr == c).astype(int)

        w0, w, hist, hist_det = treinar_perceptron(
            X_tr, r_bin, delta=delta, n_iter=n_iter, nome=nome,
        )
        modelos.append((w0, w))
        historicos.append(hist)
        historicos_detalhados.append(hist_det)

    return modelos, historicos, historicos_detalhados


def prever(X, modelos):
    """
    Gera a classe prevista para cada amostra.

    Cada modelo binario produz uma ativacao linear. A classe final e a
    que tiver a maior ativacao dentre todas.

    Papel no projeto:
    - e chamada por `main.py` para prever treino ou teste.
    """
    ativacoes = np.array([w0 + X @ w for w0, w in modelos])
    return np.argmax(ativacoes, axis=0)


def matriz_confusao(y_real, y_pred, n=None):
    """
    Constroi a matriz de confusao do problema multiclasse.

    Convencao:
    - linha   -> classe real
    - coluna  -> classe prevista

    Papel no projeto:
    - e usada por `plotar_resultados` em `main.py`.
    """
    if n is None:
        n = int(max(np.max(y_real), np.max(y_pred))) + 1

    matriz = np.zeros((n, n), dtype=int)
    for real, pred in zip(y_real, y_pred):
        matriz[real, pred] += 1
    return matriz


def metricas_por_classe(y_real, y_pred, n=None):
    """
    Calcula metricas basicas para cada classe.

    Para cada classe sao calculados:
    - VP: verdadeiro positivo
    - FN: falso negativo
    - VN: verdadeiro negativo
    - FP: falso positivo
    - P(acerto | alvo)
    - P(acerto | nao-alvo)

    Papel no projeto:
    - e chamada por `main.py` para montar a tabela final impressa.
    """
    if n is None:
        n = int(max(np.max(y_real), np.max(y_pred))) + 1

    resultados = []
    for c in range(n):
        alvo = y_real == c
        nao_alvo = ~alvo
        vp = int(np.sum((y_pred == c) & alvo))
        fn = int(np.sum((y_pred != c) & alvo))
        vn = int(np.sum((y_pred != c) & nao_alvo))
        fp = int(np.sum((y_pred == c) & nao_alvo))
        p_alvo = 100 * vp / (vp + fn) if (vp + fn) > 0 else 0.0
        p_nao_alvo = 100 * vn / (vn + fp) if (vn + fp) > 0 else 0.0
        resultados.append((vp, fn, vn, fp, p_alvo, p_nao_alvo))
    return resultados


__all__ = [
    "custo_total",
    "gradiente",
    "matriz_confusao",
    "metricas_por_classe",
    "normalizar",
    "prever",
    "sigmoid",
    "softplus_estavel",
    "treinar_multiclasse",
    "treinar_perceptron",
]
