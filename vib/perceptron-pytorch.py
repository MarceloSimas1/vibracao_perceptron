import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# MODELO DE CLASSIFICACAO COM PYTORCH
# ============================================================
# Este modulo deixa o perceptron organizado como uma rede PyTorch.
# Hoje a arquitetura padrao e linear, equivalente ao perceptron usado no
# projeto, mas a classe ja aceita camadas ocultas para evoluir para uma
# rede neural sem trocar o fluxo de treino, previsao e metricas.


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Converte entradas numericas para tensor PyTorch no device padrao.
def _to_tensor(x, dtype=DTYPE, device=DEVICE):
    """
    Converte uma entrada numerica para tensor PyTorch.

    Papel no projeto:
    - padroniza tipo e device;
    - permite receber listas, tuplas ou tensores sem depender de NumPy.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=dtype, device=device)


# Converte rotulos para tensores inteiros do PyTorch.
def _to_long_tensor(x, device=DEVICE):
    """
    Converte rotulos para torch.long.

    Papel no projeto:
    - prepara classes para comparacoes, argmax e bincount;
    - mantem metricas e treino dentro do PyTorch.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(dtype=torch.long, device=device)
    return torch.as_tensor(x, dtype=torch.long, device=device)


# Escolhe a funcao de ativacao das camadas ocultas.
def _criar_ativacao(nome):
    """
    Cria uma ativacao PyTorch pelo nome.

    Papel no projeto:
    - centraliza a escolha da nao linearidade;
    - facilita trocar o perceptron linear por uma rede com camadas ocultas.
    """
    ativacoes = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
    }
    if nome not in ativacoes:
        raise ValueError(f"Ativacao desconhecida: {nome}")
    return ativacoes[nome]()


# Modelo binario flexivel: sem camadas ocultas vira perceptron linear.
class RedeBinaria(nn.Module):
    """
    Rede de classificacao binaria com saida logit.

    Papel no projeto:
    - representa cada classificador um-contra-todos;
    - permite evoluir de perceptron linear para MLP mudando `camadas_ocultas`.
    """

    # Monta a arquitetura linear ou multicamadas.
    def __init__(self, n_features, camadas_ocultas=None, ativacao="relu"):
        """
        Inicializa a rede binaria.

        Parametros
        ----------
        n_features : int
            Numero de features de entrada.
        camadas_ocultas : list[int] | None
            Tamanhos das camadas ocultas. `None` mantem o perceptron linear.
        ativacao : str
            Ativacao usada entre camadas ocultas.
        """
        super().__init__()
        camadas_ocultas = camadas_ocultas or []

        camadas = []
        entrada = n_features
        for saida in camadas_ocultas:
            camadas.append(nn.Linear(entrada, saida))
            camadas.append(_criar_ativacao(ativacao))
            entrada = saida
        camadas.append(nn.Linear(entrada, 1))

        self.rede = nn.Sequential(*camadas)
        self.double()
        self.reset_parameters()

    # Inicializa o perceptron zerado e redes futuras com inicializacao propria.
    def reset_parameters(self):
        """
        Reinicializa os parametros da rede.

        Papel no projeto:
        - usa pesos e vieses zerados no caso linear, como no codigo original;
        - usa inicializacao Kaiming quando houver camadas ocultas.
        """
        lineares = [modulo for modulo in self.rede if isinstance(modulo, nn.Linear)]

        if len(lineares) == 1:
            nn.init.zeros_(lineares[0].weight)
            nn.init.zeros_(lineares[0].bias)
            return

        for modulo in lineares:
            if isinstance(modulo, nn.Linear):
                nn.init.kaiming_uniform_(modulo.weight, nonlinearity="relu")
                nn.init.zeros_(modulo.bias)

    # Executa a rede e retorna logits, sem aplicar sigmoide.
    def forward(self, X):
        """
        Calcula a ativacao da rede.

        Papel no projeto:
        - fornece o logit usado pelo custo softplus;
        - fornece a ativacao usada para escolher a classe prevista.
        """
        X_t = _to_tensor(X, device=next(self.parameters()).device)
        return self.rede(X_t).squeeze(-1)

    # Extrai os parametros lineares quando a rede nao tem camada oculta.
    def parametros_lineares(self):
        """
        Retorna vies e pesos do modelo linear.

        Papel no projeto:
        - ajuda a inspecionar o perceptron treinado;
        - levanta erro se a arquitetura ja tiver virado uma rede multicamadas.
        """
        lineares = [m for m in self.rede if isinstance(m, nn.Linear)]
        if len(lineares) != 1:
            raise RuntimeError("parametros_lineares so vale para rede sem camadas ocultas.")
        camada = lineares[0]
        return camada.bias.detach().clone(), camada.weight.detach().clone().squeeze(0)


# Normaliza treino e teste usando media/desvio calculados no treino.
def normalizar(X_tr, X_te):
    """
    Aplica normalizacao z-score com base apenas no conjunto de treino.

    Papel no projeto:
    - evita que features em escalas muito diferentes dominem o treino;
    - retorna tensores PyTorch normalizados e estatisticas do treino.
    """
    X_tr_t = _to_tensor(X_tr)
    X_te_t = _to_tensor(X_te)
    media = torch.mean(X_tr_t, dim=0)
    desvio = torch.std(X_tr_t, dim=0, unbiased=False) + 1e-12
    return (X_tr_t - media) / desvio, (X_te_t - media) / desvio, media, desvio


# Calcula a funcao sigmoide com a implementacao nativa do PyTorch.
def sigmoid(t):
    """
    Calcula a sigmoide.

    Papel no projeto:
    - transforma logits em valores entre 0 e 1;
    - pode ser usada para interpretar a confianca de um modelo binario.
    """
    return torch.sigmoid(_to_tensor(t))


# Calcula softplus usando a funcao estavel do PyTorch.
def softplus_estavel(t):
    """
    Calcula softplus de forma numericamente estavel.

    Papel no projeto:
    - implementa log(1 + exp(t)) sem overflow;
    - e usada como bloco do custo logistico.
    """
    return F.softplus(_to_tensor(t))


# Converte o rotulo binario para o sinal usado pelo custo softplus.
def epsilon(r):
    """
    Converte rotulos binarios para fator de sinal.

    Convencao:
    - r = 1  -> epsilon = -1
    - r = 0  -> epsilon = +1
    """
    r_t = _to_tensor(r)
    return torch.where(r_t == 1, -torch.ones_like(r_t), torch.ones_like(r_t))


# Calcula o custo total de uma rede binaria em um conjunto de amostras.
def custo_total(modelo, X, r):
    """
    Avalia o custo total do modelo.

    Papel no projeto:
    - mede a qualidade atual da rede;
    - preserva o grafo do autograd para treino e analise de gradientes.
    """
    r_t = _to_tensor(r, device=next(modelo.parameters()).device)
    logits = modelo(X)
    return torch.sum(F.softplus(epsilon(r_t) * logits))


# Calcula os gradientes dos parametros da rede por autograd.
def gradiente(modelo, X, r):
    """
    Calcula o gradiente do custo em relacao aos parametros da rede.

    Papel no projeto:
    - usa diferenciacao automatica do PyTorch;
    - funciona tanto para perceptron linear quanto para redes futuras.
    """
    modelo.zero_grad(set_to_none=True)
    custo = custo_total(modelo, X, r)
    custo.backward()
    return {
        nome: parametro.grad.detach().clone()
        for nome, parametro in modelo.named_parameters()
        if parametro.grad is not None
    }


# Inicializa a estrutura que guarda a evolucao do treino.
def _criar_historico_detalhado():
    """
    Inicializa o historico detalhado do treinamento.

    Papel no projeto:
    - armazena epocas, custos e parametros;
    - suporta redes futuras salvando `state_dict` em cada epoca.
    """
    return {
        "epocas": [],
        "custos": [],
        "w0": [],
        "w": [],
        "gradiente": [],
        "grad_w0": [],
        "grad_w": [],
        "parametros": [],
        "gradientes": [],
        "norma_gradiente": [],
    }


# Calcula a norma global dos gradientes da rede.
def _norma_gradiente(gradientes):
    """
    Calcula a norma euclidiana global dos gradientes.

    Papel no projeto:
    - resume a intensidade da atualizacao;
    - funciona para qualquer quantidade de camadas.
    """
    if not gradientes:
        return torch.zeros((), dtype=DTYPE, device=DEVICE)
    soma = torch.zeros((), dtype=DTYPE, device=next(iter(gradientes.values())).device)
    for grad in gradientes.values():
        soma = soma + torch.sum(grad**2)
    return torch.sqrt(soma)


# Salva custo, parametros e gradientes de uma epoca.
def _registrar_estado_treino(historico, epoca, modelo, X, r, custo=None):
    """
    Registra o estado atual do treino.

    Papel no projeto:
    - preserva uma copia dos parametros da rede;
    - recalcula gradientes no ponto salvo para manter o historico consistente.
    """
    if custo is None:
        custo = custo_total(modelo, X, r)

    gradientes = gradiente(modelo, X, r)
    parametros = {
        nome: valor.detach().clone()
        for nome, valor in modelo.state_dict().items()
    }

    try:
        w0, w = modelo.parametros_lineares()
        grad_w0 = gradientes["rede.0.bias"].reshape(())
        grad_w = gradientes["rede.0.weight"].reshape(-1)
        gradiente_linear = torch.cat((grad_w0.reshape(1), grad_w))
    except RuntimeError:
        w0 = torch.tensor(float("nan"), dtype=DTYPE)
        w = torch.empty(0, dtype=DTYPE)
        grad_w0 = torch.tensor(float("nan"), dtype=DTYPE)
        grad_w = torch.empty(0, dtype=DTYPE)
        gradiente_linear = torch.empty(0, dtype=DTYPE)

    historico["epocas"].append(int(epoca))
    historico["custos"].append(float(custo.detach().item()))
    historico["w0"].append(w0.reshape(()).detach().cpu())
    historico["w"].append(w.detach().cpu())
    historico["gradiente"].append(gradiente_linear.detach().cpu())
    historico["grad_w0"].append(grad_w0.detach().cpu())
    historico["grad_w"].append(grad_w.detach().cpu())
    historico["parametros"].append(parametros)
    historico["gradientes"].append(gradientes)
    historico["norma_gradiente"].append(float(_norma_gradiente(gradientes).item()))


# Converte listas simples do historico para tensores quando possivel.
def _historico_em_tensores(historico):
    """
    Converte campos numericos do historico para tensores PyTorch.

    Papel no projeto:
    - deixa epocas, custos e norma do gradiente prontos para plotagem;
    - mantem parametros e gradientes como listas de dicionarios de tensores.
    """
    return {
        "epocas": torch.tensor(historico["epocas"], dtype=torch.long),
        "custos": torch.tensor(historico["custos"], dtype=DTYPE),
        "w0": torch.stack(historico["w0"]).to(dtype=DTYPE),
        "w": torch.stack(historico["w"]).to(dtype=DTYPE),
        "gradiente": torch.stack(historico["gradiente"]).to(dtype=DTYPE),
        "grad_w0": torch.stack(historico["grad_w0"]).to(dtype=DTYPE),
        "grad_w": torch.stack(historico["grad_w"]).to(dtype=DTYPE),
        "parametros": historico["parametros"],
        "gradientes": historico["gradientes"],
        "norma_gradiente": torch.tensor(
            historico["norma_gradiente"], dtype=DTYPE
        ),
    }


# Treina uma rede binaria por descida de gradiente com backtracking.
def treinar_perceptron(
    X_tr,
    r_tr,
    delta=1e-3,
    n_iter=150,
    nome="",
    camadas_ocultas=None,
    ativacao="relu",
):
    """
    Treina um classificador binario.

    Papel no projeto:
    - com `camadas_ocultas=None`, treina o perceptron linear;
    - com `camadas_ocultas=[...]`, treina uma rede neural binaria.
    """
    X_tr_t = _to_tensor(X_tr)
    r_tr_t = _to_tensor(r_tr)
    _, n_features = X_tr_t.shape
    modelo = RedeBinaria(n_features, camadas_ocultas=camadas_ocultas, ativacao=ativacao)
    modelo.to(device=DEVICE, dtype=DTYPE)

    hist = [float(custo_total(modelo, X_tr_t, r_tr_t).detach().item())]
    historico_detalhado = _criar_historico_detalhado()
    _registrar_estado_treino(
        historico_detalhado, epoca=0, modelo=modelo, X=X_tr_t, r=r_tr_t
    )

    for epoca in range(1, n_iter + 1):
        custo = custo_total(modelo, X_tr_t, r_tr_t)
        modelo.zero_grad(set_to_none=True)
        custo.backward()

        parametros_atuais = {
            nome_param: parametro.detach().clone()
            for nome_param, parametro in modelo.named_parameters()
        }
        gradientes = {
            nome_param: parametro.grad.detach().clone()
            for nome_param, parametro in modelo.named_parameters()
        }

        with torch.no_grad():
            for nome_param, parametro in modelo.named_parameters():
                parametro.copy_(parametros_atuais[nome_param] - delta * gradientes[nome_param])

        c_new = float(custo_total(modelo, X_tr_t, r_tr_t).detach().item())

        while c_new > hist[-1] and delta > 1e-14:
            delta *= 0.5
            with torch.no_grad():
                for nome_param, parametro in modelo.named_parameters():
                    parametro.copy_(
                        parametros_atuais[nome_param] - delta * gradientes[nome_param]
                    )
            c_new = float(custo_total(modelo, X_tr_t, r_tr_t).detach().item())

        hist.append(c_new)
        _registrar_estado_treino(
            historico_detalhado,
            epoca=epoca,
            modelo=modelo,
            X=X_tr_t,
            r=r_tr_t,
            custo=torch.tensor(c_new, dtype=DTYPE, device=DEVICE),
        )

    print(f"  [{nome:<18}] Custo inicial: {hist[0]:>10.1f}  -> final: {hist[-1]:>8.1f}")
    return modelo, hist, _historico_em_tensores(historico_detalhado)


# Treina um modelo binario para cada classe no esquema um-contra-todos.
def treinar_multiclasse(
    X_tr,
    y_tr,
    nomes=None,
    delta=1e-3,
    n_iter=150,
    camadas_ocultas=None,
    ativacao="relu",
):
    """
    Treina classificadores binarios um-contra-todos.

    Papel no projeto:
    - organiza uma lista de redes binaria, uma por classe;
    - permite trocar perceptron por MLP usando os mesmos parametros de treino.
    """
    modelos = []
    historicos = []
    historicos_detalhados = []
    y_tr_t = _to_long_tensor(y_tr)
    n_classes = int(torch.max(y_tr_t).item()) + 1

    print("\n=== TREINAMENTO - UM-CONTRA-TODOS ===")

    for c in range(n_classes):
        nome = nomes[c] if nomes is not None else f"Classe {c}"
        r_bin = (y_tr_t == c).to(dtype=DTYPE)

        modelo, hist, hist_det = treinar_perceptron(
            X_tr,
            r_bin,
            delta=delta,
            n_iter=n_iter,
            nome=nome,
            camadas_ocultas=camadas_ocultas,
            ativacao=ativacao,
        )
        modelos.append(modelo)
        historicos.append(hist)
        historicos_detalhados.append(hist_det)

    return modelos, historicos, historicos_detalhados


# Escolhe a classe com maior logit entre as redes treinadas.
def prever(X, modelos):
    """
    Gera a classe prevista para cada amostra.

    Papel no projeto:
    - calcula uma ativacao por classe;
    - retorna o indice da rede com maior ativacao.
    """
    X_t = _to_tensor(X)
    with torch.no_grad():
        ativacoes = torch.stack([modelo(X_t) for modelo in modelos])
        return torch.argmax(ativacoes, dim=0)


# Monta a matriz de confusao com operacoes PyTorch.
def matriz_confusao(y_real, y_pred, n=None):
    """
    Constroi a matriz de confusao do problema multiclasse.

    Convencao:
    - linha   -> classe real
    - coluna  -> classe prevista
    """
    y_real_t = _to_long_tensor(y_real)
    y_pred_t = _to_long_tensor(y_pred)

    if n is None:
        n = int(torch.maximum(torch.max(y_real_t), torch.max(y_pred_t)).item()) + 1

    indices = y_real_t * n + y_pred_t
    return torch.bincount(indices, minlength=n * n).reshape(n, n)


# Calcula metricas por classe usando tensores PyTorch.
def metricas_por_classe(y_real, y_pred, n=None):
    """
    Calcula metricas basicas para cada classe.

    Para cada classe sao calculados:
    - VP, FN, VN e FP;
    - P(acerto | alvo);
    - P(acerto | nao-alvo).
    """
    y_real_t = _to_long_tensor(y_real)
    y_pred_t = _to_long_tensor(y_pred)

    if n is None:
        n = int(torch.maximum(torch.max(y_real_t), torch.max(y_pred_t)).item()) + 1

    resultados = []
    for c in range(n):
        alvo = y_real_t == c
        nao_alvo = ~alvo
        pred_c = y_pred_t == c

        vp = int(torch.sum(pred_c & alvo).item())
        fn = int(torch.sum((~pred_c) & alvo).item())
        vn = int(torch.sum((~pred_c) & nao_alvo).item())
        fp = int(torch.sum(pred_c & nao_alvo).item())

        p_alvo = 100 * vp / (vp + fn) if (vp + fn) > 0 else 0.0
        p_nao_alvo = 100 * vn / (vn + fp) if (vn + fp) > 0 else 0.0
        resultados.append((vp, fn, vn, fp, p_alvo, p_nao_alvo))

    return resultados


__all__ = [
    "RedeBinaria",
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
