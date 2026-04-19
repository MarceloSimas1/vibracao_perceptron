import matplotlib.pyplot as plt
import numpy as np


def salvar_ou_mostrar(fig, caminho_arquivo, mostrar_graficos=False):
    """
    Salva a figura e, opcionalmente, abre a janela interativa.

    Por padrao o script roda de forma nao bloqueante, apenas gerando
    os arquivos PNG. A exibicao interativa pode ser habilitada via CLI.
    """
    if caminho_arquivo is not None:
        fig.savefig(caminho_arquivo, dpi=150, bbox_inches="tight")
        print(f"  Grafico salvo em '{caminho_arquivo}'")

    if mostrar_graficos:
        plt.show()
    elif caminho_arquivo is not None:
        plt.close(fig)


def plotar_metricas(
    historicos,
    nomes=None,
    titulo="Metricas de treinamento",
    caminho_arquivo=None,
    mostrar_graficos=False,
):
    """
    Plota custo e norma do gradiente ao longo das epocas.

    Aceita um unico historico detalhado ou uma lista deles.
    """
    historicos_lista = _como_lista_de_historicos(historicos)
    rotulos = _resolver_rotulos(historicos_lista, nomes)

    fig, (ax_custo, ax_grad) = plt.subplots(1, 2, figsize=(13, 5))

    for historico, rotulo in zip(historicos_lista, rotulos):
        epocas = np.asarray(historico["epocas"])
        custos = np.asarray(historico["custos"])
        norma_gradiente = np.asarray(historico["norma_gradiente"])

        ax_custo.plot(epocas, custos, linewidth=2, label=rotulo)
        ax_grad.plot(epocas, norma_gradiente, linewidth=2, label=rotulo)

    ax_custo.set_title("Evolucao do custo")
    ax_custo.set_xlabel("Epoca")
    ax_custo.set_ylabel("Custo")
    ax_custo.grid(True, alpha=0.3)

    ax_grad.set_title("Norma do gradiente")
    ax_grad.set_xlabel("Epoca")
    ax_grad.set_ylabel("||grad C||")
    ax_grad.grid(True, alpha=0.3)

    if len(historicos_lista) > 1:
        ax_custo.legend()
        ax_grad.legend()

    fig.suptitle(titulo, fontsize=12, fontweight="bold")
    fig.tight_layout()
    salvar_ou_mostrar(fig, caminho_arquivo, mostrar_graficos=mostrar_graficos)
    return fig, (ax_custo, ax_grad)


def plotar_componentes_gradiente(
    historico,
    nomes_features=None,
    incluir_vies=True,
    titulo="Componentes do gradiente",
    caminho_arquivo=None,
    mostrar_graficos=False,
):
    """
    Plota cada componente do gradiente ao longo das epocas.

    Esse grafico complementa a norma do gradiente, permitindo observar
    sinal, magnitude e ritmo de convergencia de cada derivada parcial.
    """
    epocas = np.asarray(historico["epocas"])
    grad_w = np.asarray(historico["grad_w"])

    if grad_w.ndim != 2:
        raise ValueError("O historico de gradientes precisa ter shape (epocas, n_features).")

    nomes_componentes = _resolver_nomes_features(grad_w.shape[1], nomes_features)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if incluir_vies:
        grad_w0 = np.asarray(historico["grad_w0"])
        ax.plot(epocas, grad_w0, linewidth=2.2, label="grad w0")

    for indice, nome in enumerate(nomes_componentes):
        ax.plot(epocas, grad_w[:, indice], linewidth=1.8, label=f"grad {nome}")

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    ax.set_title(titulo)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Valor da derivada")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2 if len(nomes_componentes) <= 5 else 3, fontsize=9)

    fig.tight_layout()
    salvar_ou_mostrar(fig, caminho_arquivo, mostrar_graficos=mostrar_graficos)
    return fig, ax


def plotar_gradiente_3d(
    historico,
    nomes_features=None,
    incluir_vies=True,
    titulo="Gradiente 3D",
    caminho_arquivo=None,
    mostrar_graficos=False,
    max_epocas=320,
):
    """
    Plota uma superficie 3D do gradiente ao longo das epocas.

    Eixos do grafico:
    - x: epoca do treinamento;
    - y: componente do gradiente;
    - z: valor da derivada parcial.

    Para evitar figuras pesadas demais, o eixo das epocas pode ser
    reduzido por amostragem uniforme quando o historico e muito longo.
    """
    epocas = np.asarray(historico["epocas"])
    grad_w = np.asarray(historico["grad_w"])

    if grad_w.ndim != 2:
        raise ValueError("O historico de gradientes precisa ter shape (epocas, n_features).")

    nomes_componentes = _resolver_nomes_features(grad_w.shape[1], nomes_features)
    rotulos = [f"grad {nome}" for nome in nomes_componentes]
    gradientes = grad_w.T

    if incluir_vies:
        grad_w0 = np.asarray(historico["grad_w0"])
        gradientes = np.vstack((grad_w0[None, :], gradientes))
        rotulos = ["grad w0"] + rotulos

    epocas_plot, gradientes_plot = _reduzir_epocas_plot_3d(
        epocas,
        gradientes,
        max_epocas=max_epocas,
    )

    eixo_epocas, eixo_componentes = np.meshgrid(
        epocas_plot,
        np.arange(len(rotulos), dtype=float),
    )

    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    superficie = ax.plot_surface(
        eixo_epocas,
        eixo_componentes,
        gradientes_plot,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )

    for indice in range(len(rotulos)):
        ax.plot(
            epocas_plot,
            np.full(epocas_plot.shape, indice, dtype=float),
            gradientes_plot[indice],
            color="black",
            linewidth=0.7,
            alpha=0.35,
        )

    ax.set_title(titulo)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Componente")
    ax.set_zlabel("Valor da derivada")
    ax.set_yticks(np.arange(len(rotulos), dtype=float))
    ax.set_yticklabels(rotulos)
    ax.view_init(elev=28, azim=-128)

    barra = fig.colorbar(superficie, ax=ax, shrink=0.78, pad=0.08)
    barra.set_label("Gradiente")

    fig.subplots_adjust(left=0.03, right=0.86, bottom=0.06, top=0.92)
    salvar_ou_mostrar(fig, caminho_arquivo, mostrar_graficos=mostrar_graficos)
    return fig, ax


def plotar_evolucao_pesos(
    historico,
    nomes_features=None,
    incluir_vies=True,
    titulo="Evolucao dos pesos",
    caminho_arquivo=None,
    mostrar_graficos=False,
):
    """
    Plota todos os pesos ao longo das epocas.

    Diferente da trajetoria 2D, este grafico mostra explicitamente como
    cada parametro evolui durante o treinamento.
    """
    epocas = np.asarray(historico["epocas"])
    pesos = np.asarray(historico["w"])

    if pesos.ndim != 2:
        raise ValueError("O historico de pesos precisa ter shape (epocas, n_features).")

    nomes_componentes = _resolver_nomes_features(pesos.shape[1], nomes_features)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if incluir_vies:
        vies = np.asarray(historico["w0"])
        ax.plot(epocas, vies, linewidth=2.2, label="w0")

    for indice, nome in enumerate(nomes_componentes):
        ax.plot(epocas, pesos[:, indice], linewidth=1.8, label=nome)

    ax.set_title(titulo)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Valor do parametro")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2 if len(nomes_componentes) <= 5 else 3, fontsize=9)

    fig.tight_layout()
    salvar_ou_mostrar(fig, caminho_arquivo, mostrar_graficos=mostrar_graficos)
    return fig, ax


def plotar_fronteira_decisao(
    X,
    r,
    w0,
    w,
    nomes_features=None,
    titulo="Dados e fronteira de decisao",
    caminho_arquivo=None,
    mostrar_graficos=False,
):
    """
    Plota os dados e a fronteira de decisao para problemas com 2 features.
    """
    X = np.asarray(X)
    r = np.asarray(r)
    w = np.asarray(w)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("A fronteira de decisao so pode ser plotada quando X tem 2 features.")
    if w.shape[0] != 2:
        raise ValueError("O vetor de pesos precisa ter exatamente 2 componentes.")

    nomes_eixos = _resolver_nomes_features(2, nomes_features)
    classes = np.unique(r)
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for i, classe in enumerate(classes):
        mascara = r == classe
        ax.scatter(
            X[mascara, 0],
            X[mascara, 1],
            s=42,
            alpha=0.8,
            color=cores[i % len(cores)],
            label=f"Classe {classe}",
        )

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    dx = max(0.05 * (x_max - x_min), 1e-12)
    dy = max(0.05 * (y_max - y_min), 1e-12)

    ax.set_xlim(x_min - dx, x_max + dx)
    ax.set_ylim(y_min - dy, y_max + dy)

    if np.abs(w[1]) > 1e-12:
        x_linha = np.linspace(x_min - dx, x_max + dx, 300)
        y_linha = -(w0 + w[0] * x_linha) / w[1]
        ax.plot(x_linha, y_linha, color="black", linewidth=2, label="Fronteira")
    elif np.abs(w[0]) > 1e-12:
        x_vertical = -w0 / w[0]
        ax.axvline(x_vertical, color="black", linewidth=2, label="Fronteira")

    ax.set_title(titulo)
    ax.set_xlabel(nomes_eixos[0])
    ax.set_ylabel(nomes_eixos[1])
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    salvar_ou_mostrar(fig, caminho_arquivo, mostrar_graficos=mostrar_graficos)
    return fig, ax


def imprimir_tabela_resumo_treino(
    historicos,
    nomes=None,
    indices_pesos=(0, 1),
    nomes_features=None,
):
    """
    Imprime um resumo compacto do treino para um ou varios modelos.

    A tabela mostra, por modelo:
    - custo inicial e final;
    - reducao percentual do custo;
    - norma do gradiente no inicio e no fim;
    - vies final;
    - valores finais de dois pesos selecionados.
    """
    historicos_lista = _como_lista_de_historicos(historicos)
    rotulos = _resolver_rotulos(historicos_lista, nomes)

    if not historicos_lista:
        print("\n=== RESUMO DO TREINO ===")
        print("Nenhum historico disponivel.")
        return

    n_features = np.asarray(historicos_lista[0]["w"]).shape[1]
    idx_i, idx_j = indices_pesos
    if idx_i == idx_j:
        raise ValueError("Escolha dois pesos diferentes para a tabela de resumo.")
    if idx_i < 0 or idx_j < 0 or idx_i >= n_features or idx_j >= n_features:
        raise IndexError("Indices de pesos fora do intervalo valido.")

    nomes_eixos = _resolver_nomes_features(n_features, nomes_features)

    cabecalho = (
        f"{'Modelo':<22} {'Epocas':>6} {'Custo Ini':>12} {'Custo Fim':>12}"
        f" {'Red.%':>8} {'||g|| Ini':>12} {'||g|| Fim':>12} {'w0 fim':>11}"
        f" {nomes_eixos[idx_i]:>11} {nomes_eixos[idx_j]:>11}"
    )

    print("\n=== RESUMO DO TREINO ===")
    print(cabecalho)
    print("-" * len(cabecalho))

    for rotulo, historico in zip(rotulos, historicos_lista):
        custos = np.asarray(historico["custos"])
        norma_gradiente = np.asarray(historico["norma_gradiente"])
        pesos = np.asarray(historico["w"])
        vies = np.asarray(historico["w0"])

        custo_ini = float(custos[0])
        custo_fim = float(custos[-1])
        reducao_pct = 100.0 * (custo_ini - custo_fim) / max(abs(custo_ini), 1e-12)

        print(
            f"{rotulo:<22} {len(custos) - 1:>6} {custo_ini:>12.4f} {custo_fim:>12.4f}"
            f" {reducao_pct:>8.2f} {norma_gradiente[0]:>12.4f} {norma_gradiente[-1]:>12.4f}"
            f" {vies[-1]:>11.4f} {pesos[-1, idx_i]:>11.4f} {pesos[-1, idx_j]:>11.4f}"
        )


def imprimir_tabela_epocas(
    historico,
    nome="Modelo",
    indices_pesos=(0, 1),
    nomes_features=None,
    max_linhas=None,
):
    """
    Imprime uma tabela por epoca para um modelo especifico.

    `max_linhas` pode ser usado para limitar a saida. Quando informado,
    sao mostradas as primeiras e as ultimas linhas do treino.
    """
    epocas = np.asarray(historico["epocas"])
    custos = np.asarray(historico["custos"])
    norma_gradiente = np.asarray(historico["norma_gradiente"])
    vies = np.asarray(historico["w0"])
    pesos = np.asarray(historico["w"])

    idx_i, idx_j = indices_pesos
    if idx_i == idx_j:
        raise ValueError("Escolha dois pesos diferentes para a tabela por epoca.")
    if idx_i < 0 or idx_j < 0 or idx_i >= pesos.shape[1] or idx_j >= pesos.shape[1]:
        raise IndexError("Indices de pesos fora do intervalo valido.")

    nomes_eixos = _resolver_nomes_features(pesos.shape[1], nomes_features)
    linhas = list(range(len(epocas)))
    truncado = False

    if max_linhas is not None and len(linhas) > max_linhas:
        n_cabeca = max(1, max_linhas // 2)
        n_cauda = max_linhas - n_cabeca
        linhas = list(range(n_cabeca)) + list(range(len(epocas) - n_cauda, len(epocas)))
        truncado = True

    cabecalho = (
        f"{'Epoca':>6} {'Custo':>12} {'||g||':>12} {'w0':>11}"
        f" {nomes_eixos[idx_i]:>11} {nomes_eixos[idx_j]:>11}"
    )

    print(f"\n=== HISTORICO POR EPOCA: {nome} ===")
    print(cabecalho)
    print("-" * len(cabecalho))

    ultima_linha = None
    for indice in linhas:
        if truncado and ultima_linha is not None and indice - ultima_linha > 1:
            print("   ...")

        print(
            f"{epocas[indice]:>6} {custos[indice]:>12.4f} {norma_gradiente[indice]:>12.4f}"
            f" {vies[indice]:>11.4f} {pesos[indice, idx_i]:>11.4f} {pesos[indice, idx_j]:>11.4f}"
        )
        ultima_linha = indice


def _como_lista_de_historicos(historicos):
    if isinstance(historicos, dict):
        return [historicos]
    return list(historicos)


def _resolver_rotulos(historicos, nomes):
    if nomes is not None:
        return list(nomes)
    if len(historicos) == 1:
        return ["Modelo"]
    return [f"Modelo {i}" for i in range(len(historicos))]


def _resolver_nomes_features(n_features, nomes_features):
    if nomes_features is not None:
        return list(nomes_features)
    return [f"w{i + 1}" for i in range(n_features)]


def _reduzir_epocas_plot_3d(epocas, gradientes, max_epocas=320):
    epocas = np.asarray(epocas)
    gradientes = np.asarray(gradientes)

    if max_epocas is None or len(epocas) <= max_epocas:
        return epocas, gradientes

    indices = np.linspace(0, len(epocas) - 1, num=max_epocas, dtype=int)
    indices = np.unique(indices)
    return epocas[indices], gradientes[:, indices]


__all__ = [
    "imprimir_tabela_epocas",
    "imprimir_tabela_resumo_treino",
    "plotar_componentes_gradiente",
    "plotar_evolucao_pesos",
    "plotar_fronteira_decisao",
    "plotar_gradiente_3d",
    "plotar_metricas",
    "salvar_ou_mostrar",
]
