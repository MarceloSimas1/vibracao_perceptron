import argparse

import matplotlib.pyplot as plt
import numpy as np

from modelo_matematico import (
    MATERIAIS,
    N_CLASSES,
    NOMES,
    frequencia_natural,
    gerar_dataset,
    simular_vibracao,
)
from perceptron import (
    matriz_confusao,
    metricas_por_classe,
    normalizar,
    prever,
)
from visualizacao_perceptron import (
    imprimir_tabela_resumo_treino,
    plotar_componentes_gradiente,
    plotar_evolucao_pesos,
    plotar_fronteira_decisao,
    plotar_metricas,
    treinar_multiclasse_com_historico,
)


# ============================================================
# ARQUIVO PRINCIPAL
# ============================================================
# Este modulo nao implementa a fisica nem o algoritmo de aprendizado.
# Ele apenas organiza a execucao do projeto:
#   1. gera sinais de exemplo;
#   2. gera o dataset sintetico;
#   3. separa treino e teste;
#   4. normaliza as features;
#   5. treina os classificadores;
#   6. avalia os resultados;
#   7. gera os graficos finais.


NOMES_FEATURES = [
    "RMS",
    "Pico",
    "Fat. Crista",
    "Freq. Dom.",
    "Decaimento",
    "Energia Esp.",
]
INDICES_PESOS_RESUMO = (0, 1)


def plotar_resultados(historicos, y_real, y_pred, mostrar_graficos=False):
    """
    Gera os graficos finais do experimento.

    Este relatorio visual possui duas partes:
    1. curva do custo total ao longo do treino para cada classe;
    2. matriz de confusao no conjunto de teste.

    Papel no projeto:
    - e chamada no final de `main`;
    - usa `matriz_confusao` do modulo `perceptron.py`.

    Parametros
    ----------
    historicos : list
        Historico do custo para cada classificador binario.
    y_real : np.ndarray
        Rotulos reais do conjunto de teste.
    y_pred : np.ndarray
        Rotulos previstos pelo modelo.
    """
    cores = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Painel 1: mostra se o custo caiu durante o treinamento.
    for i, hist in enumerate(historicos):
        ax1.semilogy(hist, label=NOMES[i], color=cores[i], linewidth=2)
    ax1.set_title(
        "Evolucao do Custo Total (escala log)", fontsize=12, fontweight="bold"
    )
    ax1.set_xlabel("Iteracao", fontsize=10)
    ax1.set_ylabel("Custo C", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Painel 2: compara classes reais e previstas.
    matriz = matriz_confusao(y_real, y_pred, n=N_CLASSES)
    im = ax2.imshow(matriz, cmap="Blues")
    ax2.set_xticks(range(N_CLASSES))
    ax2.set_yticks(range(N_CLASSES))
    ax2.set_xticklabels(NOMES, rotation=18, ha="right", fontsize=9)
    ax2.set_yticklabels(NOMES, fontsize=9)
    ax2.set_xlabel("Classe Prevista", fontsize=10)
    ax2.set_ylabel("Classe Real", fontsize=10)
    ax2.set_title(
        "Matriz de Confusao - Conjunto de Teste", fontsize=12, fontweight="bold"
    )

    limiar = matriz.max() / 2 if matriz.size else 0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax2.text(
                j,
                i,
                str(matriz[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if matriz[i, j] > limiar else "black",
            )

    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Perceptron - Classificacao de Material em Viga Engastada",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.35)
    salvar_ou_mostrar(
        fig,
        "resultados_perceptron_viga.png",
        mostrar_graficos=mostrar_graficos,
    )


def plotar_sinais_exemplo(mostrar_graficos=False):
    """
    Plota um sinal representativo para cada material da base.

    O objetivo aqui nao e treinar o modelo, e sim mostrar visualmente
    como os materiais diferem em frequencia e amortecimento.

    Papel no projeto:
    - e chamada no inicio de `main`;
    - usa `simular_vibracao` e `frequencia_natural` do modulo fisico.
    """
    t = np.linspace(0, 1.0, 1000)
    cores = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    fig.suptitle(
        "Sinais de Vibracao Livre - Viga Engastada (por material)",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (nome, props), cor in zip(axes.flat, MATERIAIS.items(), cores):
        sinal = simular_vibracao(props["E"], props["rho"], props["zeta"], t)
        fn_hz = frequencia_natural(props["E"], props["rho"]) / (2 * np.pi)
        ax.plot(t, sinal, color=cor, linewidth=0.9)
        ax.set_title(
            f"{nome}  |  f1 = {fn_hz:.1f} Hz  |  zeta = {props['zeta']}",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xlabel("Tempo [s]", fontsize=8)
        ax.set_ylabel("Deslocamento", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    salvar_ou_mostrar(fig, "sinais_vibracao.png", mostrar_graficos=mostrar_graficos)


def salvar_ou_mostrar(fig, caminho_arquivo, mostrar_graficos=False):
    """
    Salva a figura e, opcionalmente, abre a janela interativa.

    Por padrao o script roda de forma nao bloqueante, apenas gerando
    os arquivos PNG. A exibicao interativa pode ser habilitada via CLI.
    """
    fig.savefig(caminho_arquivo, dpi=150, bbox_inches="tight")
    if mostrar_graficos:
        plt.show()
    else:
        plt.close(fig)
    print(f"  Grafico salvo em '{caminho_arquivo}'")


def criar_parser():
    parser = argparse.ArgumentParser(
        description="Classifica materiais de uma viga engastada a partir de sinais de vibracao."
    )
    parser.add_argument(
        "--mostrar-graficos",
        action="store_true",
        help="abre as janelas do Matplotlib apos salvar os arquivos PNG",
    )
    return parser


def main(mostrar_graficos=False):
    """
    Orquestra o experimento completo de classificacao.

    Fluxo principal:
    1. plota sinais de exemplo;
    2. gera o dataset sintetico;
    3. divide os dados em treino e teste;
    4. normaliza as features;
    5. treina os classificadores;
    6. avalia a acuracia e metricas;
    7. gera os graficos finais.

    Esta funcao e o ponto de entrada do projeto quando o arquivo e
    executado diretamente.
    """
    print("=" * 58)
    print("  PERCEPTRON - MATERIAL DE VIGA ENGASTADA POR VIBRACAO")
    print("=" * 58)

    print("\n[0] Gerando sinais de vibracao por material...")
    plotar_sinais_exemplo(mostrar_graficos=mostrar_graficos)

    print("\n[1] Gerando dataset simulado (250 amostras x 4 materiais)...")
    X, y = gerar_dataset(n_por_classe=250)
    print(f"    Total: {X.shape[0]} amostras  |  {X.shape[1]} features por amostra")
    print(f"    Features: {', '.join(NOMES_FEATURES)}")

    # Embaralha o dataset e separa treino/teste em 80/20.
    np.random.seed(0)
    idx = np.random.permutation(len(y))
    corte = int(0.8 * len(y))
    X_tr, X_te = X[idx[:corte]], X[idx[corte:]]
    y_tr, y_te = y[idx[:corte]], y[idx[corte:]]
    print(f"\n    Treino: {len(y_tr)} amostras  |  Teste: {len(y_te)} amostras")

    print("\n[2] Normalizando features (z-score, parametros do treino)...")
    X_tr_n, X_te_n, _, _ = normalizar(X_tr, X_te)

    # Treina um classificador um-contra-todos para cada material.
    modelos, historicos, historicos_detalhados = treinar_multiclasse_com_historico(
        X_tr_n,
        y_tr,
        nomes=NOMES,
        delta=1e-3,
        n_iter=5000,
    )

    print("\n[3] Avaliando no conjunto de teste...")
    y_pred = prever(X_te_n, modelos)
    acuracia = np.mean(y_pred == y_te) * 100
    print(f"\n    Acuracia geral: {acuracia:.1f} %")

    print("\n    Desempenho por material (conjunto de teste):")
    print(
        f"    {'Material':<20} {'VP':>4} {'FN':>4} {'VN':>5} {'FP':>4}"
        f"  {'P(acerto|alvo)':>15}  {'P(acerto|nao-alvo)':>19}"
    )
    print("    " + "-" * 82)

    # Resume o desempenho de cada classe separadamente.
    metricas = metricas_por_classe(y_te, y_pred, n=N_CLASSES)
    for i, (vp, fn, vn, fp, p_alvo, p_nao_alvo) in enumerate(metricas):
        print(
            f"    {NOMES[i]:<20} {vp:>4} {fn:>4} {vn:>5} {fp:>4}"
            f"  {p_alvo:>14.1f}%  {p_nao_alvo:>18.1f}%"
        )

    print("\n[4] Resumo do treino no terminal...")
    imprimir_tabela_resumo_treino(
        historicos_detalhados,
        nomes=NOMES,
        indices_pesos=INDICES_PESOS_RESUMO,
        nomes_features=NOMES_FEATURES,
    )

    print("\n[5] Gerando visualizacoes...")
    plotar_resultados(historicos, y_te, y_pred, mostrar_graficos=mostrar_graficos)
    plotar_metricas(
        historicos_detalhados,
        nomes=NOMES,
        titulo="Metricas do treinamento por classe",
        caminho_arquivo="metricas_treinamento.png",
        mostrar_graficos=mostrar_graficos,
    )

    for nome, hist_det in zip(NOMES, historicos_detalhados):
        nome_arquivo = nome.lower().replace(" ", "_")
        plotar_evolucao_pesos(
            hist_det,
            nomes_features=NOMES_FEATURES,
            titulo=f"Evolucao dos pesos - {nome}",
            caminho_arquivo=f"evolucao_pesos_{nome_arquivo}.png",
            mostrar_graficos=mostrar_graficos,
        )
        plotar_componentes_gradiente(
            hist_det,
            nomes_features=NOMES_FEATURES,
            titulo=f"Componentes do gradiente - {nome}",
            caminho_arquivo=f"componentes_gradiente_{nome_arquivo}.png",
            mostrar_graficos=mostrar_graficos,
        )

    if X_tr_n.shape[1] == 2:
        for classe, ((w0, w), nome) in enumerate(zip(modelos, NOMES)):
            r_bin = (y_tr == classe).astype(int)
            nome_arquivo = nome.lower().replace(" ", "_")
            plotar_fronteira_decisao(
                X_tr_n,
                r_bin,
                w0,
                w,
                nomes_features=NOMES_FEATURES[:2],
                titulo=f"Fronteira de decisao - {nome} vs resto",
                caminho_arquivo=f"fronteira_decisao_{nome_arquivo}.png",
                mostrar_graficos=mostrar_graficos,
            )
    else:
        print(
            "  Fronteira de decisao nao gerada: o dataset atual possui mais de 2 features."
        )

    print("\n" + "=" * 58)
    print("  Concluido. Verifique os arquivos .png gerados.")
    print("=" * 58)


if __name__ == "__main__":
    args = criar_parser().parse_args()
    main(mostrar_graficos=args.mostrar_graficos)
