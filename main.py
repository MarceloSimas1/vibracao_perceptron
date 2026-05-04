import argparse
import importlib
import importlib.util
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vib.modelo_matematico import (
    MATERIAIS,
    N_CLASSES,
    NOMES,
    frequencia_natural,
    gerar_dataset,
    simular_vibracao,
)
from vib.visualizacao import (
    imprimir_tabela_resumo_treino,
    plotar_componentes_gradiente,
    plotar_evolucao_pesos,
    plotar_fronteira_decisao,
    plotar_gradiente_3d,
    plotar_metricas,
    salvar_ou_mostrar,
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
PASTA_RESULTADOS = "resultados"
N_POR_CLASSE = 250
FRACAO_TREINO = 0.8
DELTA = 1e-4
N_ITER = 10000


def carregar_perceptron(tipo):
    """
    Carrega o backend de aprendizado escolhido para o experimento.

    `numpy` usa `vib/perceptron.py`.
    `pytorch` usa `vib/perceptron-pytorch.py`, carregado por caminho porque
    o hifen no nome do arquivo impede import Python convencional.
    """
    if tipo == "numpy":
        return importlib.import_module("vib.perceptron")

    caminho = Path(__file__).resolve().parent / "vib" / "perceptron-pytorch.py"
    spec = importlib.util.spec_from_file_location("vib.perceptron_pytorch", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _como_numpy(valor):
    """
    Converte tensores PyTorch para NumPy quando alguma etapa externa exige.

    O backend PyTorch continua treinando com tensores; esta conversao fica
    restrita a metricas simples, Matplotlib e funcoes de visualizacao antigas.
    """
    if hasattr(valor, "detach"):
        return valor.detach().cpu().numpy()
    return np.asarray(valor)


def _extrair_parametros_lineares(modelo):
    """
    Extrai vies e pesos de um modelo linear em qualquer backend.

    No backend NumPy, o modelo ja e `(w0, w)`.
    No backend PyTorch, o modelo expõe `parametros_lineares()`.
    """
    if hasattr(modelo, "parametros_lineares"):
        w0, w = modelo.parametros_lineares()
        return float(_como_numpy(w0)), _como_numpy(w)
    return modelo


def _desenhar_matriz(ax, matriz, titulo):
    matriz = _como_numpy(matriz)
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(NOMES, rotation=18, ha="right", fontsize=9)
    ax.set_yticklabels(NOMES, fontsize=9)
    ax.set_xlabel("Classe Prevista", fontsize=10)
    ax.set_ylabel("Classe Real", fontsize=10)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    limiar = matriz.max() / 2 if matriz.size else 0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(
                j,
                i,
                str(matriz[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                color="white" if matriz[i, j] > limiar else "black",
            )
    return im


def plotar_resultados(
    historicos,
    y_tr_real,
    y_tr_pred,
    y_te_real,
    y_te_pred,
    matriz_confusao_fn,
    mostrar_graficos=False,
):
    cores = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))

    for i, hist in enumerate(historicos):
        ax1.semilogy(hist, label=NOMES[i], color=cores[i], linewidth=2)
    ax1.set_title(
        "Evolucao do Custo Total (escala log)", fontsize=12, fontweight="bold"
    )
    ax1.set_xlabel("Iteracao", fontsize=10)
    ax1.set_ylabel("Custo C", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    n_tr = len(y_tr_real)
    im2 = _desenhar_matriz(
        ax2,
        matriz_confusao_fn(y_tr_real, y_tr_pred, n=N_CLASSES),
        f"Matriz de Confusao - Treino ({n_tr} amostras)",
    )
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    n_te = len(y_te_real)
    im3 = _desenhar_matriz(
        ax3,
        matriz_confusao_fn(y_te_real, y_te_pred, n=N_CLASSES),
        f"Matriz de Confusao - Teste ({n_te} amostras)",
    )
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Perceptron - Classificacao de Material em Viga Engastada",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.38)
    salvar_ou_mostrar(
        fig,
        f"{PASTA_RESULTADOS}/resultados_perceptron_viga.png",
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
    t = np.linspace(0, 3.0, 1000)
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
    salvar_ou_mostrar(
        fig,
        f"{PASTA_RESULTADOS}/sinais_vibracao.png",
        mostrar_graficos=mostrar_graficos,
    )


def criar_parser():
    parser = argparse.ArgumentParser(
        description="Classifica materiais de uma viga engastada a partir de sinais de vibracao."
    )
    parser.add_argument(
        "--mostrar-graficos",
        action="store_true",
        help="abre as janelas do Matplotlib apos salvar os arquivos PNG",
    )
    parser.add_argument(
        "--perceptron",
        choices=("numpy", "pytorch"),
        default="numpy",
        help="seleciona o backend: perceptron.py (numpy) ou perceptron-pytorch.py",
    )
    return parser


def main(mostrar_graficos=False, tipo_perceptron="numpy"):
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
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    perceptron = carregar_perceptron(tipo_perceptron)

    print("=" * 58)
    print("  PERCEPTRON - MATERIAL DE VIGA ENGASTADA POR VIBRACAO")
    print("=" * 58)
    print(f"  Backend selecionado: perceptron-{tipo_perceptron}")

    print("\n[0] Gerando sinais de vibracao por material...")
    plotar_sinais_exemplo(mostrar_graficos=mostrar_graficos)

    print(
        f"\n[1] Gerando dataset simulado ({N_POR_CLASSE} amostras x {N_CLASSES} materiais)..."
    )
    X, y = gerar_dataset(n_por_classe=N_POR_CLASSE)
    print(f"    Total: {X.shape[0]} amostras  |  {X.shape[1]} features por amostra")
    print(f"    Features: {', '.join(NOMES_FEATURES)}")

    np.random.seed(0)
    idx = np.random.permutation(len(y))
    corte = int(FRACAO_TREINO * len(y))
    X_tr, X_te = X[idx[:corte]], X[idx[corte:]]
    y_tr, y_te = y[idx[:corte]], y[idx[corte:]]
    print(f"\n    Treino: {len(y_tr)} amostras  |  Teste: {len(y_te)} amostras")

    print("\n[2] Normalizando features (z-score, parametros do treino)...")
    X_tr_n, X_te_n, _, _ = perceptron.normalizar(X_tr, X_te)

    # Treina um classificador um-contra-todos para cada material.
    modelos, historicos, historicos_detalhados = perceptron.treinar_multiclasse(
        X_tr_n,
        y_tr,
        nomes=NOMES,
        delta=DELTA,
        n_iter=N_ITER,
    )

    print("\n[3] Avaliando no conjunto de treino e teste...")
    y_tr_pred = perceptron.prever(X_tr_n, modelos)
    y_pred = perceptron.prever(X_te_n, modelos)
    y_tr_pred_np = _como_numpy(y_tr_pred).astype(int)
    y_pred_np = _como_numpy(y_pred).astype(int)
    X_tr_n_np = _como_numpy(X_tr_n)
    acuracia_tr = np.mean(y_tr_pred_np == y_tr) * 100
    acuracia = np.mean(y_pred_np == y_te) * 100
    print(
        f"\n    Acuracia treino: {acuracia_tr:.1f} %  |  Acuracia teste: {acuracia:.1f} %"
    )

    print("\n    Desempenho por material (conjunto de teste):")
    print(
        f"    {'Material':<20} {'VP':>4} {'FN':>4} {'VN':>5} {'FP':>4}"
        f"  {'P(acerto|alvo)':>15}  {'P(acerto|nao-alvo)':>19}"
    )
    print("    " + "-" * 82)

    # Resume o desempenho de cada classe separadamente.
    metricas = perceptron.metricas_por_classe(y_te, y_pred, n=N_CLASSES)
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
    plotar_resultados(
        historicos,
        y_tr,
        y_tr_pred_np,
        y_te,
        y_pred_np,
        matriz_confusao_fn=perceptron.matriz_confusao,
        mostrar_graficos=mostrar_graficos,
    )
    plotar_metricas(
        historicos_detalhados,
        nomes=NOMES,
        titulo="Metricas do treinamento por classe",
        caminho_arquivo=f"{PASTA_RESULTADOS}/metricas_treinamento.png",
        mostrar_graficos=mostrar_graficos,
    )

    for nome, hist_det in zip(NOMES, historicos_detalhados):
        nome_arquivo = nome.lower().replace(" ", "_")
        plotar_evolucao_pesos(
            hist_det,
            nomes_features=NOMES_FEATURES,
            titulo=f"Evolucao dos pesos - {nome}",
            caminho_arquivo=f"{PASTA_RESULTADOS}/evolucao_pesos_{nome_arquivo}.png",
            mostrar_graficos=mostrar_graficos,
        )
        plotar_componentes_gradiente(
            hist_det,
            nomes_features=NOMES_FEATURES,
            titulo=f"Componentes do gradiente - {nome}",
            caminho_arquivo=f"{PASTA_RESULTADOS}/componentes_gradiente_{nome_arquivo}.png",
            mostrar_graficos=mostrar_graficos,
        )
        plotar_gradiente_3d(
            hist_det,
            nomes_features=NOMES_FEATURES,
            titulo=f"Gradiente 3D - {nome}",
            caminho_arquivo=f"{PASTA_RESULTADOS}/gradiente_3d_{nome_arquivo}.png",
            mostrar_graficos=mostrar_graficos,
        )

    if X_tr_n_np.shape[1] == 2:
        for classe, (modelo, nome) in enumerate(zip(modelos, NOMES)):
            w0, w = _extrair_parametros_lineares(modelo)
            r_bin = (y_tr == classe).astype(int)
            nome_arquivo = nome.lower().replace(" ", "_")
            plotar_fronteira_decisao(
                X_tr_n_np,
                r_bin,
                w0,
                w,
                nomes_features=NOMES_FEATURES[:2],
                titulo=f"Fronteira de decisao - {nome} vs resto",
                caminho_arquivo=f"{PASTA_RESULTADOS}/fronteira_decisao_{nome_arquivo}.png",
                mostrar_graficos=mostrar_graficos,
            )
    else:
        print(
            "  Fronteira de decisao nao gerada: o dataset atual possui mais de 2 features."
        )

    print("\n" + "=" * 58)
    print(f"  Concluido. Verifique a pasta '{PASTA_RESULTADOS}/'.")
    print("=" * 58)


if __name__ == "__main__":
    args = criar_parser().parse_args()
    main(mostrar_graficos=args.mostrar_graficos, tipo_perceptron=args.perceptron)
