import colorsys
import re
import bs4
import numpy as np
import pandas as pd
import requests
from PIL import Image
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

# python -m streamlit run H:\Python\teste_ocr_hex\random_forest.py

st.set_page_config(
    page_title="Classificador de Cores com Random Forest", layout="wide"
)
st.title("🎨 Classificador de Cores de Imagem")

# ==========================================
# 1. FUNÇÕES AUXILIARES DE PROCESSAMENTO
# ==========================================


def categorizar_cor(nome):
    """Mapeia o nome em inglês/HTML para uma família principal de cores."""
    nome = nome.lower()
    if any(
        k in nome
        for k in [
            "blue",
            "cyan",
            "aqua",
            "azure",
            "aliceblue",
            "cadetblue",
            "sky",
            "navy",
            "turquoise",
            "steel",
        ]
    ):
        return "Azul"
    elif any(
        k in nome
        for k in [
            "green",
            "lime",
            "chartreuse",
            "emerald",
            "springgreen",
            "olive",
            "aquamarine",
            "sea",
        ]
    ):
        return "Verde"
    elif any(
        k in nome
        for k in [
            "red",
            "firebrick",
            "crimson",
            "coral",
            "tomato",
            "salmon",
            "maroon",
            "indianred",
        ]
    ):
        return "Vermelho"
    elif any(
        k in nome for k in ["pink", "magenta", "hotpink", "deeppink", "blush"]
    ):
        return "Rosa"
    elif any(
        k in nome
        for k in [
            "purple",
            "violet",
            "orchid",
            "plum",
            "thistle",
            "lavender",
            "indigo",
        ]
    ):
        return "Roxo"
    elif any(
        k in nome
        for k in [
            "yellow",
            "gold",
            "khaki",
            "lemon",
            "goldenrod",
            "cornsilk",
            "chiffon",
        ]
    ):
        return "Amarelo"
    elif any(
        k in nome
        for k in ["orange", "peach", "papaya", "bisque", "moccasin", "sandy"]
    ):
        return "Laranja"
    elif any(
        k in nome
        for k in [
            "brown",
            "chocolate",
            "sienna",
            "saddle",
            "tan",
            "wheat",
            "beige",
            "peru",
            "burlywood",
        ]
    ):
        return "Marrom"
    elif any(
        k in nome
        for k in [
            "gray",
            "grey",
            "slate",
            "black",
            "white",
            "snow",
            "ghost",
            "gainsboro",
            "smoke",
            "ivory",
            "linen",
            "misty",
            "seashell",
        ]
    ):
        return "Cinza/Neutro"
    else:
        return "Outros"


def aumentar_dados(df_original, multiplicador=10, ruida_max=8):
    """Gera dados sintéticos aplicando pequenas variações no RGB das amostras originais."""
    novas_linhas = []
    for _, row in df_original.iterrows():
        novas_linhas.append(
            {
                "nome_cor": row["nome_cor"],
                "r": row["r"],
                "g": row["g"],
                "b": row["b"],
                "categoria": row["categoria"],
            }
        )

    for _ in range(multiplicador):
        for _, row in df_original.iterrows():
            r_ruido = int(
                np.clip(
                    row["r"] + np.random.randint(-ruida_max, ruida_max + 1),
                    0,
                    255,
                )
            )
            g_ruido = int(
                np.clip(
                    row["g"] + np.random.randint(-ruida_max, ruida_max + 1),
                    0,
                    255,
                )
            )
            b_ruido = int(
                np.clip(
                    row["b"] + np.random.randint(-ruida_max, ruida_max + 1),
                    0,
                    255,
                )
            )

            novas_linhas.append(
                {
                    "nome_cor": row["nome_cor"],
                    "r": r_ruido,
                    "g": g_ruido,
                    "b": b_ruido,
                    "categoria": row["categoria"],
                }
            )

    return pd.DataFrame(novas_linhas)


def calcular_features(r, g, b):
    """Converte valores RGB para HSV e calcula luminância."""
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    h_graus = h * 360
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return [r, g, b, h_graus, s, v, luminance]


colunas_features = ["r", "g", "b", "h", "s", "v", "luminance"]

# ==========================================
# 2. TREINAMENTO DO MODELO (COM CACHE)
# ==========================================


@st.cache_resource
def carregar_e_treinar_modelo():
    """Realiza o scraping e treina o Random Forest uma única vez."""
    url = "https://www.flextool.com.br/tabela_cores.html"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    response.encoding = "utf-8"
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    data = []
    for row in soup.find_all("tr"):
        cols = [ele.text.strip() for ele in row.find_all(["td", "th"])]
        if len(cols) >= 3:
            nome_cor = cols[0]
            if nome_cor.lower() in ["cor", "código java", "código html"]:
                continue
            rgb_vals = re.findall(r"\d+", cols[1])
            if len(rgb_vals) == 3:
                r, g, b = map(int, rgb_vals)
                data.append({"nome_cor": nome_cor, "r": r, "g": g, "b": b})

    df_raw = (
        pd.DataFrame(data)
        .drop_duplicates(subset=["nome_cor"])
        .reset_index(drop=True)
    )
    df_raw["categoria"] = df_raw["nome_cor"].apply(categorizar_cor)
    df_raw = df_raw[df_raw["categoria"] != "Outros"].reset_index(drop=True)

    df_aumentado = aumentar_dados(df_raw, multiplicador=10, ruida_max=10)

    features_matriz = [
        calcular_features(row["r"], row["g"], row["b"])
        for _, row in df_aumentado.iterrows()
    ]
    X = pd.DataFrame(features_matriz, columns=colunas_features)
    y = df_aumentado["categoria"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [100, 200], 
        "max_depth": [10, 20, None], 
        "min_samples_split": [2, 5], 
        "class_weight": ["balanced"],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    modelo = grid_search.best_estimator_

    acuracia = accuracy_score(y_test, modelo.predict(X_test))
    return modelo, acuracia


with st.spinner("Carregando base de dados e treinando modelo IA..."):
    modelo_otimizado, acuracia_modelo = carregar_e_treinar_modelo()

st.sidebar.success(f"Modelo pronto! Acurácia: {acuracia_modelo * 100:.2f}%")

# ==========================================
# 3. INTERFACE DE USUÁRIO E PROCESSAMENTO
# ==========================================

arquivo_imagem = st.file_uploader(
    "Envie uma imagem para análise", type=["jpg", "jpeg", "png", "webp"]
)

if arquivo_imagem is not None:
    img = Image.open(arquivo_imagem).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img, caption="Imagem Carregada")

    with col2:
        with st.spinner("Extraindo e identificando cores..."):
            # Opcional: Reduz a imagem para max 64 cores se for muito grande
            cores_rgb = list(set(img.getdata()))

            # Matriz de features vetorizada
            matriz_features = [
                calcular_features(r, g, b) for r, g, b in cores_rgb
            ]
            df_batch = pd.DataFrame(
                matriz_features, columns=colunas_features
            )

            # Predição em lote
            predicoes = modelo_otimizado.predict(df_batch)

            # Montagem do DataFrame final
            df_resultados = pd.DataFrame(
                {
                    "R": [c[0] for c in cores_rgb],
                    "G": [c[1] for c in cores_rgb],
                    "B": [c[2] for c in cores_rgb],
                    "Categoria": predicoes,
                }
            )

            # Contagem de ocorrências por categoria
            contagem_cores = df_resultados["Categoria"].value_counts()

            st.subheader("Distribuição de Cores Identificadas")
            st.bar_chart(contagem_cores)

            st.subheader("Tabela de Resultados")
            st.dataframe(df_resultados)