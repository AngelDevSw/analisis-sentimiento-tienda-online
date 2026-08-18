# Procesamiento de Lenguaje Natural - Proyecto integrador
# Python 3.11 + Miniconda + Visual Studio Code
# Proyecto: Análisis de sentimiento en reseñas de productos de tienda en línea
# Corte III: corpus final de 1000 reseñas,
# conservación de stopwords críticas y evaluación final de modelos
# Autor: Angel Montoya, Manuel Miranda, Eduardo Taurino
# Fecha: 2026-07-02

from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import math
import re

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


ARCHIVO_ENTRADA = Path("corpus_sentimiento.csv")
ARCHIVO_SALIDA_REGLAS = Path("resultados_analisis_sentimiento.csv")
CARPETA_SALIDA = Path("salida")
CARPETA_SALIDA_MODELO = Path("salida_modelo")
MODELO_SPACY = "es_core_news_sm"
ETIQUETAS_ESPERADAS = {"positivo", "negativo", "neutro", "mixto"}

STOPWORDS_CRITICAS = {
    "no",
    "nunca",
    "jamás",
    "pero",
    "aunque",
    "sin",
    "embargo",
}

PALABRAS_POSITIVAS = {
    "excelente", "perfecto", "perfecta", "bien", "bueno", "buena", "buen", "buenisimo",
    "buenisima", "genial", "increible", "maravilla", "encantar", "gustar", "feliz",
    "contento", "satisfecho", "recomendar", "recomendado", "recomendable", "confiable",
    "original", "premium", "elegante", "resistente", "solido", "robusto", "duradero",
    "funcional", "practico", "facil", "rapido", "puntual", "impecable", "cumplir",
    "superar", "mejor", "calidad", "accesible", "justo", "atento", "protegido",
    "sellado", "chido", "padre", "bonito", "util", "comodo", "seguro", "limpio",
}

PALABRAS_NEGATIVAS = {
    "dañado", "dañada", "danado", "danada", "aplastado", "aplastada", "mala", "malo",
    "malisimo", "malisima", "pesimo", "pesima", "decepcion", "decepcionado",
    "decepcionante", "terrible", "horrible", "barato", "barata", "fragil", "endeble",
    "defectuoso", "incompleto", "incompleta", "equivocado", "equivocada", "engaño",
    "enganosa", "estafa", "peligroso", "inaceptable", "vergonzoso", "grosero",
    "falla", "fallas", "fallo", "fallar", "roto", "rota", "romper", "rompio",
    "rayado", "rayada", "rayadura", "mancha", "manchado", "golpeado", "abierto",
    "abierta", "usado", "humedad", "ruido", "baja", "debil", "dificil", "tardar",
    "tarde", "nunca", "devolver", "devolucion", "faltar", "faltaban", "error",
    "peor", "sucio", "caducado", "caducada", "cobrar", "doble", "maltratado",
}

PALABRAS_NEUTRAS = {
    "producto", "articulo", "pedido", "paquete", "caja", "empaque", "embalaje",
    "vendedor", "compra", "entrega", "envio", "precio", "calidad", "material",
    "color", "talla", "tamano", "foto", "imagen", "descripcion", "ficha", "pagina",
    "anuncio", "caracteristica", "especificacion", "funcion", "uso", "diario", "basico",
    "basica", "normal", "estandar", "promedio", "simple", "sencillo", "sencilla",
    "correcto", "correcta", "adecuado", "adecuada", "aceptable", "condicion", "fecha",
    "tiempo", "rango", "proceso", "categoria", "observacion", "comentario", "recibido",
}

PALABRAS_PROTEGIDAS = (
    STOPWORDS_CRITICAS
    | PALABRAS_POSITIVAS
    | PALABRAS_NEGATIVAS
    | PALABRAS_NEUTRAS
)

CONECTORES_MIXTOS = {
    "pero", "aunque", "sin embargo", "aun así", "aun asi", "no obstante", "por otro lado"
}




# -----------------------------------------------------------------------------
# 1. Carga y validación de recursos
# -----------------------------------------------------------------------------

def cargar_modelo_spacy() -> spacy.language.Language:
    """Carga el modelo de spaCy para español y muestra una instrucción clara si falta."""
    try:
        return spacy.load(MODELO_SPACY)
    except OSError as exc:
        raise RuntimeError(
            f"No se encontró el modelo de spaCy '{MODELO_SPACY}'.\n"
            f"Instálalo con este comando:\n"
            f"python -m spacy download {MODELO_SPACY}"
        ) from exc


def cargar_stopwords_espanol(nlp: spacy.language.Language) -> set[str]:
    """
    Carga stopwords en español desde NLTK y spaCy,
    conservando palabras críticas para el análisis de sentimiento.
    """
    try:
        stopwords_nltk = set(stopwords.words("spanish"))
    except LookupError:
        nltk.download("stopwords")
        stopwords_nltk = set(stopwords.words("spanish"))

    stopwords_spacy = set(nlp.Defaults.stop_words)

    # Combinar las stopwords de ambas librerías
    stopwords_combinadas = stopwords_nltk.union(stopwords_spacy)

    # Conservar términos importantes para sentimiento,
    # negación y contraste.
    stopwords_combinadas = stopwords_combinadas - PALABRAS_PROTEGIDAS

    return stopwords_combinadas


def cargar_corpus(ruta: Path) -> pd.DataFrame:
    """Lee el corpus en CSV y valida columnas obligatorias, datos vacíos y etiquetas."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {ruta.resolve()}")

    datos = pd.read_csv(ruta, encoding="utf-8-sig")
    columnas_obligatorias = {"id", "texto", "etiqueta"}
    columnas_actuales = set(datos.columns)
    columnas_faltantes = columnas_obligatorias - columnas_actuales

    if columnas_faltantes:
        faltantes = ", ".join(sorted(columnas_faltantes))
        raise ValueError(f"El CSV no contiene las columnas obligatorias: {faltantes}")

    datos = datos[["id", "texto", "etiqueta"]].copy()
    datos["texto"] = datos["texto"].astype(str).str.strip()
    datos["etiqueta"] = datos["etiqueta"].astype(str).str.strip().str.lower()
    datos = datos.dropna(subset=["id", "texto", "etiqueta"])
    datos = datos[datos["texto"] != ""]

    etiquetas_encontradas = set(datos["etiqueta"].unique())
    etiquetas_no_validas = etiquetas_encontradas - ETIQUETAS_ESPERADAS
    if etiquetas_no_validas:
        no_validas = ", ".join(sorted(etiquetas_no_validas))
        raise ValueError(f"El corpus contiene etiquetas no válidas: {no_validas}")

    return datos


# -----------------------------------------------------------------------------
# 2. Preprocesamiento con spaCy y NLTK
# -----------------------------------------------------------------------------

def limpieza_basica(texto: str) -> str:
    """Realiza limpieza inicial antes de enviar el texto a spaCy."""
    texto = str(texto).lower()
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"[@#]\w+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def procesar_texto_spacy(texto: str, nlp: spacy.language.Language, stopwords_es: set[str]) -> tuple[str, list[str]]:
    """Procesa un comentario con spaCy: tokeniza, elimina ruido, stopwords y lematiza."""
    texto_limpio = limpieza_basica(texto)
    documento = nlp(texto_limpio)
    tokens_utiles = []

    for token in documento:
        lema = token.lemma_.lower().strip()
        texto_token = token.text.lower().strip()

        if token.is_space or token.is_punct:
            continue

        if token.like_num:
            continue
        # Eliminar stopwords, excepto palabras protegidas
        if (
            texto_token not in PALABRAS_PROTEGIDAS
            and lema not in PALABRAS_PROTEGIDAS
            and (
                token.is_stop
                or lema in stopwords_es
                or texto_token in stopwords_es
            )
        ):
            continue
        # Eliminar palabras demaciadas cortas
        # Excepto aquellas que sean sumamente relevantes. 
        if len(lema) <= 2 and texto_token not in PALABRAS_PROTEGIDAS:
            continue

        if not re.match(r"^[a-záéíóúüñ]+$", lema):
            continue

        tokens_utiles.append(lema)


    # Respaldo: Evitar que una reseña quede completamente vacía
    if not tokens_utiles:
        for token in documento:
            if token.is_space or token.is_punct:
                continue
            texto_token = token.text.lower().strip()
            lema = token.lemma_.lower().strip()
            # Conservar números si el comentario está formado
            # principalmente por una valoración como 10/10.
            if token.like_num:
                tokens_utiles.append(f"num_{texto_token}")
                continue
            # En el respaldo conservamos palabras con contenido
            # aunque originalmente fueran consideradas stopwords.
            if (
                len(lema) > 2
                and re.match(r"^[a-záéíóúüñ]+$", lema)
            ):
                tokens_utiles.append(lema)

    return " ".join(tokens_utiles), tokens_utiles


def preprocesar_corpus(datos: pd.DataFrame, nlp: spacy.language.Language, stopwords_es: set[str]) -> pd.DataFrame:
    """Agrega texto procesado y tokens útiles al corpus original."""
    textos_procesados = []
    tokens_por_comentario = []

    for texto in datos["texto"]:
        texto_procesado, tokens = procesar_texto_spacy(texto, nlp, stopwords_es)
        textos_procesados.append(texto_procesado)
        tokens_por_comentario.append(tokens)

    datos_procesados = datos.copy()
    datos_procesados["texto_procesado"] = textos_procesados
    datos_procesados["tokens_utiles"] = tokens_por_comentario
    datos_procesados["tokens_utiles_texto"] = datos_procesados["tokens_utiles"].apply(lambda tokens: ", ".join(tokens))
    datos_procesados["palabras_frecuentes"] = datos_procesados["tokens_utiles"].apply(
        lambda tokens: ", ".join([f"{palabra}:{conteo}" for palabra, conteo in Counter(tokens).most_common(5)])
    )

    return datos_procesados


# -----------------------------------------------------------------------------
# 3. Clasificación por reglas como línea base
# -----------------------------------------------------------------------------

def clasificar_sentimiento_reglas(tokens: list[str], texto_original: str) -> str:
    """Clasifica sentimiento mediante reglas simples; funciona solo como línea base."""
    texto_normalizado = limpieza_basica(texto_original)
    positivos = sum(1 for token in tokens if token in PALABRAS_POSITIVAS)
    negativos = sum(1 for token in tokens if token in PALABRAS_NEGATIVAS)
    neutros = sum(1 for token in tokens if token in PALABRAS_NEUTRAS)
    tiene_conector_mixto = any(conector in texto_normalizado for conector in CONECTORES_MIXTOS)

    if (positivos > 0 and negativos > 0) or (tiene_conector_mixto and (positivos > 0 or negativos > 0)):
        return "mixto"
    if positivos > negativos:
        return "positivo"
    if negativos > positivos:
        return "negativo"
    if neutros > 0:
        return "neutro"
    return "neutro"


def generar_linea_base_reglas(datos: pd.DataFrame) -> pd.DataFrame:
    """Genera la evaluación exploratoria del clasificador basado en reglas."""
    resultados = datos.copy()
    resultados["sentimiento_reglas"] = resultados.apply(
        lambda fila: clasificar_sentimiento_reglas(fila["tokens_utiles"], fila["texto"]),
        axis=1,
    )
    resultados["evaluacion_reglas"] = resultados.apply(
        lambda fila: "correcto" if fila["etiqueta"] == fila["sentimiento_reglas"] else "revisar",
        axis=1,
    )

    return resultados[[
        "id", "texto", "etiqueta", "texto_procesado", "tokens_utiles_texto",
        "palabras_frecuentes", "sentimiento_reglas", "evaluacion_reglas"
    ]].rename(columns={
        "texto": "comentario_original",
        "etiqueta": "etiqueta_original",
        "tokens_utiles_texto": "tokens_utiles",
    })


def guardar_linea_base_reglas(resultados_reglas: pd.DataFrame, ruta: Path) -> None:
    """Guarda el análisis por reglas en CSV."""
    resultados_reglas.to_csv(ruta, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)


# -----------------------------------------------------------------------------
# 4. Representación numérica del corpus
# -----------------------------------------------------------------------------

def calcular_tf_manual(tokens: list[str], vocabulario: list[str]) -> dict[str, float]:
    """Calcula frecuencia de término para un documento."""
    frecuencia = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {termino: 0.0 for termino in vocabulario}
    return {termino: frecuencia.get(termino, 0) / total for termino in vocabulario}


def calcular_idf_manual(documentos_tokens: list[list[str]], vocabulario: list[str]) -> dict[str, float]:
    """Calcula IDF suavizado para el vocabulario completo."""
    total_documentos = len(documentos_tokens)
    idf = {}

    for termino in vocabulario:
        documentos_con_termino = sum(1 for tokens in documentos_tokens if termino in set(tokens))
        idf[termino] = math.log((1 + total_documentos) / (1 + documentos_con_termino)) + 1

    return idf


def similitud_coseno(vector_a: dict[str, float], vector_b: dict[str, float], vocabulario: list[str]) -> float:
    """Calcula similitud coseno entre dos vectores TF-IDF."""
    producto_punto = sum(vector_a[termino] * vector_b[termino] for termino in vocabulario)
    norma_a = math.sqrt(sum(vector_a[termino] ** 2 for termino in vocabulario))
    norma_b = math.sqrt(sum(vector_b[termino] ** 2 for termino in vocabulario))

    if norma_a == 0 or norma_b == 0:
        return 0.0
    return producto_punto / (norma_a * norma_b)


def generar_archivos_representacion(datos: pd.DataFrame, carpeta_salida: Path) -> None:
    """Genera preprocesamiento, BoW, TF, TF-IDF, términos principales y similitud coseno."""
    carpeta_salida.mkdir(exist_ok=True)

    datos[["id", "texto", "etiqueta", "texto_procesado", "tokens_utiles_texto"]].rename(columns={
        "texto": "texto_original",
        "etiqueta": "etiqueta_original",
        "tokens_utiles_texto": "tokens_utiles",
    }).to_csv(carpeta_salida / "01_preprocesamiento.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    vectorizador_bow = CountVectorizer()
    matriz_bow = vectorizador_bow.fit_transform(datos["texto_procesado"])
    vocabulario_bow = vectorizador_bow.get_feature_names_out()
    bow_df = pd.DataFrame(matriz_bow.toarray(), columns=vocabulario_bow)
    bow_df.insert(0, "etiqueta_original", datos["etiqueta"].values)
    bow_df.insert(0, "id", datos["id"].values)
    bow_df.to_csv(carpeta_salida / "02_matriz_bow.csv", index=False, encoding="utf-8-sig")

    documentos_tokens = datos["tokens_utiles"].tolist()
    vocabulario = sorted({token for tokens in documentos_tokens for token in tokens})
    idf = calcular_idf_manual(documentos_tokens, vocabulario)

    filas_tf = []
    filas_tfidf = []
    filas_top_tfidf = []
    vectores_tfidf = []

    for _, fila in datos.iterrows():
        tokens = fila["tokens_utiles"]
        tf = calcular_tf_manual(tokens, vocabulario)
        tfidf = {termino: tf[termino] * idf[termino] for termino in vocabulario}
        vectores_tfidf.append(tfidf)

        filas_tf.append({
            "id": fila["id"],
            "etiqueta_original": fila["etiqueta"],
            **{termino: round(valor, 6) for termino, valor in tf.items()},
        })
        filas_tfidf.append({
            "id": fila["id"],
            "etiqueta_original": fila["etiqueta"],
            **{termino: round(valor, 6) for termino, valor in tfidf.items()},
        })

        top_terminos = sorted(tfidf.items(), key=lambda item: item[1], reverse=True)[:5]
        filas_top_tfidf.append({
            "id": fila["id"],
            "etiqueta_original": fila["etiqueta"],
            "texto_original": fila["texto"],
            "top_terminos_tfidf": ", ".join([f"{termino}:{valor:.4f}" for termino, valor in top_terminos if valor > 0]),
        })

    pd.DataFrame(filas_tf).to_csv(carpeta_salida / "03_matriz_tf.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_tfidf).to_csv(carpeta_salida / "04_matriz_tfidf.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_top_tfidf).to_csv(carpeta_salida / "05_top_terminos_tfidf.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    similitudes = []
    total_documentos = len(datos)
    ids = datos["id"].tolist()
    etiquetas = datos["etiqueta"].tolist()

    for i in range(total_documentos):
        for j in range(i + 1, total_documentos):
            similitud = similitud_coseno(vectores_tfidf[i], vectores_tfidf[j], vocabulario)
            similitudes.append({
                "id_documento_1": ids[i],
                "etiqueta_documento_1": etiquetas[i],
                "id_documento_2": ids[j],
                "etiqueta_documento_2": etiquetas[j],
                "similitud_coseno": round(similitud, 6),
            })

    similitudes_df = pd.DataFrame(similitudes).sort_values("similitud_coseno", ascending=False).head(50)
    similitudes_df.to_csv(carpeta_salida / "06_similitud_documentos.csv", index=False, encoding="utf-8-sig")

    print("\nREPRESENTACIÓN NUMÉRICA DEL CORPUS")
    print("-" * 60)
    print(f"Documentos procesados: {len(datos)}")
    print(f"Tamaño del vocabulario manual: {len(vocabulario)}")
    print(f"Archivos generados en: {carpeta_salida.resolve()}")


# -----------------------------------------------------------------------------
# 5. Entrenamiento y evaluación de modelos supervisados
# -----------------------------------------------------------------------------

def construir_modelos() -> dict[str, object]:
    """Define los cuatro modelos supervisados comparados en el Corte II."""
    return {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "LinearSVC": LinearSVC(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    }


def entrenar_y_evaluar_modelos(datos: pd.DataFrame, carpeta_salida_modelo: Path) -> None:
    """Entrena cuatro modelos, compara métricas y guarda resultados del mejor modelo."""
    carpeta_salida_modelo.mkdir(exist_ok=True)

    datos_modelo = datos[["id", "texto", "etiqueta", "texto_procesado"]].copy()
    datos_modelo = datos_modelo.rename(columns={
        "texto": "texto_original",
        "etiqueta": "etiqueta_original",
    })
    datos_modelo.to_csv(
        carpeta_salida_modelo / "01_datos_preprocesados_spacy.csv",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
    )

    x = datos["texto_procesado"]
    y = datos["etiqueta"]

    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba, indices_entrenamiento, indices_prueba = train_test_split(
        x,
        y,
        datos.index,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    vectorizador = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x_entrenamiento_tfidf = vectorizador.fit_transform(x_entrenamiento)
    x_prueba_tfidf = vectorizador.transform(x_prueba)

    modelos = construir_modelos()
    metricas = []
    predicciones_por_modelo = {}

    for nombre_modelo, modelo in modelos.items():
        modelo.fit(x_entrenamiento_tfidf, y_entrenamiento)
        predicciones = modelo.predict(x_prueba_tfidf)
        predicciones_por_modelo[nombre_modelo] = predicciones

        metricas.append({
            "modelo": nombre_modelo,
            "accuracy": round(accuracy_score(y_prueba, predicciones), 6),
            "precision_macro": round(precision_score(y_prueba, predicciones, average="macro", zero_division=0), 6),
            "recall_macro": round(recall_score(y_prueba, predicciones, average="macro", zero_division=0), 6),
            "f1_macro": round(f1_score(y_prueba, predicciones, average="macro", zero_division=0), 6),
        })

    metricas_df = pd.DataFrame(metricas).sort_values("f1_macro", ascending=False)
    metricas_df.to_csv(carpeta_salida_modelo / "02_metricas_modelos.csv", index=False, encoding="utf-8-sig")

    mejor_modelo_nombre = metricas_df.iloc[0]["modelo"]
    mejores_predicciones = predicciones_por_modelo[mejor_modelo_nombre]

    vocabulario_df = pd.DataFrame({"termino": vectorizador.get_feature_names_out()})
    vocabulario_df.to_csv(carpeta_salida_modelo / "03_vocabulario_tfidf.csv", index=False, encoding="utf-8-sig")

    matriz_tfidf_completa = vectorizador.transform(datos["texto_procesado"])
    matriz_tfidf_df = pd.DataFrame(matriz_tfidf_completa.toarray(), columns=vectorizador.get_feature_names_out())
    matriz_tfidf_df.insert(0, "etiqueta_original", datos["etiqueta"].values)
    matriz_tfidf_df.insert(0, "id", datos["id"].values)
    matriz_tfidf_df.to_csv(carpeta_salida_modelo / "04_matriz_tfidf.csv", index=False, encoding="utf-8-sig")

    etiquetas_ordenadas = ["mixto", "negativo", "neutro", "positivo"]
    matriz_confusion = confusion_matrix(y_prueba, mejores_predicciones, labels=etiquetas_ordenadas)
    matriz_confusion_df = pd.DataFrame(
        matriz_confusion,
        index=[f"real_{etiqueta}" for etiqueta in etiquetas_ordenadas],
        columns=[f"predicho_{etiqueta}" for etiqueta in etiquetas_ordenadas],
    )
    matriz_confusion_df.to_csv(carpeta_salida_modelo / "05_matriz_confusion_mejor_modelo.csv", encoding="utf-8-sig")

    datos_prueba = datos.loc[indices_prueba].copy()
    predicciones_prueba_df = pd.DataFrame({
        "id": datos_prueba["id"].values,
        "texto_original": datos_prueba["texto"].values,
        "texto_procesado": datos_prueba["texto_procesado"].values,
        "etiqueta_real": y_prueba.values,
        "etiqueta_predicha": mejores_predicciones,
        "modelo": mejor_modelo_nombre,
    })
    predicciones_prueba_df.to_csv(
        carpeta_salida_modelo / "06_predicciones_prueba.csv",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
    )

    reporte = classification_report(
        y_prueba,
        mejores_predicciones,
        labels=etiquetas_ordenadas,
        output_dict=True,
        zero_division=0,
    )
    reporte_df = pd.DataFrame(reporte).transpose()
    reporte_df.to_csv(carpeta_salida_modelo / "07_reporte_clasificacion_mejor_modelo.csv", encoding="utf-8-sig")

    errores_df = predicciones_prueba_df[
        predicciones_prueba_df["etiqueta_real"] != predicciones_prueba_df["etiqueta_predicha"]
    ].copy()
    errores_df.to_csv(
        carpeta_salida_modelo / "08_errores_modelo.csv",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
    )

    print("\nMODELOS SUPERVISADOS DE MACHINE LEARNING")
    print("-" * 60)
    print(metricas_df.to_string(index=False))
    mejor_fila = metricas_df.iloc[0]
    print("\nMejor modelo seleccionado por F1 macro:")
    print(f"- Modelo          : {mejor_modelo_nombre}")
    print(f"- Accuracy        : {mejor_fila['accuracy']:.4f}")
    print(f"- Precision macro : {mejor_fila['precision_macro']:.4f}")
    print(f"- Recall macro    : {mejor_fila['recall_macro']:.4f}")
    print(f"- F1 macro        : {mejor_fila['f1_macro']:.4f}")
    print(f"- Errores en prueba: {len(errores_df)}")
    print(f"Archivos generados en: {carpeta_salida_modelo.resolve()}")


# -----------------------------------------------------------------------------
# 6. Resúmenes en consola
# -----------------------------------------------------------------------------

def imprimir_resumen_corpus(datos: pd.DataFrame) -> None:
    """Muestra un resumen breve del corpus cargado."""
    print("\nRESUMEN DEL CORPUS")
    print("-" * 60)
    print(f"Archivo de entrada: {ARCHIVO_ENTRADA.resolve()}")
    print(f"Total de comentarios: {len(datos)}")
    print("Distribución de etiquetas:")
    for etiqueta, cantidad in datos["etiqueta"].value_counts().sort_index().items():
        print(f"- {etiqueta}: {cantidad}")


def imprimir_resumen_reglas(resultados_reglas: pd.DataFrame) -> None:
    """Muestra el resumen de la línea base por reglas sin imprimir los 600 comentarios."""
    total = len(resultados_reglas)
    correctos = (resultados_reglas["evaluacion_reglas"] == "correcto").sum()
    revisar = total - correctos
    exactitud = (correctos / total) * 100 if total else 0

    print("\nLÍNEA BASE POR REGLAS")
    print("-" * 60)
    print(f"Clasificaciones correctas   : {correctos}")
    print(f"Clasificaciones por revisar : {revisar}")
    print(f"Exactitud aproximada        : {exactitud:.2f}%")
    print("Distribución calculada por reglas:")
    for etiqueta, cantidad in resultados_reglas["sentimiento_reglas"].value_counts().sort_index().items():
        print(f"- {etiqueta}: {cantidad}")


# -----------------------------------------------------------------------------
# 7. Función principal
# -----------------------------------------------------------------------------

def main() -> None:
    #Ejecuta el pipeline completo del Corte III
    print("PROYECTO INTEGRADOR DE PLN - CORTE III")
    print("Análisis de sentimiento en reseñas de productos de tienda en línea")
    print("=" * 80)

    nlp = cargar_modelo_spacy()
    stopwords_es = cargar_stopwords_espanol(nlp)
    print(f"Modelo spaCy cargado: {MODELO_SPACY}")
    print(f"Stopwords combinadas: {len(stopwords_es)}")

    datos = cargar_corpus(ARCHIVO_ENTRADA)
    imprimir_resumen_corpus(datos)

    datos_procesados = preprocesar_corpus(datos, nlp, stopwords_es)
    print("\nProcesamiento con spaCy finalizado correctamente.")

    resultados_reglas = generar_linea_base_reglas(datos_procesados)
    guardar_linea_base_reglas(resultados_reglas, ARCHIVO_SALIDA_REGLAS)
    imprimir_resumen_reglas(resultados_reglas)

    generar_archivos_representacion(datos_procesados, CARPETA_SALIDA)
    entrenar_y_evaluar_modelos(datos_procesados, CARPETA_SALIDA_MODELO)

    print("\nPROCESO FINALIZADO")
    print("-" * 60)
    print(f"Resultados por reglas: {ARCHIVO_SALIDA_REGLAS.resolve()}")
    print(f"Carpeta de representación: {CARPETA_SALIDA.resolve()}")
    print(f"Carpeta de modelos: {CARPETA_SALIDA_MODELO.resolve()}")


if __name__ == "__main__":
    main()
