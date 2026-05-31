# Procesamiento de Lenguaje Natural - Proyecto integrador 
# Python 3.11 + Miniconda + Visual Studio Code
# Objetivo: leer comentarios, limpiar texto, tokenizar, clasificar y guardar resultados.
# nombre del proyecto: Análisis de sentimiento en reseñas de productos de tienda en línea
# Autor: Angel Montoya, Manuel Miranda, Eduardo Taurino
# Fecha: 2026-05-29

# 1 Importar librerías
from pathlib import Path # Manejo de rutas de archivos
import csv # Lectura y escritura de archivos CSV
import re # Expresiones regulares para limpieza de texto
import unicodedata # Normalización de texto para quitar acentos
from collections import Counter # Contar frecuencia de palabras
import math # Operaciones matemáticas para TF, IDF, TF-IDF y similitud coseno

import pandas as pd # Manejo de datos en tablas para el modelo supervisado
from sklearn.feature_extraction.text import TfidfVectorizer # Representación TF-IDF con scikit-learn
from sklearn.model_selection import train_test_split # División de datos en entrenamiento y prueba
from sklearn.naive_bayes import MultinomialNB # Algoritmo Naive Bayes Multinomial
from sklearn.linear_model import LogisticRegression # Algoritmo de Regresión Logística
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # Métricas de evaluación
from sklearn.metrics import classification_report, confusion_matrix # Reporte y matriz de confusión

ARCHIVO_ENTRADA = Path("corpus_sentimiento.csv")
ARCHIVO_SALIDA = Path("resultados_analisis_sentimiento.csv")
CARPETA_SALIDA = Path("salida")
CARPETA_SALIDA_MODELO = Path("salida_modelo")

STOPWORDS = {
    # Artículos y determinantes
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "mi", "mis", "tu", "tus", "su", "sus",

    # Preposiciones y conectores
    "a", "ante", "bajo", "con", "contra", "de", "del", "desde", "durante",
    "en", "entre", "hacia", "hasta", "para", "por", "segun", "sin", "sobre",
    "tras", "y", "e", "o", "u", "pero", "aunque", "porque", "que", "como",
    "cuando", "donde", "si", "ni", "mas",

    # Pronombres y palabras frecuentes
    "me", "te", "se", "lo", "le", "les", "nos", "yo", "tu", "ella", "el",
    "ellos", "ellas", "esto", "eso", "aquello", "algo", "nada", "todo", "todos",
    "todas", "algun", "alguna", "ningun", "ninguna",

    # Verbos auxiliares o muy generales
    "es", "son", "fue", "era", "esta", "estan", "estaba", "estaban", "ser",
    "estar", "haber", "hay", "ha", "han", "he", "hace", "hacer", "tiene",
    "tienen", "tenia", "tener", "viene", "llego", "llegar",

    # Intensificadores o expresiones poco informativas por sí solas
    "muy", "mucho", "mucha", "muchos", "muchas", "mas", "menos", "tan",
    "tanto", "bastante", "cada", "vez", "veces", "dia", "dias"
}

PALABRAS_POSITIVAS = {
    "excelente", "perfecto", "perfecta", "perfectamente", "buena", "buen", "bueno",
    "buenisima", "buenisimo", "genial", "fantastico", "increible", "maravilla",
    "encanto", "encantado", "feliz", "contento", "satisfecho", "satisfactoria",
    "recomiendo", "recomendado", "recomendable", "confiable", "autentico", "original",
    "premium", "elegante", "resistente", "solido", "robusto", "duradero", "funcional",
    "practico", "facil", "rapido", "rapidisimo", "puntual", "impecable",
    "correctamente", "cumple", "supero", "mejor", "calidad", "accesible",
    "justo", "atento", "servicio", "primera", "protegido", "sellado"
}

PALABRAS_NEGATIVAS = {
    "danado", "danada", "aplastada", "malas", "mala", "malo", "malisima", "malisimo",
    "pesima", "pesimo", "decepcion", "decepcionado", "decepcionante", "terrible",
    "horrible", "barata", "barato", "fragil", "endeble", "defectuoso", "incompleto",
    "incompleta", "equivocada", "enganosa", "estafa", "estafado", "peligroso",
    "inaceptable", "vergonzoso", "grosero", "fallas", "falla", "fallo", "roto",
    "rompio", "desmorona", "oxido", "rayaduras", "rayado", "manchas", "golpeado",
    "abierta", "usado", "humedad", "quimicos", "ruido", "pixelada", "baja",
    "debil", "corto", "corta", "dificil", "tardo", "nunca", "devolver",
    "devolucion", "faltaban", "faltaron", "error", "no", "nada", "sin", "peor"
}

PALABRAS_NEUTRAS = {
    "producto", "articulo", "pedido", "paquete", "caja", "empaque", "embalaje",
    "vendedor", "compra", "entrega", "envio", "precio", "calidad", "material",
    "color", "talla", "tamano", "foto", "fotos", "imagen", "imagenes", "descripcion",
    "ficha", "pagina", "anuncio", "caracteristicas", "especificaciones",
    "funcion", "funciona", "uso", "diario", "basico", "basica", "normal",
    "estandar", "promedio", "simple", "sencillo", "sencilla", "correcto", "correcta",
    "adecuado", "adecuada", "aceptable", "condiciones", "fecha", "tiempo",
    "rango", "proceso", "categoria", "mercado", "observaciones", "comentarios"
}


# 2 Definir funciones para cada etapa del pipeline de análisis de sentimiento

# 2.1 Cargar comentarios desde el archivo CSV
def cargar_comentarios(ruta: Path) -> list[dict[str, str]]:
    """Lee el archivo CSV del proyecto y devuelve una lista de comentarios."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {ruta}")

    comentarios = []

    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)

        for fila in lector:
            comentarios.append({
                "id": fila.get("id", "").strip(),
                "texto": fila.get("texto", "").strip(),
                "etiqueta_original": fila.get("etiqueta", "").strip().lower(),
            })

    return [comentario for comentario in comentarios if comentario["texto"]]

# 2.2 Limpiar el texto de cada comentario
def quitar_acentos(texto: str) -> str:
    normalizado = unicodedata.normalize("NFD", texto)
    return "".join(c for c in normalizado if unicodedata.category(c) != "Mn")

# Eliminar acentos, convertir a minúsculas, eliminar signos de puntuación y espacios extra
def limpiar_texto(texto: str) -> str:
    """Normaliza el texto: minúsculas, sin acentos, sin signos y sin espacios extra."""
    texto = texto.lower()
    texto = quitar_acentos(texto)
    texto = re.sub(r"[^a-zñ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# 2.3 Tokenizar el texto limpio en palabras individuales
def tokenizar(texto_limpio: str) -> list[str]:
    """Divide el texto limpio en palabras individuales."""
    return texto_limpio.split()

# 2.4 Eliminar palabras vacías y tokens muy cortos para conservar términos útiles
def quitar_stopwords(tokens: list[str]) -> list[str]:
    """Elimina palabras vacías y tokens muy cortos para conservar términos útiles."""
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]

# 2.5 Clasificar el sentimiento del comentario basado en la presencia de palabras positivas, negativas y neutras
def clasificar_sentimiento(tokens: list[str]) -> str:
    """Clasifica el sentimiento con base en palabras positivas y negativas."""
    positivos = sum(1 for token in tokens if token in PALABRAS_POSITIVAS)
    negativos = sum(1 for token in tokens if token in PALABRAS_NEGATIVAS)
    neutros = sum(1 for token in tokens if token in PALABRAS_NEUTRAS)

    if positivos > 0 and negativos > 0:
        return "mixto"
    if positivos > negativos:
        return "positivo"
    if negativos > positivos:
        return "negativo"
    if neutros > 0:
        return "neutro"
    return "neutro"

# 2.6 Evaluar la clasificación comparando el sentimiento calculado con la etiqueta original del dataset
def evaluar_clasificacion(etiqueta_original: str, sentimiento_calculado: str) -> str:
    """Compara la etiqueta original del dataset con el sentimiento calculado."""
    if etiqueta_original == sentimiento_calculado:
        return "correcto"
    return "revisar"

# 2.7 Procesar cada comentario del proyecto y generar los resultados del análisis
def analizar_comentarios(comentarios: list[dict[str, str]]) -> list[dict[str, str]]:
    """Procesa cada comentario del proyecto y genera los resultados del análisis."""
    resultados = []

    for comentario in comentarios:
        texto_original = comentario["texto"]
        texto_limpio = limpiar_texto(texto_original)
        tokens = tokenizar(texto_limpio)
        tokens_utiles = quitar_stopwords(tokens)
        frecuencia = Counter(tokens_utiles)
        sentimiento = clasificar_sentimiento(tokens_utiles)
        evaluacion = evaluar_clasificacion(comentario["etiqueta_original"], sentimiento)

        resultados.append({
            "id": comentario["id"],
            "comentario_original": texto_original,
            "etiqueta_original": comentario["etiqueta_original"],
            "texto_limpio": texto_limpio,
            "tokens_utiles": ", ".join(tokens_utiles),
            "palabras_frecuentes": ", ".join([f"{palabra}:{conteo}" for palabra, conteo in frecuencia.most_common(5)]),
            "sentimiento_calculado": sentimiento,
            "evaluacion": evaluacion,
        })

    return resultados

# 2.8 Guardar los resultados del análisis en un nuevo archivo CSV y mostrar un resumen de los resultados en consola
def guardar_resultados(resultados: list[dict[str, str]], ruta: Path) -> None:
    """Guarda los resultados del análisis en un archivo CSV."""
    campos = [
        "id",
        "comentario_original",
        "etiqueta_original",
        "texto_limpio",
        "tokens_utiles",
        "palabras_frecuentes",
        "sentimiento_calculado",
        "evaluacion",
    ]

    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)


# 2.9 Generar archivos de representación numérica: preprocesamiento, BoW, TF, TF-IDF y similitud

def obtener_tokens_desde_resultado(fila: dict[str, str]) -> list[str]:
    """Convierte la columna de tokens útiles en una lista de tokens."""
    tokens = fila.get("tokens_utiles", "")
    if not tokens:
        return []
    return [token.strip() for token in tokens.split(",") if token.strip()]


def construir_vocabulario(resultados: list[dict[str, str]]) -> list[str]:
    """Construye un vocabulario ordenado a partir de los tokens útiles."""
    vocabulario = set()

    for fila in resultados:
        vocabulario.update(obtener_tokens_desde_resultado(fila))

    return sorted(vocabulario)


def calcular_bow(tokens: list[str], vocabulario: list[str]) -> dict[str, int]:
    """Calcula la representación Bag of Words para un documento."""
    frecuencia = Counter(tokens)
    return {termino: frecuencia.get(termino, 0) for termino in vocabulario}


def calcular_tf(tokens: list[str], vocabulario: list[str]) -> dict[str, float]:
    """Calcula la frecuencia de término normalizada para un documento."""
    frecuencia = Counter(tokens)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return {termino: 0.0 for termino in vocabulario}

    return {termino: frecuencia.get(termino, 0) / total_tokens for termino in vocabulario}


def calcular_idf(documentos_tokens: list[list[str]], vocabulario: list[str]) -> dict[str, float]:
    """Calcula el IDF suavizado para cada término del vocabulario."""
    total_documentos = len(documentos_tokens)
    idf = {}

    for termino in vocabulario:
        documentos_con_termino = sum(1 for tokens in documentos_tokens if termino in set(tokens))
        idf[termino] = math.log((1 + total_documentos) / (1 + documentos_con_termino)) + 1

    return idf


def calcular_tfidf(tf: dict[str, float], idf: dict[str, float], vocabulario: list[str]) -> dict[str, float]:
    """Calcula la representación TF-IDF para un documento."""
    return {termino: tf[termino] * idf[termino] for termino in vocabulario}


def similitud_coseno(vector_a: dict[str, float], vector_b: dict[str, float], vocabulario: list[str]) -> float:
    """Calcula la similitud coseno entre dos documentos representados con TF-IDF."""
    producto_punto = sum(vector_a[termino] * vector_b[termino] for termino in vocabulario)
    norma_a = math.sqrt(sum(vector_a[termino] ** 2 for termino in vocabulario))
    norma_b = math.sqrt(sum(vector_b[termino] ** 2 for termino in vocabulario))

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return producto_punto / (norma_a * norma_b)


def guardar_csv_generico(ruta: Path, campos: list[str], filas: list[dict[str, str | int | float]]) -> None:
    """Guarda cualquier lista de diccionarios en un archivo CSV."""
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def generar_archivos_representacion(resultados: list[dict[str, str]], carpeta_salida: Path) -> None:
    """Genera los archivos de salida para representar numéricamente el corpus."""
    carpeta_salida.mkdir(exist_ok=True)

    documentos_tokens = [obtener_tokens_desde_resultado(fila) for fila in resultados]
    vocabulario = construir_vocabulario(resultados)
    idf = calcular_idf(documentos_tokens, vocabulario)

    filas_preprocesamiento = []
    filas_bow = []
    filas_tf = []
    filas_tfidf = []
    filas_top_tfidf = []
    vectores_tfidf = []

    for indice, fila in enumerate(resultados):
        tokens = documentos_tokens[indice]
        bow = calcular_bow(tokens, vocabulario)
        tf = calcular_tf(tokens, vocabulario)
        tfidf = calcular_tfidf(tf, idf, vocabulario)
        vectores_tfidf.append(tfidf)

        filas_preprocesamiento.append({
            "id": fila["id"],
            "texto_original": fila["comentario_original"],
            "etiqueta_original": fila["etiqueta_original"],
            "texto_limpio": fila["texto_limpio"],
            "tokens_utiles": fila["tokens_utiles"],
        })

        filas_bow.append({
            "id": fila["id"],
            "etiqueta_original": fila["etiqueta_original"],
            **bow,
        })

        filas_tf.append({
            "id": fila["id"],
            "etiqueta_original": fila["etiqueta_original"],
            **{termino: round(valor, 6) for termino, valor in tf.items()},
        })

        filas_tfidf.append({
            "id": fila["id"],
            "etiqueta_original": fila["etiqueta_original"],
            **{termino: round(valor, 6) for termino, valor in tfidf.items()},
        })

        top_terminos = sorted(tfidf.items(), key=lambda item: item[1], reverse=True)[:5]
        filas_top_tfidf.append({
            "id": fila["id"],
            "etiqueta_original": fila["etiqueta_original"],
            "texto_original": fila["comentario_original"],
            "top_terminos_tfidf": ", ".join([f"{termino}:{valor:.4f}" for termino, valor in top_terminos if valor > 0]),
        })

    similitudes = []
    total_documentos = len(resultados)

    for i in range(total_documentos):
        for j in range(i + 1, total_documentos):
            similitud = similitud_coseno(vectores_tfidf[i], vectores_tfidf[j], vocabulario)
            similitudes.append({
                "id_documento_1": resultados[i]["id"],
                "etiqueta_documento_1": resultados[i]["etiqueta_original"],
                "id_documento_2": resultados[j]["id"],
                "etiqueta_documento_2": resultados[j]["etiqueta_original"],
                "similitud_coseno": round(similitud, 6),
            })

    similitudes_ordenadas = sorted(similitudes, key=lambda fila: fila["similitud_coseno"], reverse=True)[:50]

    guardar_csv_generico(
        carpeta_salida / "01_preprocesamiento.csv",
        ["id", "texto_original", "etiqueta_original", "texto_limpio", "tokens_utiles"],
        filas_preprocesamiento,
    )
    guardar_csv_generico(
        carpeta_salida / "02_matriz_bow.csv",
        ["id", "etiqueta_original", *vocabulario],
        filas_bow,
    )
    guardar_csv_generico(
        carpeta_salida / "03_matriz_tf.csv",
        ["id", "etiqueta_original", *vocabulario],
        filas_tf,
    )
    guardar_csv_generico(
        carpeta_salida / "04_matriz_tfidf.csv",
        ["id", "etiqueta_original", *vocabulario],
        filas_tfidf,
    )
    guardar_csv_generico(
        carpeta_salida / "05_top_terminos_tfidf.csv",
        ["id", "etiqueta_original", "texto_original", "top_terminos_tfidf"],
        filas_top_tfidf,
    )
    guardar_csv_generico(
        carpeta_salida / "06_similitud_documentos.csv",
        ["id_documento_1", "etiqueta_documento_1", "id_documento_2", "etiqueta_documento_2", "similitud_coseno"],
        similitudes_ordenadas,
    )

    print("\nREPRESENTACIÓN NUMÉRICA DEL CORPUS")
    print("-" * 50)
    print(f"Documentos procesados: {len(resultados)}")
    print(f"Tamaño del vocabulario: {len(vocabulario)}")
    print(f"Archivos generados en: {carpeta_salida.resolve()}")
    print("\nTop 5 similitudes:")
    for fila in similitudes_ordenadas[:5]:
        print(
            f"- Documento {fila['id_documento_1']} y documento {fila['id_documento_2']}: "
            f"{fila['similitud_coseno']:.4f}"
        )

# 2.10 Entrenar y evaluar modelos supervisados de Machine Learning

def entrenar_y_evaluar_modelos(comentarios: list[dict[str, str]], carpeta_salida_modelo: Path) -> None:
    """Entrena modelos supervisados para clasificar sentimiento y guarda archivos de evaluación."""
    carpeta_salida_modelo.mkdir(exist_ok=True)

    datos = pd.DataFrame(comentarios)
    datos = datos.rename(columns={"texto": "texto_original"})
    datos["texto_limpio"] = datos["texto_original"].apply(limpiar_texto)

    datos_preprocesados = datos[["id", "texto_original", "etiqueta_original", "texto_limpio"]]
    datos_preprocesados.to_csv(carpeta_salida_modelo / "01_datos_preprocesados.csv", index=False, encoding="utf-8-sig")

    x = datos["texto_limpio"]
    y = datos["etiqueta_original"]

    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    vectorizador = TfidfVectorizer()
    x_entrenamiento_tfidf = vectorizador.fit_transform(x_entrenamiento)
    x_prueba_tfidf = vectorizador.transform(x_prueba)

    modelos = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

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

    metricas_df = pd.DataFrame(metricas)
    metricas_df.to_csv(carpeta_salida_modelo / "02_metricas_modelos.csv", index=False, encoding="utf-8-sig")

    mejor_modelo_nombre = metricas_df.sort_values("f1_macro", ascending=False).iloc[0]["modelo"]
    mejor_modelo = modelos[mejor_modelo_nombre]
    mejores_predicciones = predicciones_por_modelo[mejor_modelo_nombre]

    vocabulario_df = pd.DataFrame({"termino": vectorizador.get_feature_names_out()})
    vocabulario_df.to_csv(carpeta_salida_modelo / "03_vocabulario_tfidf.csv", index=False, encoding="utf-8-sig")

    matriz_tfidf_completa = vectorizador.transform(datos["texto_limpio"])
    matriz_tfidf_df = pd.DataFrame(
        matriz_tfidf_completa.toarray(),
        columns=vectorizador.get_feature_names_out(),
    )
    matriz_tfidf_df.insert(0, "etiqueta_original", datos["etiqueta_original"])
    matriz_tfidf_df.insert(0, "id", datos["id"])
    matriz_tfidf_df.to_csv(carpeta_salida_modelo / "04_matriz_tfidf.csv", index=False, encoding="utf-8-sig")

    etiquetas_ordenadas = sorted(y.unique())
    matriz_confusion = confusion_matrix(y_prueba, mejores_predicciones, labels=etiquetas_ordenadas)
    matriz_confusion_df = pd.DataFrame(
        matriz_confusion,
        index=[f"real_{etiqueta}" for etiqueta in etiquetas_ordenadas],
        columns=[f"predicho_{etiqueta}" for etiqueta in etiquetas_ordenadas],
    )
    matriz_confusion_df.to_csv(carpeta_salida_modelo / "05_matriz_confusion_mejor_modelo.csv", encoding="utf-8-sig")

    predicciones_prueba_df = pd.DataFrame({
        "texto": x_prueba.values,
        "etiqueta_real": y_prueba.values,
        "etiqueta_predicha": mejores_predicciones,
        "modelo": mejor_modelo_nombre,
    })
    predicciones_prueba_df.to_csv(carpeta_salida_modelo / "06_predicciones_prueba.csv", index=False, encoding="utf-8-sig")

    reporte = classification_report(
        y_prueba,
        mejores_predicciones,
        labels=etiquetas_ordenadas,
        output_dict=True,
        zero_division=0,
    )
    reporte_df = pd.DataFrame(reporte).transpose()
    reporte_df.to_csv(carpeta_salida_modelo / "07_reporte_clasificacion_mejor_modelo.csv", encoding="utf-8-sig")

    print("\nMODELO SUPERVISADO DE MACHINE LEARNING")
    print("-" * 50)
    print(f"Modelo seleccionado: {mejor_modelo_nombre}")
    print("\nMétricas principales:")
    mejor_fila = metricas_df[metricas_df["modelo"] == mejor_modelo_nombre].iloc[0]
    print(f"- Accuracy        : {mejor_fila['accuracy']:.4f}")
    print(f"- Precision macro : {mejor_fila['precision_macro']:.4f}")
    print(f"- Recall macro    : {mejor_fila['recall_macro']:.4f}")
    print(f"- F1 macro        : {mejor_fila['f1_macro']:.4f}")
    print(f"\nArchivos generados en: {carpeta_salida_modelo.resolve()}")

# 2.11 Funciones para imprimir resultados y resumen en consola
def imprimir_resultados(resultados: list[dict[str, str]]) -> None:
    """Muestra en consola los resultados principales del análisis."""
    for fila in resultados:
        print("=" * 100)
        print(f"ID                  : {fila['id']}")
        print(f"Comentario original : {fila['comentario_original']}")
        print(f"Etiqueta original   : {fila['etiqueta_original']}")
        print(f"Texto limpio        : {fila['texto_limpio']}")
        print(f"Tokens útiles       : {fila['tokens_utiles']}")
        print(f"Sentimiento calculado: {fila['sentimiento_calculado']}")
        print(f"Evaluación          : {fila['evaluacion']}")
    print("=" * 100)

# 2.12 Resumen general de etiquetas, sentimientos y aciertos
def imprimir_resumen(resultados: list[dict[str, str]]) -> None:
    """Imprime un resumen general de etiquetas, sentimientos y aciertos."""
    total = len(resultados)
    correctos = sum(1 for fila in resultados if fila["evaluacion"] == "correcto")
    revisar = total - correctos

    conteo_etiquetas = Counter(fila["etiqueta_original"] for fila in resultados)
    conteo_sentimientos = Counter(fila["sentimiento_calculado"] for fila in resultados)

    print("\nRESUMEN DEL ANÁLISIS")
    print("-" * 50)
    print(f"Total de comentarios analizados: {total}")
    print(f"Clasificaciones correctas      : {correctos}")
    print(f"Clasificaciones por revisar    : {revisar}")

    if total > 0:
        exactitud = (correctos / total) * 100
        print(f"Exactitud aproximada           : {exactitud:.2f}%")

    print("\nDistribución de etiquetas originales:")
    for etiqueta, conteo in conteo_etiquetas.items():
        print(f"- {etiqueta}: {conteo}")

    print("\nDistribución de sentimientos calculados:")
    for sentimiento, conteo in conteo_sentimientos.items():
        print(f"- {sentimiento}: {conteo}")

# 3 Función principal para ejecutar el pipeline completo
def main() -> None:
    comentarios = cargar_comentarios(ARCHIVO_ENTRADA)
    resultados = analizar_comentarios(comentarios)
    imprimir_resultados(resultados)
    imprimir_resumen(resultados)
    guardar_resultados(resultados, ARCHIVO_SALIDA)
    generar_archivos_representacion(resultados, CARPETA_SALIDA)
    entrenar_y_evaluar_modelos(comentarios, CARPETA_SALIDA_MODELO)
    print(f"\nResultados guardados en: {ARCHIVO_SALIDA.resolve()}")

# Ejecutar la función principal si este script se ejecuta directamente
if __name__ == "__main__":
    main()
