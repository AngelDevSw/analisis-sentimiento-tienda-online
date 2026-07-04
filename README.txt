# Proyecto Integrador de Procesamiento de Lenguaje Natural

## Nombre del proyecto

Análisis de sentimiento en reseñas de productos de tienda en línea.

## Descripción

Este proyecto desarrolla un pipeline de Procesamiento de Lenguaje Natural para clasificar comentarios de usuarios de una tienda en línea en español.

Las clases de sentimiento son:

- positivo
- negativo
- neutro
- mixto

El corpus fue construido de forma controlada y no contiene datos personales sensibles.

## Corpus

Archivo de entrada:

- corpus_sentimiento.csv

Estructura del archivo:

- id
- texto
- etiqueta

Distribución del corpus:

- positivo: 150 comentarios
- negativo: 150 comentarios
- neutro: 150 comentarios
- mixto: 150 comentarios
- total: 600 comentarios

## Archivo principal

- pipeline.py

Este archivo ejecuta el flujo completo del proyecto.

## Tecnologías utilizadas

- Python 3.11
- pandas
- scikit-learn
- NLTK
- spaCy
- Modelo de spaCy: es_core_news_sm

## Instalación

Activar el entorno de trabajo:

```bash
conda activate pln311
```

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

Descargar el modelo de spaCy en español:

```bash
python -m spacy download es_core_news_sm
```

## Ejecución

Ejecutar el pipeline:

```bash
python pipeline.py
```

## Salidas generadas

Archivo principal de línea base por reglas:

- resultados_analisis_sentimiento.csv

Carpeta `salida`:

- 01_preprocesamiento.csv
- 02_matriz_bow.csv
- 03_matriz_tf.csv
- 04_matriz_tfidf.csv
- 05_top_terminos_tfidf.csv
- 06_similitud_documentos.csv

Carpeta `salida_modelo`:

- 01_datos_preprocesados_spacy.csv
- 02_metricas_modelos.csv
- 03_vocabulario_tfidf.csv
- 04_matriz_tfidf.csv
- 05_matriz_confusion_mejor_modelo.csv
- 06_predicciones_prueba.csv
- 07_reporte_clasificacion_mejor_modelo.csv
- 08_errores_modelo.csv

## Modelos evaluados

- MultinomialNB
- LogisticRegression
- LinearSVC
- RandomForestClassifier

El mejor modelo se selecciona usando la métrica F1 macro.

## Autores

- Miguel Ángel Montoya Cerro
- Manuel Miranda Astorga
- Eduardo Taurino Martínez Morales

## Repositorio

https://github.com/AngelDevSw/analisis-sentimiento-tienda-online