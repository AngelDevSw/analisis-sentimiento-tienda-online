

# Proyecto Integrador de Procesamiento de Lenguaje Natural

## Nombre del proyecto

Análisis de sentimiento en reseñas de productos de tienda en línea.

## Descripción general

Este proyecto desarrolla un pipeline básico de Procesamiento de Lenguaje Natural (PLN) aplicado a comentarios en español de una tienda en línea. El objetivo principal es analizar reseñas de productos y clasificarlas automáticamente en cuatro categorías de sentimiento:

- positivo
- negativo
- neutro
- mixto

El corpus utilizado fue construido de forma controlada e inventada, simulando comentarios reales de comercio electrónico. No contiene datos personales sensibles como nombres de usuarios, correos electrónicos, teléfonos, direcciones, números de pedido o datos bancarios.

## Objetivo del proyecto

Desarrollar un pipeline de Procesamiento de Lenguaje Natural capaz de procesar y clasificar comentarios de usuarios de una tienda en línea en español, asignando automáticamente una de cuatro categorías de sentimiento: positivo, negativo, neutro o mixto.

## Estructura del proyecto

El proyecto contiene los siguientes archivos principales:

- corpus_sentimiento.csv
  Archivo de entrada que contiene el corpus etiquetado con 300 comentarios.

- pipeline.py
  Archivo principal del proyecto. Contiene el código del pipeline de PLN.

- resultados_analisis_sentimiento.csv
  Archivo de salida generado después de ejecutar el pipeline. Contiene los resultados del análisis.

- requirements.txt
  Archivo con las dependencias necesarias para ejecutar el proyecto.

- README.TXT
  Archivo de documentación con la descripción y pasos de ejecución del proyecto.

## Corpus utilizado

El archivo corpus_sentimiento.csv contiene 300 comentarios distribuidos de forma balanceada en cuatro categorías:


| positivo | 75 |
| negativo | 75 |
| neutro | 75 |
| mixto | 75 |
| Total | 300 |

Cada registro del corpus contiene las siguientes columnas:

- id: identificador numérico del comentario.
- texto: comentario original del usuario.
- etiqueta: sentimiento asignado previamente al comentario.

Ejemplo:

id,texto,etiqueta
1,"Excelente producto, llegó antes de lo esperado y en perfectas condiciones.",positivo
152,"El producto es bueno, pero el envío tardó más de lo prometido.",mixto
226,"El artículo es correcto, llegó como se esperaba.",neutro

## Descripción de las etiquetas

### Positivo

Se utiliza cuando el comentario expresa satisfacción, aprobación o una experiencia favorable con el producto, el envío, el vendedor o el proceso de compra.

Ejemplo:

"Excelente producto, llegó antes de lo esperado y en perfectas condiciones."

### Negativo

Se utiliza cuando el comentario expresa inconformidad, queja, falla, mala calidad, retraso o una experiencia desfavorable.

Ejemplo:

"Producto llegó roto, muy decepcionado con la compra."

### Neutro

Se utiliza cuando el comentario es descriptivo o informativo, sin expresar una emoción positiva o negativa clara.

Ejemplo:

"El artículo es correcto, llegó como se esperaba."

### Mixto

Se utiliza cuando el comentario contiene elementos positivos y negativos al mismo tiempo.

Ejemplo:

"El producto es bueno, pero el envío tardó más de lo prometido."

## Funcionamiento del pipeline

El archivo pipeline.py realiza las siguientes etapas:

1. Carga el archivo corpus_sentimiento.csv.
2. Lee cada comentario y su etiqueta original.
3. Limpia el texto convirtiéndolo a minúsculas, eliminando acentos, signos de puntuación y espacios innecesarios.
4. Tokeniza el texto, separándolo en palabras individuales.
5. Elimina palabras vacías o poco informativas mediante una lista de stopwords.
6. Identifica palabras positivas, negativas y neutras.
7. Clasifica el sentimiento calculado como positivo, negativo, neutro o mixto.
8. Compara la etiqueta original con el sentimiento calculado.
9. Genera una evaluación básica: correcto o revisar.
10. Guarda los resultados en el archivo resultados_analisis_sentimiento.csv.

## Requisitos previos

Para ejecutar el proyecto se recomienda tener instalado:

- Python 3.11 o superior
- Miniconda o Anaconda
- Visual Studio Code
- Terminal de macOS, Linux o Windows

## Instalación de dependencias

Desde la carpeta del proyecto, ejecutar:

python -m pip install -r requirements.txt

Nota: aunque el archivo requirements.txt incluye pandas y scikit-learn, el pipeline actual utiliza principalmente librerías estándar de Python como csv, re, unicodedata, pathlib y collections.

## Ejecución del proyecto

Para ejecutar el pipeline, abrir una terminal dentro de la carpeta del proyecto y ejecutar:

python pipeline.py

En macOS también puede ejecutarse con:

python3 pipeline.py

Si se utiliza un entorno de Conda, primero activar el entorno correspondiente y después ejecutar el archivo:

conda activate nombre_del_entorno
python pipeline.py

## Archivo de entrada

El pipeline toma como entrada el archivo:

corpus_sentimiento.csv

Este archivo debe estar en la misma carpeta que pipeline.py.

## Archivo de salida

Después de ejecutar el pipeline, se genera el archivo:

resultados_analisis_sentimiento.csv

Este archivo contiene las siguientes columnas:

- id
- comentario_original
- etiqueta_original
- texto_limpio
- tokens_utiles
- palabras_frecuentes
- sentimiento_calculado
- evaluacion

## Resultados esperados en consola

Al ejecutar el archivo pipeline.py, el programa muestra en consola:

- El id del comentario.
- El comentario original.
- La etiqueta original.
- El texto limpio.
- Los tokens útiles.
- El sentimiento calculado.
- La evaluación del resultado.

Al final se muestra un resumen general con:

- Total de comentarios analizados.
- Clasificaciones correctas.
- Clasificaciones por revisar.
- Exactitud aproximada.
- Distribución de etiquetas originales.
- Distribución de sentimientos calculados.

## Interpretación de la evaluación

La columna evaluacion puede tener dos valores:

- correcto: cuando la etiqueta original coincide con el sentimiento calculado.
- revisar: cuando la etiqueta original no coincide con el sentimiento calculado.

Esta evaluación permite identificar posibles errores del clasificador basado en palabras clave y sirve como base para mejorar el pipeline en etapas posteriores.

## Limitaciones del proyecto

El pipeline actual utiliza una estrategia basada en listas de palabras positivas, negativas y neutras. Por lo tanto, tiene algunas limitaciones:

- No comprende completamente el contexto de una oración.
- Puede clasificar incorrectamente comentarios con negaciones, por ejemplo: "no recomiendo".
- Puede fallar cuando una palabra aparece en diferentes contextos.
- No utiliza todavía modelos de aprendizaje automático avanzados.
- La categoría mixto depende de encontrar palabras positivas y negativas en el mismo comentario.

## Posibles mejoras futuras

En siguientes etapas del proyecto se pueden implementar mejoras como:

- Ampliar el corpus hasta llegar a 1000 documentos.
- Mejorar las listas de palabras positivas, negativas y neutras.
- Agregar una matriz de confusión.
- Calcular métricas como precisión, recall y F1-score.
- Usar técnicas de vectorización como Bag of Words o TF-IDF.
- Entrenar un modelo de clasificación supervisada.
- Comparar el rendimiento de varios modelos de aprendizaje automático.

## Autores

- Angel Montoya
- Manuel Miranda
- Eduardo Taurino

## Fecha

2026-05-29
# Proyecto Integrador de Procesamiento de Lenguaje Natural

Análisis de sentimiento en reseñas de productos de tienda en línea

Ambiente obligatorio:
- Python 3.11 o superior
- Miniconda o Anaconda
- Visual Studio Code

Instalación de dependencias:
1. Abrir la carpeta del proyecto en Visual Studio Code.
2. Abrir una terminal dentro de la carpeta del proyecto.
3. Activar el ambiente de trabajo:
   conda activate pln311
4. Instalar las librerías necesarias:
   python -m pip install -r requirements.txt

Ejecución:
1. Abrir esta carpeta en Visual Studio Code.
2. Verificar que el archivo corpus_sentimiento.csv esté en la misma carpeta que pipeline.py.
3. Activar el ambiente:
   conda activate pln311
4. Ejecutar el pipeline principal:
   python pipeline.py

En macOS también puede ejecutarse con:
   python3 pipeline.py

Archivo de entrada:
- corpus_sentimiento.csv

Archivo principal:
- pipeline.py

Archivos y carpetas generadas:
- resultados_analisis_sentimiento.csv
- salida/01_preprocesamiento.csv
- salida/02_matriz_bow.csv
- salida/03_matriz_tf.csv
- salida/04_matriz_tfidf.csv
- salida/05_top_terminos_tfidf.csv
- salida/06_similitud_documentos.csv
- salida_modelo/01_datos_preprocesados.csv
- salida_modelo/02_metricas_modelos.csv
- salida_modelo/03_vocabulario_tfidf.csv
- salida_modelo/04_matriz_tfidf.csv
- salida_modelo/05_matriz_confusion_mejor_modelo.csv
- salida_modelo/06_predicciones_prueba.csv
- salida_modelo/07_reporte_clasificacion_mejor_modelo.csv

Objetivo:
Desarrollar un pipeline de Procesamiento de Lenguaje Natural capaz de procesar comentarios de usuarios de una tienda en línea en español, limpiar el texto, representar los datos de forma numérica y evaluar modelos supervisados para clasificar automáticamente el sentimiento como positivo, negativo, neutro o mixto.

Descripción general:
Este proyecto analiza reseñas de productos de una tienda en línea usando técnicas de Procesamiento de Lenguaje Natural. El corpus fue construido de forma controlada e inventada, simulando opiniones reales de comercio electrónico. No contiene datos personales sensibles como nombres, correos electrónicos, teléfonos, direcciones, números de pedido o datos bancarios.

Corpus utilizado:
El archivo corpus_sentimiento.csv contiene 300 comentarios distribuidos de forma balanceada en cuatro categorías:

Etiqueta    Cantidad    Porcentaje
positivo    75          25%
negativo    75          25%
neutro      75          25%
mixto       75          25%
Total       300         100%

Cada registro del corpus contiene tres columnas:
- id: identificador numérico del comentario.
- texto: comentario original del usuario.
- etiqueta: sentimiento asignado previamente al comentario.

Ejemplo:
id,texto,etiqueta
1,"Excelente producto, llegó antes de lo esperado y en perfectas condiciones.",positivo
151,"El producto llegó en el tiempo estimado, sin más novedades.",neutro
226,"El producto funciona muy bien, pero el envío tardó más de lo esperado.",mixto

Descripción de las etiquetas:

Positivo:
Se utiliza cuando el comentario expresa satisfacción, aprobación o una experiencia favorable con el producto, el envío, el vendedor o el proceso de compra.

Negativo:
Se utiliza cuando el comentario expresa inconformidad, falla, mala calidad, retraso o una experiencia desfavorable.

Neutro:
Se utiliza cuando el comentario es descriptivo o informativo, sin expresar una emoción positiva o negativa clara.

Mixto:
Se utiliza cuando el comentario contiene elementos positivos y negativos al mismo tiempo.

Funcionamiento general del pipeline:
1. Carga el archivo corpus_sentimiento.csv.
2. Lee cada comentario y su etiqueta original.
3. Limpia el texto convirtiéndolo a minúsculas y eliminando acentos, signos de puntuación y espacios innecesarios.
4. Tokeniza el texto, separándolo en palabras individuales.
5. Elimina stopwords o palabras vacías que aportan poco significado.
6. Identifica palabras positivas, negativas y neutras mediante listas definidas en el código.
7. Calcula una clasificación inicial basada en reglas.
8. Compara la etiqueta original con el sentimiento calculado.
9. Guarda los resultados en resultados_analisis_sentimiento.csv.
10. Genera archivos de representación numérica en la carpeta salida.
11. Entrena y evalúa modelos supervisados en la carpeta salida_modelo.

Primera salida: resultados_analisis_sentimiento.csv
Este archivo contiene la clasificación inicial basada en reglas simples de palabras clave. Incluye las columnas id, comentario_original, etiqueta_original, texto_limpio, tokens_utiles, palabras_frecuentes, sentimiento_calculado y evaluacion.

La columna evaluacion puede tener dos valores:
- correcto: cuando la etiqueta original coincide con el sentimiento calculado.
- revisar: cuando la etiqueta original no coincide con el sentimiento calculado.

Carpeta salida:
La carpeta salida contiene los archivos relacionados con el preprocesamiento y la representación numérica del texto. Estos archivos permiten observar cómo el texto se transforma en datos numéricos mediante técnicas como Bag of Words, TF y TF-IDF. También se calcula la similitud entre documentos usando similitud coseno.

Carpeta salida_modelo:
La carpeta salida_modelo contiene los resultados de la etapa de Machine Learning supervisado. En esta etapa se entrena y evalúa el corpus usando dos algoritmos:
- Multinomial Naive Bayes
- Regresión Logística

El pipeline compara ambos modelos usando métricas como accuracy, precision_macro, recall_macro y f1_macro. Después selecciona automáticamente el mejor modelo con base en el valor de f1_macro.

Librerías necesarias:
El proyecto utiliza librerías estándar de Python y dos librerías externas.

Librerías estándar:
- csv
- re
- unicodedata
- pathlib
- collections
- math

Librerías externas:
- pandas
- scikit-learn

El archivo requirements.txt debe contener:
pandas
scikit-learn

Resultados esperados en consola:
Al ejecutar el pipeline, se muestra información como comentarios procesados, texto limpio, tokens útiles, sentimiento calculado, evaluación de la clasificación inicial, resumen general de aciertos y revisiones, archivos generados en la carpeta salida, modelo supervisado seleccionado, métricas principales y archivos generados en la carpeta salida_modelo.

Limitaciones actuales:
- La primera clasificación se basa en reglas simples de palabras clave.
- Algunas palabras pueden cambiar de significado según el contexto.
- Las negaciones pueden afectar la clasificación, por ejemplo: "no recomiendo".
- El corpus es inventado y controlado, por lo que no representa toda la variedad del lenguaje real.
- Los modelos supervisados dependen del tamaño y calidad del corpus disponible.

Posibles mejoras futuras:
- Ampliar el corpus a más comentarios.
- Mejorar las listas de palabras positivas, negativas y neutras.
- Probar otros modelos de clasificación.
- Ajustar parámetros de TF-IDF.
- Aplicar validación cruzada.
- Generar gráficas para métricas y matriz de confusión.
- Comparar el rendimiento entre modelos tradicionales y modelos más avanzados.

Autores:
- Angel Montoya
- Manuel Miranda
- Eduardo Taurino

Fecha:
2026-05-29