@echo off
REM ============================================================
REM Script para generar la estructura del proyecto "avocado-regression-project"
REM Cada carpeta incluye un README.md con su descripción
REM ============================================================

set PROJECT=avocado-regression-project

echo Creando estructura del proyecto %PROJECT% ...
mkdir %PROJECT%
cd %PROJECT%

REM ==== Carpetas principales ====
mkdir data
mkdir data\raw
mkdir data\processed
mkdir notebooks
mkdir models
mkdir models\linear_regression
mkdir models\random_forest
mkdir models\neural_network
mkdir models\decision_tree
mkdir results
mkdir results\plots
mkdir docs

REM ==== Archivos base ====
echo # Proyecto de Regresión de Aguacates > README.md
echo Proyecto para predecir el precio de los aguacates usando diferentes modelos de regresión. >> README.md
echo. >> README.md
echo Cada integrante trabaja en un modelo diferente dentro de la carpeta *models/*. >> README.md
echo. >> README.md
echo - César → Regresión Lineal >> README.md
echo - Joseph → Random Forest >> README.md
echo - Cuéllar → Red Neuronal >> README.md
echo - Samuel → Árbol de Decisión >> README.md

echo pandas>>requirements.txt
echo numpy>>requirements.txt
echo matplotlib>>requirements.txt
echo seaborn>>requirements.txt
echo scikit-learn>>requirements.txt
echo tensorflow>>requirements.txt

echo /__pycache__/ > .gitignore
echo /.vscode/ >> .gitignore
echo /.ipynb_checkpoints/ >> .gitignore
echo /data/raw/* >> .gitignore

REM ==== README para cada carpeta ====

REM data/raw
echo # data/raw > data\raw\README.md
echo Contiene el dataset original sin modificar. >> data\raw\README.md
echo No se debe alterar ni subir datasets externos aquí. >> data\raw\README.md

REM data/processed
echo # data/processed > data\processed\README.md
echo Incluye los datos limpios y normalizados listos para el modelado. >> data\processed\README.md
echo Generados en la etapa de limpieza (punto 2 del proyecto). >> data\processed\README.md

REM notebooks
echo # notebooks > notebooks\README.md
echo Contiene los notebooks de análisis exploratorio y limpieza de datos. >> notebooks\README.md
echo - 1_analisis_exploratorio.ipynb: visualizaciones, correlaciones, outliers. >> notebooks\README.md
echo - 2_limpieza_normalizacion.ipynb: tratamiento de nulos y normalización. >> notebooks\README.md

REM models/linear_regression
echo # models/linear_regression > models\linear_regression\README.md
echo Modelo desarrollado por César. >> models\linear_regression\README.md
echo Implementa una regresión lineal sobre el dataset procesado. >> models\linear_regression\README.md
echo Evaluar con métricas MSE y R². >> models\linear_regression\README.md

REM models/random_forest
echo # models/random_forest > models\random_forest\README.md
echo Modelo desarrollado por Joseph. >> models\random_forest\README.md
echo Implementa un Random Forest Regressor. >> models\random_forest\README.md
echo Ajustar hiperparámetros y comparar resultados. >> models\random_forest\README.md

REM models/neural_network
echo # models/neural_network > models\neural_network\README.md
echo Modelo desarrollado por Cuéllar. >> models\neural_network\README.md
echo Implementa una red neuronal simple usando TensorFlow o Keras. >> models\neural_network\README.md
echo Evaluar desempeño y posibles mejoras. >> models\neural_network\README.md

REM models/decision_tree
echo # models/decision_tree > models\decision_tree\README.md
echo Modelo desarrollado por Samuel. >> models\decision_tree\README.md
echo Implementa un Decision Tree Regressor. >> models\decision_tree\README.md
echo Comparar los resultados frente a los otros modelos. >> models\decision_tree\README.md

REM results
echo # results > results\README.md
echo Carpeta con resultados de los modelos. >> results\README.md
echo Incluir métricas comparativas (MSE, R²) y gráficos finales. >> results\README.md

REM results/plots
echo # results/plots > results\plots\README.md
echo Contiene las gráficas de desempeño, comparaciones y visualizaciones finales. >> results\plots\README.md

REM docs
echo # docs > docs\README.md
echo Contiene los documentos finales del proyecto. >> docs\README.md
echo - informe_final.pdf: análisis completo del proyecto. >> docs\README.md
echo - presentacion_final.pptx: resumen visual para exposición. >> docs\README.md

echo.
echo ✅ Estructura del proyecto creada exitosamente en %PROJECT%.
pause
