import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path

print("ANÁLISIS DE PRECIOS DE AGUACATES CON RANDOM FOREST")

# 1. CARGAR Y EXPLORAR DATOS
print("\n Paso 1: Cargando datos y explorandolos.")
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]

csv_path = project_root / 'data' / 'raw' / 'avocado.csv'


if not csv_path.exists():
    matches = list(project_root.rglob('avocado.csv'))
    if matches:
        csv_path = matches[0]
    else:
        raise FileNotFoundError(f"No se encontró 'avocado.csv' a partir de: {project_root}")

df = pd.read_csv(csv_path)
print(f"Archivo cargado desde: {csv_path}")

print(f"Dimensiones: {df.shape}")
print(f"\nPrimeras filas:\n{df.head()}")
print(f"\nTipos de datos:\n{df.dtypes}")
print(f"\nValores faltantes:\n{df.isnull().sum()}")

# 2. LIMPIEZA
print("\n Paso 2: Limpieza de datos.")
df = df.dropna()

columnas_a_eliminar = ['Date', 'Unnamed: 0']
df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])

df = pd.get_dummies(df, columns=['type', 'region'], drop_first=True)

X = df.drop('AveragePrice', axis=1)
y = df['AveragePrice']

print(f"Características: {X.shape[1]} | Registros: {X.shape[0]}")

# 3. DIVIDIR DATOS
print("\n Paso 3: Dividimos datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"Entrenamiento: {X_train.shape[0]} | Prueba: {X_test.shape[0]}")


# 4. OPTIMIZACIÓN DE HIPERPARÁMETROS CON GRIDSEARCHCV

print("\n Paso 4: Buscando hiperparametros optimos.")
print("   Esto puede demorar un poco, así que se paciente ;)")

# Hemos eliminado combinaciones menos probables de ser óptimas
param_grid = {
    'n_estimators': [150, 250],           # 2 valores (suficiente, 150+ es generalmente bueno)
    'max_depth': [12, 18, None],          # 3 valores (None = sin límite, para explorar)
    'min_samples_split': [2, 5],          # 2 valores (2 y 5 son los más comunes)
    'min_samples_leaf': [1, 2]            # 2 valores (1 y 2 casi siempre son óptimos)
}
# Total: 2 × 3 × 2 × 2 = 24 combinaciones × 5-fold CV = 120 entrenamientos (~1-2 minutos)

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n Busqueda completada")
print(f"Mejores parametros: {grid_search.best_params_}")
print(f"Mejor score en CV: {grid_search.best_score_:.4f}")

# 5. ENTRENAR MODELO FINAL CON MEJORES PARÁMETROS
print("\n Paso 5: Entrenando al modelo.")

rf_model = grid_search.best_estimator_

# Validación cruzada con modelo final
cv_scores = cross_val_score(rf_model, X_train, y_train, 
                            cv=5, scoring='r2', n_jobs=-1)
print(f"Validación cruzada (5-fold):")
print(f"  Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Media: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 6. PREDICCIONES
print("\n Paso 6: Realizando predicciones.")
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# 7. EVALUACIÓN COMPLETA
print("\n Paso 7: Evaluacion del modelo")

def calcular_metricas(y_true, y_pred, nombre_conjunto):
    """Calculamos e imprimimos todas las métricas"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n📊 CONJUNTO {nombre_conjunto}:")
    print(f"   MSE (Error Cuadrático Medio):        {mse:.4f}")
    print(f"   RMSE (Raíz del Error Cuadrático):   ${rmse:.2f}")
    print(f"   MAE (Error Absoluto Medio):         ${mae:.2f}")
    print(f"   MAPE (Error Porcentual Medio):      {mape:.2f}%")
    print(f"   R² (Coeficiente de Determinación):  {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

metricas_train = calcular_metricas(y_train, y_pred_train, "ENTRENAMIENTO")
metricas_test = calcular_metricas(y_test, y_pred_test, "PRUEBA")

# R² Ajustado
n = X_test.shape[0]
p = X_test.shape[1]
r2_adj = 1 - (1 - metricas_test['r2']) * (n - 1) / (n - p - 1)
print(f"   R² Ajustado:                         {r2_adj:.4f}")

# Diagnóstico de overfitting
print("\n" + "=" * 70)
diff_r2 = metricas_train['r2'] - metricas_test['r2']
if diff_r2 > 0.15:
    print(f"¡Cuidado! Overfitting detectado (Diferencia R²: {diff_r2:.4f})")
    print("   Considera: aumentar max_depth max, o reducir n_estimators")
elif diff_r2 > 0.05:
    print(f" Ligero overfitting (Diferencia R²: {diff_r2:.4f})")
else:
    print(f"Excelente generalizacion, Durisimo bro. (Diferencia R²: {diff_r2:.4f})")

# 8. IMPORTANCIA DE VARIABLES
print(" Paso 8: Analisis de importancia de variables.")

feature_importance = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': rf_model.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\n Interpretacion de variables mas importantes")
top_features = feature_importance.head(5)
for idx, (_, row) in enumerate(top_features.iterrows(), 1):
    print(f"   {idx}. {row['Variable']}: {row['Importancia']:.3f} ({row['Importancia']*100:.1f}%)")

print("\n   Variables con importancia < 1%:", 
      len(feature_importance[feature_importance['Importancia'] < 0.01]))

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 15 features
axes[0].barh(range(len(feature_importance.head(15))), 
             feature_importance.head(15)['Importancia'].values,
             color='steelblue')
axes[0].set_yticks(range(len(feature_importance.head(15))))
axes[0].set_yticklabels(feature_importance.head(15)['Variable'].values)
axes[0].set_xlabel('Importancia', fontsize=11)
axes[0].set_title('Top 15 Variables Más Importantes', fontsize=12, fontweight='bold')
axes[0].axvline(x=0.01, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Umbral 1%')
axes[0].legend()
axes[0].grid(axis='x', alpha=0.3)

# Distribución de importancias
axes[1].hist(feature_importance['Importancia'], bins=30, color='coral', edgecolor='black')
axes[1].set_xlabel('Importancia', fontsize=11)
axes[1].set_ylabel('Frecuencia', fontsize=11)
axes[1].set_title('Distribución de Importancia de Variables', fontsize=12, fontweight='bold')
axes[1].axvline(x=feature_importance['Importancia'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f"Media: {feature_importance['Importancia'].mean():.3f}")
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 9. ANÁLISIS DE PREDICCIONES
print("Paso 9: Analisis de predicciones vs realidad")

comparison_df = pd.DataFrame({
    'Real': y_test.values,
    'Predicho': y_pred_test,
    'Diferencia': np.abs(y_test.values - y_pred_test),
    'Error%': np.abs((y_test.values - y_pred_test) / y_test.values) * 100
})

print("\nPrimeras 10 predicciones:")
print(comparison_df.head(10).to_string())

print(f"\n📉 ANÁLISIS DE ERRORES:")
print(f"   Error promedio (MAE):           ${comparison_df['Diferencia'].mean():.2f}")
print(f"   Error máximo:                   ${comparison_df['Diferencia'].max():.2f}")
print(f"   Error mínimo:                   ${comparison_df['Diferencia'].min():.2f}")
print(f"   Percentil 25:                   ${comparison_df['Diferencia'].quantile(0.25):.2f}")
print(f"   Mediana (P50):                  ${comparison_df['Diferencia'].quantile(0.50):.2f}")
print(f"   Percentil 75:                   ${comparison_df['Diferencia'].quantile(0.75):.2f}")
print(f"   Error porcentual promedio:      {comparison_df['Error%'].mean():.2f}%")

# Visualizaciones
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter: Predicciones vs Real
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', s=50)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Predicción Perfecta')
axes[0, 0].set_xlabel('Precio Real ($)', fontsize=11)
axes[0, 0].set_ylabel('Precio Predicho ($)', fontsize=11)
axes[0, 0].set_title('Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Residuos
residuos = y_test.values - y_pred_test
axes[0, 1].scatter(y_pred_test, residuos, alpha=0.6, edgecolors='k', s=50)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Precio Predicho ($)', fontsize=11)
axes[0, 1].set_ylabel('Residuos ($)', fontsize=11)
axes[0, 1].set_title('Análisis de Residuos', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Distribución de residuos
axes[1, 0].hist(residuos, bins=30, color='skyblue', edgecolor='black')
axes[1, 0].axvline(x=residuos.mean(), color='red', linestyle='--', lw=2, 
                   label=f'Media: ${residuos.mean():.2f}')
axes[1, 0].set_xlabel('Residuos ($)', fontsize=11)
axes[1, 0].set_ylabel('Frecuencia', fontsize=11)
axes[1, 0].set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Error porcentual
axes[1, 1].hist(comparison_df['Error%'], bins=30, color='lightcoral', edgecolor='black')
axes[1, 1].axvline(x=comparison_df['Error%'].mean(), color='darkred', linestyle='--', lw=2,
                   label=f'Media: {comparison_df["Error%"].mean():.2f}%')
axes[1, 1].set_xlabel('Error Porcentual (%)', fontsize=11)
axes[1, 1].set_ylabel('Frecuencia', fontsize=11)
axes[1, 1].set_title('Distribución de Error Porcentual', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 10. CONCLUSIONES
print("\n" + "=" * 70)
print("Conclusiones del analisis")
print("=" * 70)

if metricas_test['r2'] > 0.85:
    print("\n Excelente: El modelo explica >85% de la variabilidad")
elif metricas_test['r2'] > 0.7:
    print("\n Bueno: El modelo explica >70% de la variabilidad")
elif metricas_test['r2'] > 0.5:
    print("\n Aceptable: El modelo explica >50% de la variabilidad")
else:
    print("\n Mejorable: Considera agregar más características o probar otros modelos")

print(f"\nError típico: ±${metricas_test['mae']:.2f}")
print(f"Error porcentual típico: ±{comparison_df['Error%'].mean():.2f}%")


print("¡ANÁLISIS COMPLETADO!")
