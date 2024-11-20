import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm

# Título del dashboard
st.title("Análisis de Impacto de Fuentes de Oportunidades")

# Cargar datos
@st.cache
def load_data(file_path):
    # Leer el archivo procesado
    data = pd.read_csv(file_path, encoding='latin1')
    return data

file_path = "bd_processed.csv"  # Cambia esto según la ubicación de tu archivo
data = load_data(file_path)

# Resumen descriptivo
st.header("Datos Cargados")
st.write("Vista previa de los datos:")
st.write(data.head())

st.write("Columnas en los datos cargados:")
st.write(data.columns.tolist())

# Seleccionar variables independientes (todas las columnas dummy relacionadas)
X = data[[col for col in data.columns if 'Fuente original de trafico' in col or 'Unidad de negocio asignada' in col]]
X = sm.add_constant(X)  # Agregar constante para el modelo
y = data['etapa_binaria']  # Variable dependiente

# Modelo logit
logit_model = sm.Logit(y, X).fit(disp=False)
logit_params = logit_model.params
logit_pvalues = logit_model.pvalues

# Gráfico de coeficientes significativos
st.header("Coeficientes Significativos del Modelo Logit")
significant_params = logit_params[logit_pvalues < 0.05].drop('const', errors='ignore')

if not significant_params.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    significant_params.sort_values().plot(kind='barh', edgecolor='black', ax=ax)
    ax.set_title('Coeficientes Significativos - Modelo Logit')
    ax.set_xlabel('Valor del Coeficiente')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig)
else:
    st.write("No hay coeficientes significativos en el modelo.")

# Curva ROC
st.header("Curva ROC del Modelo Logit")
logit_pred_probs = logit_model.predict(X)
fpr, tpr, thresholds = roc_curve(y, logit_pred_probs)
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
ax_roc.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], 'k--', label='Random guess')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Curva ROC - Modelo Logit')
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# Exploración interactiva de variables
st.header("Explorar Impacto de Variables")
if not significant_params.empty:
    selected_variable = st.selectbox("Selecciona una variable", significant_params.index)
    impact_value = significant_params[selected_variable]
    st.write(f"El impacto de **{selected_variable}** en la probabilidad de 'ganado' es: **{impact_value:.2f}**")
else:
    st.write("No hay variables significativas para explorar.")

