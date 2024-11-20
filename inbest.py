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
    data = pd.read_csv(file_path, encoding='latin1')
    return data

file_path = "bd_processed.csv"  # Cambia esto según la ubicación de tu archivo
data = load_data(file_path)

# Resumen descriptivo
st.header("Datos Cargados")
st.write("Vista previa de los datos:")
st.write(data.head())

# Distribución de oportunidades por unidad de negocio
st.header("Distribución de Oportunidades por Unidad de Negocio")
unit_counts = data['Unidad de negocio asignada_Enterprise Solutions'].value_counts()
unit_labels = ['Enterprise Solutions', 'Cloud & AI Solutions']
fig_units, ax_units = plt.subplots()
ax_units.pie(unit_counts, labels=unit_labels, autopct='%1.1f%%', startangle=90)
ax_units.set_title("Distribución por Unidad de Negocio")
st.pyplot(fig_units)

# Proporción de oportunidades ganadas vs. no ganadas
st.header("Proporción de Oportunidades Ganadas vs. No Ganadas")
outcome_counts = data['etapa_binaria'].value_counts()
outcome_labels = ['No Ganado', 'Ganado']
fig_outcomes, ax_outcomes = plt.subplots()
ax_outcomes.pie(outcome_counts, labels=outcome_labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
ax_outcomes.set_title("Proporción de Resultados")
st.pyplot(fig_outcomes)

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

# Comparación de fuentes de tráfico
st.header("Comparación de Fuentes de Tráfico")
traffic_columns = [col for col in data.columns if 'Fuente original de trafico' in col]
traffic_counts = data[traffic_columns].sum()
fig_traffic, ax_traffic = plt.subplots(figsize=(10, 6))
traffic_counts.sort_values().plot(kind='bar', ax=ax_traffic, color='skyblue', edgecolor='black')
ax_traffic.set_title("Número de Oportunidades por Fuente de Tráfico")
ax_traffic.set_ylabel("Cantidad")
ax_traffic.set_xlabel("Fuente de Tráfico")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig_traffic)


