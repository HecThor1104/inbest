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
    data['etapa_binaria'] = data['Etapa del negocio'].apply(lambda x: 1 if x.strip().lower() == 'ganado' else 0)
    categorical_columns = ['Fuente original de trafico', 'Unidad de negocio asignada']
    data_dummies = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    return data, data_dummies

file_path = "bd.csv"  # Cambia esto según tu archivo
data, data_dummies = load_data(file_path)

# Resumen descriptivo
st.header("Datos Cargados")
st.write(data.head())

# Seleccionar variables independientes
X = data_dummies[[col for col in data_dummies.columns if 'Fuente original de trafico' in col or 'Unidad de negocio asignada' in col]]
X = sm.add_constant(X)
y = data_dummies['etapa_binaria']

# Modelo logit
logit_model = sm.Logit(y, X).fit(disp=False)
logit_params = logit_model.params
logit_pvalues = logit_model.pvalues

# Gráfico de coeficientes significativos
st.header("Coeficientes Significativos del Modelo Logit")
significant_params = logit_params[logit_pvalues < 0.05].drop('const')
fig, ax = plt.subplots(figsize=(10, 6))
significant_params.sort_values().plot(kind='barh', edgecolor='black', ax=ax)
ax.set_title('Coeficientes Significativos - Modelo Logit')
ax.set_xlabel('Valor del Coeficiente')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.grid(axis='x', linestyle='--', alpha=0.7)
st.pyplot(fig)

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
selected_variable = st.selectbox("Selecciona una variable", significant_params.index)
impact_value = significant_params[selected_variable]
st.write(f"El impacto de **{selected_variable}** en la probabilidad de 'ganado' es: **{impact_value:.2f}**")
