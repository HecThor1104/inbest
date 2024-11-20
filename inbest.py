import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
import seaborn as sns
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Oportunidades",
    page_icon="📊",
    layout="wide"
)

# Estilo global de gráficos
sns.set_style("whitegrid")
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 6)
})

# Título del dashboard
st.markdown("<h1 style='text-align: center; color: #4A4A4A;'>🎯 Análisis de Impacto de Fuentes de Oportunidades</h1>", unsafe_allow_html=True)
st.markdown("**Este dashboard interactivo permite explorar las fuentes de tráfico y su impacto en los resultados de oportunidades.**")

# Cargar datos
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')
    return data

file_path = "bd_processed.csv"  # Cambia según la ubicación de tu archivo
data = load_data(file_path)

# Filtros de unidades de negocio
st.markdown("### Selecciona Unidades de Negocio")
units = ['Cloud & AI Solutions', 'Enterprise Solutions']
unit_filter = st.multiselect(
    "Filtra las unidades de negocio a analizar",
    options=units,
    default=units
)

# Aplicar filtros
if "Cloud & AI Solutions" not in unit_filter:
    data = data[data['Unidad de negocio asignada_Enterprise Solutions'] == 1]
if "Enterprise Solutions" not in unit_filter:
    data = data[data['Unidad de negocio asignada_Enterprise Solutions'] == 0]

# Métricas principales
st.markdown("### 📈 Métricas Generales")
col1, col2, col3 = st.columns(3)
total_opportunities = len(data)
won_opportunities = data['etapa_binaria'].sum()
conversion_rate = (won_opportunities / total_opportunities) * 100 if total_opportunities > 0 else 0
col1.metric("Total de Oportunidades", total_opportunities)
col2.metric("Oportunidades Ganadas", won_opportunities)
col3.metric("Tasa de Conversión", f"{conversion_rate:.1f} %")

# Gráfico de pastel: Proporción de resultados
st.markdown("### ✅ Proporción de Resultados")
outcome_counts = data['etapa_binaria'].value_counts()
outcome_labels = ['No Ganado', 'Ganado']
fig_outcomes = px.pie(
    names=outcome_labels,
    values=outcome_counts,
    title="Proporción de Oportunidades Ganadas vs. No Ganadas",
    color=outcome_labels,
    color_discrete_map={"Ganado": "#2ECC71", "No Ganado": "#E74C3C"}
)
st.plotly_chart(fig_outcomes, use_container_width=True)

# Modelo logit y coeficientes
X = data[[col for col in data.columns if 'Fuente original de trafico' in col or 'Unidad de negocio asignada' in col]]
X = sm.add_constant(X)
y = data['etapa_binaria']
logit_model = sm.Logit(y, X).fit(disp=False)
logit_params = logit_model.params
logit_pvalues = logit_model.pvalues

# Gráfico de coeficientes significativos
st.markdown("### 📈 Coeficientes Significativos del Modelo Logit")
significant_params = logit_params[logit_pvalues < 0.05].drop('const', errors='ignore')
if not significant_params.empty:
    fig, ax = plt.subplots()
    sns.barplot(x=significant_params.values, y=significant_params.index, palette="Blues_r", ax=ax)
    ax.set_title('Impacto de Variables Significativas', fontsize=16)
    ax.set_xlabel('Valor del Coeficiente', fontsize=14)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    st.pyplot(fig)
else:
    st.write("No hay coeficientes significativos en el modelo.")

# Curva ROC
st.markdown("### 📉 Curva ROC del Modelo Logit")
logit_pred_probs = logit_model.predict(X)
fpr, tpr, thresholds = roc_curve(y, logit_pred_probs)
roc_auc = auc(fpr, tpr)

fig_roc = px.area(
    x=fpr,
    y=tpr,
    title=f"Curva ROC (AUC = {roc_auc:.2f})",
    labels={"x": "False Positive Rate", "y": "True Positive Rate"},
    color_discrete_sequence=["#1ABC9C"]
)
fig_roc.add_shape(
    type="line",
    x0=0, y0=0, x1=1, y1=1,
    line=dict(color="gray", dash="dash")
)
st.plotly_chart(fig_roc, use_container_width=True)

# Gráfico de barras: Oportunidades por fuente de tráfico
st.markdown("### 🌐 Comparación de Fuentes de Tráfico")
traffic_columns = [col for col in data.columns if 'Fuente original de trafico' in col]
traffic_counts = data[traffic_columns].sum()
fig_traffic = px.bar(
    x=traffic_counts.index,
    y=traffic_counts.values,
    title="Número de Oportunidades por Fuente de Tráfico",
    labels={"x": "Fuente", "y": "Cantidad"},
    color=traffic_counts.index,
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_traffic, use_container_width=True)

# Información adicional
st.markdown("### ℹ️ Información Adicional")
st.write("""
Este dashboard permite analizar las oportunidades generadas por diferentes fuentes de tráfico y unidades de negocio.
Interactúa con los gráficos para obtener más información.
""")


