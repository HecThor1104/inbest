import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Oportunidades",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background-color: #343a40;
            color: white;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .chart-title {
            font-weight: bold;
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Título del dashboard
st.markdown('<div class="main-header">📊 Análisis Visual de Fuentes de Oportunidades</div>', unsafe_allow_html=True)

# Cargar datos
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')
    return data

file_path = "bd_processed.csv"  # Cambia esto si necesitas otro archivo
data = load_data(file_path)

# Sidebar con opciones interactivas
st.sidebar.header("Filtros")
unit_filter = st.sidebar.multiselect(
    "Selecciona Unidades de Negocio",
    options=["Cloud & AI Solutions", "Enterprise Solutions"],
    default=["Cloud & AI Solutions", "Enterprise Solutions"]
)

# Filtrar datos según la selección
if "Cloud & AI Solutions" not in unit_filter:
    data = data[data['Unidad de negocio asignada_Enterprise Solutions'] == 1]
if "Enterprise Solutions" not in unit_filter:
    data = data[data['Unidad de negocio asignada_Enterprise Solutions'] == 0]

# Visualización: Proporción de Oportunidades Ganadas
st.markdown('<div class="chart-title">Proporción de Resultados</div>', unsafe_allow_html=True)
outcome_counts = data['etapa_binaria'].value_counts()
outcome_labels = ['No Ganado', 'Ganado']
colors = ['#E74C3C', '#2ECC71']
fig_outcomes, ax_outcomes = plt.subplots()
ax_outcomes.pie(outcome_counts, labels=outcome_labels, autopct='%1.1f%%', startangle=90, colors=colors)
st.pyplot(fig_outcomes)

# Visualización: Número de oportunidades por fuente de tráfico
st.markdown('<div class="chart-title">Oportunidades por Fuente de Tráfico</div>', unsafe_allow_html=True)
traffic_columns = [col for col in data.columns if 'Fuente original de trafico' in col]
traffic_counts = data[traffic_columns].sum()
fig_traffic = px.bar(
    traffic_counts,
    x=traffic_counts.index,
    y=traffic_counts.values,
    title="Fuentes de Tráfico",
    labels={"x": "Fuente", "y": "Cantidad"},
    color=traffic_counts.index,
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_traffic)

# Modelo logit y coeficientes
X = data[[col for col in data.columns if 'Fuente original de trafico' in col or 'Unidad de negocio asignada' in col]]
X = sm.add_constant(X)
y = data['etapa_binaria']
logit_model = sm.Logit(y, X).fit(disp=False)
logit_params = logit_model.params
logit_pvalues = logit_model.pvalues

# Coeficientes significativos
significant_params = logit_params[logit_pvalues < 0.05].drop('const', errors='ignore')
st.markdown('<div class="chart-title">Coeficientes Significativos</div>', unsafe_allow_html=True)
if not significant_params.empty:
    fig, ax = plt.subplots()
    sns.barplot(x=significant_params.values, y=significant_params.index, palette="Blues_r", ax=ax)
    ax.set_title('Impacto de Variables', fontsize=16)
    st.pyplot(fig)
else:
    st.write("No hay coeficientes significativos en el modelo.")

# Curva ROC
logit_pred_probs = logit_model.predict(X)
fpr, tpr, thresholds = roc_curve(y, logit_pred_probs)
roc_auc = auc(fpr, tpr)
st.markdown('<div class="chart-title">Curva ROC</div>', unsafe_allow_html=True)
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
st.plotly_chart(fig_roc)

# Información adicional en el sidebar
st.sidebar.markdown("### Información")
st.sidebar.write("Este dashboard permite analizar el impacto de las fuentes de tráfico en las oportunidades ganadas.")




