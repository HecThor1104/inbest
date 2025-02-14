import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

# Autor: Hector Plascencia

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
st.markdown("<h1 style='text-align: center; color: #4A4A4A;'>🎯 Análisis de Marketing - iNBest</h1>", unsafe_allow_html=True)

# Cargar datos
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')
    return data

file_path = "bd_processed.csv"  # Cambia según la ubicación de tu archivo
data = load_data(file_path)

# Procesar nombres de columnas (limpiar "Fuente original de tráfico_" y corregir caracteres)
def clean_column_names(columns):
    return [col.split('_')[-1].replace("ñ", "ñ") for col in columns]

# Vista previa de los datos
st.markdown("### 🗂️ Vista Previa de los Datos")
st.write("En esta sección, puedes explorar los datos cargados para entender mejor el contenido del análisis.")
st.dataframe(data.head(10), use_container_width=True)

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
st.write("Aquí se presentan los valores clave que resumen las oportunidades procesadas.")
col1, col2, col3 = st.columns(3)
total_opportunities = len(data)
won_opportunities = data['etapa_binaria'].sum()
conversion_rate = (won_opportunities / total_opportunities) * 100 if total_opportunities > 0 else 0
col1.metric("Total de Oportunidades", total_opportunities)
col2.metric("Oportunidades Ganadas", won_opportunities)
col3.metric("Tasa de Conversión", f"{conversion_rate:.1f} %")

# Gráfico de pastel: Distribución de unidades de negocio
st.markdown("### 🏢 Distribución de Oportunidades por Unidad de Negocio")
st.write("Este gráfico muestra cómo se distribuyen las oportunidades entre las distintas unidades de negocio.")
unit_counts = {
    "Cloud & AI Solutions": len(data[data['Unidad de negocio asignada_Enterprise Solutions'] == 0]),
    "Enterprise Solutions": len(data[data['Unidad de negocio asignada_Enterprise Solutions'] == 1]),
}
fig_units = px.pie(
    names=list(unit_counts.keys()),
    values=list(unit_counts.values()),
    title="Distribución de Oportunidades por Unidad de Negocio",
    color=list(unit_counts.keys()),
    color_discrete_map={
        "Cloud & AI Solutions": "#AED6F1",
        "Enterprise Solutions": "#2E86C1"
    }
)
st.plotly_chart(fig_units, use_container_width=True)

# Gráfico de pastel: Proporción de resultados
st.markdown("### ✅ Proporción de Oportunidades Ganadas vs. No Ganadas")
st.write("Este gráfico ilustra la proporción de oportunidades que han sido ganadas frente a las no ganadas.")
outcome_counts = data['etapa_binaria'].value_counts()
outcome_labels = ['No Ganado', 'Ganado']
fig_outcomes = px.pie(
    names=outcome_labels,
    values=outcome_counts,
    title="Proporción de Resultados",
    color=outcome_labels,
    color_discrete_map={"Ganado": "#2ECC71", "No Ganado": "#E74C3C"}
)
st.plotly_chart(fig_outcomes, use_container_width=True)

# Gráfico de barras: Comparación de fuentes de tráfico
st.markdown("### 🌐 Comparación de Fuentes de Tráfico")
st.write("Este gráfico compara la cantidad de oportunidades generadas por cada fuente de tráfico.")
traffic_columns = [col for col in data.columns if 'Fuente original de trafico' in col]
traffic_counts = data[traffic_columns].sum()
cleaned_labels = clean_column_names(traffic_columns)
fig_traffic = px.bar(
    x=cleaned_labels,
    y=traffic_counts.values,
    title="Número de Oportunidades por Fuente de Tráfico",
    labels={"x": "Fuente", "y": "Cantidad"},
    color=cleaned_labels,
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_traffic, use_container_width=True)

# El resto del código sigue igual...










