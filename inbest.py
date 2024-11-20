import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Oportunidades",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        .main-title {
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            color: #4a4a4a;
        }
        .section-header {
            font-size: 1.5rem;
            color: #5a5a5a;
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Título del dashboard
st.markdown('<div class="main-title">📊 Análisis Visual de Fuentes de Oportunidades</div>', unsafe_allow_html=True)

# Cargar datos
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')
    return data

file_path = "bd_processed.csv"  # Cambia si necesitas otro archivo
data = load_data(file_path)

# Sidebar interactivo
st.sidebar.header("Filtros")
units = ['Cloud & AI Solutions', 'Enterprise Solutions']
unit_filter = st.sidebar.multiselect(
    "Selecciona Unidades de Negocio",
    options=units,
    default=units
)

# Filtrar datos según selección
if "Cloud & AI Solutions" not in unit_filter:
    data = data[data['Unidad de negocio asignada_Enterprise Solutions'] == 1]
if "Enterprise Solutions" not in unit_filter:
    data = data[data['Unidad de negocio asignada_Enterprise Solutions'] == 0]

# Métricas
st.markdown('<div class="section-header">📈 Métricas Principales</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

total_opportunities = len(data)
won_opportunities = data['etapa_binaria'].sum()
conversion_rate = (won_opportunities / total_opportunities) * 100 if total_opportunities > 0 else 0

col1.metric("Total de Oportunidades", total_opportunities)
col2.metric("Oportunidades Ganadas", won_opportunities, f"{conversion_rate:.1f}%")

# Visualización: Proporción de Oportunidades Ganadas
st.markdown('<div class="section-header">✅ Proporción de Resultados</div>', unsafe_allow_html=True)
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

# Visualización: Oportunidades por Fuente de Tráfico
st.markdown('<div class="section-header">🌐 Oportunidades por Fuente de Tráfico</div>', unsafe_allow_html=True)
traffic_columns = [col for col in data.columns if 'Fuente original de trafico' in col]
traffic_counts = data[traffic_columns].sum()
fig_traffic = px.bar(
    x=traffic_counts.index,
    y=traffic_counts.values,
    title="Distribución por Fuentes de Tráfico",
    labels={"x": "Fuente", "y": "Cantidad"},
    color=traffic_counts.index,
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_traffic, use_container_width=True)

# Gráfico de métricas avanzadas (Altair)
st.markdown('<div class="section-header">📊 Análisis Temporal de Fuentes</div>', unsafe_allow_html=True)
if "Fecha de creacion" in data.columns:
    data['Fecha de creacion'] = pd.to_datetime(data['Fecha de creacion'], errors='coerce')
    data['Mes'] = data['Fecha de creacion'].dt.to_period('M')
    monthly_data = data.groupby(['Mes'])[traffic_columns].sum().reset_index()
    monthly_data['Mes'] = monthly_data['Mes'].astype(str)
    chart = alt.Chart(monthly_data.melt(id_vars=['Mes'], var_name='Fuente', value_name='Oportunidades')).mark_line().encode(
        x='Mes:T',
        y='Oportunidades:Q',
        color='Fuente:N'
    ).properties(
        title="Tendencia Mensual por Fuente de Tráfico",
        width=800,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

# Información adicional
st.sidebar.header("Acerca del Dashboard")
st.sidebar.write("""
Este dashboard muestra un análisis detallado de las oportunidades generadas por diferentes fuentes de tráfico y unidades de negocio.
- Interactúa con los filtros en la barra lateral.
- Explora las métricas clave y gráficos dinámicos.
""")

