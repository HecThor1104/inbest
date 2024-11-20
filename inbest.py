import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Oportunidades",
    page_icon="📊",
    layout="wide"
)

# Título del dashboard
st.markdown("<h1 style='text-align: center; color: #4A4A4A;'>📊 Análisis de Oportunidades</h1>", unsafe_allow_html=True)

# Cargar datos
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')
    return data

file_path = "bd_processed.csv"  # Cambia si necesitas otro archivo
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

# Gráfico de barras: Oportunidades por fuente de tráfico
st.markdown("### 🌐 Distribución por Fuentes de Tráfico")
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

# Gráfico de líneas: Análisis temporal
if "Fecha de creacion" in data.columns:
    st.markdown("### 📊 Tendencias Temporales")
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

# Análisis adicional o información contextual
st.markdown("### ℹ️ Información Adicional")
st.write("""
Este dashboard permite analizar las oportunidades generadas por diferentes fuentes de tráfico y unidades de negocio.
Interacciona con los gráficos para obtener más información.
""")


