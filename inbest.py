import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
import seaborn as sns
import plotly.express as px
import altair as alt

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

# Cargar datos
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')
    return data

file_path = "bd_processed.csv"  # Cambia según la ubicación de tu archivo
data = load_data(file_path)

# Procesar nombres de columnas (limpiar "Fuente original de tráfico_" y corregir caracteres)
def clean_column_names(columns):
    return [col.split('_')[-1].replace("ñ", "ñ") for col in columns]

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

# Gráfico de pastel: Distribución de unidades de negocio
st.markdown("### 🏢 Distribución de Oportunidades por Unidad de Negocio")
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

# Gráfico de líneas: Tendencias temporales
if "Fecha de creacion" in data.columns:
    st.markdown("### 📊 Tendencias Mensuales por Fuente de Tráfico")
    data['Fecha de creacion'] = pd.to_datetime(data['Fecha de creacion'], errors='coerce')
    data['Mes'] = data['Fecha de creacion'].dt.to_period('M')
    monthly_data = data.groupby(['Mes'])[traffic_columns].sum().reset_index()
    monthly_data['Mes'] = monthly_data['Mes'].astype(str)
    melted_monthly_data = monthly_data.melt(id_vars=['Mes'], var_name='Fuente', value_name='Oportunidades')
    melted_monthly_data['Fuente'] = clean_column_names(melted_monthly_data['Fuente'])
    chart = alt.Chart(melted_monthly_data).mark_line().encode(
        x='Mes:T',
        y='Oportunidades:Q',
        color='Fuente:N'
    ).properties(
        title="Tendencia Mensual de Oportunidades por Fuente",
        width=800,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

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
    cleaned_significant_labels = clean_column_names(significant_params.index)
    fig, ax = plt.subplots()
    sns.barplot(x=significant_params.values, y=cleaned_significant_labels, palette="Blues_r", ax=ax)
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

fig_roc, ax_roc = plt.subplots()
sns.lineplot(x=fpr, y=tpr, label=f'ROC curve (area = {roc_auc:.2f})', ax=ax_roc, color="#1ABC9C")
ax_roc.plot([0, 1], [0, 1], 'k--', label='Random guess', color="gray")
ax_roc.set_xlabel('False Positive Rate', fontsize=14)
ax_roc.set_ylabel('True Positive Rate', fontsize=14)
ax_roc.set_title('Curva ROC - Modelo Logit', fontsize=16)
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# Exploración interactiva de variables
st.markdown("### 🔍 Exploración de Impacto de Variables")
if not significant_params.empty:
    selected_variable = st.selectbox("Selecciona una variable", significant_params.index)
    impact_value = significant_params[selected_variable]
    st.markdown(f"**El impacto de `{selected_variable}` en la probabilidad de 'ganado' es:** `{impact_value:.2f}`")
else:
    st.write("No hay variables significativas para explorar.")





