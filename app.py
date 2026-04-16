import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
from fpdf import FPDF
import base64

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="App Regresión Lineal", page_icon="📈", layout="wide")

# Estilos CSS
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Análisis de Regresión Lineal Simple")
st.markdown("Automatiza el análisis de regresiones calculando métricas, visualizando tendencias y exportando informes.")

# --- BARRA LATERAL: ENTRADA DE DATOS ---
st.sidebar.header("1. Entrada de Datos")
data_source = st.sidebar.radio("Selecciona el origen de los datos:", ("Ingreso Manual", "Subir Archivo (CSV/Excel)"))

df = None

if data_source == "Ingreso Manual":
    st.sidebar.markdown("Edita la tabla inferior para agregar tus puntos (X, Y):")
    # Tabla base
    default_data = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0], "Y": [2.1, 3.9, 6.2, 8.0, 10.1]})
    df = st.sidebar.data_editor(default_data, num_rows="dynamic", use_container_width=True)
else:
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo (.csv, .xlsx)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            # Permitir seleccionar qué columnas son X e Y
            st.sidebar.markdown("Selecciona las columnas para X e Y:")
            x_col = st.sidebar.selectbox("Columna X (Variable Independiente)", df_raw.columns)
            y_col = st.sidebar.selectbox("Columna Y (Variable Dependiente)", df_raw.columns)
            
            df = df_raw[[x_col, y_col]].rename(columns={x_col: "X", y_col: "Y"}).dropna()
            
        except Exception as e:
            st.sidebar.error(f"Error al leer el archivo: {e}")

if df is not None and len(df) >= 2:
    # --- MODELO Y CÁLCULOS ---
    X = df[["X"]].values
    Y = df["Y"].values

    # Modelo óptimo de Scikit-Learn
    model = LinearRegression()
    model.fit(X, Y)
    
    m_opt = float(model.coef_[0])
    b_opt = float(model.intercept_)
    y_pred_opt = model.predict(X)
    r2_opt = r2_score(Y, y_pred_opt)

    st.sidebar.markdown("---")
    st.sidebar.header("2. Configuración del Modelo")
    
    model_type = st.sidebar.radio("¿Qué ecuación deseas usar?", ("Modelo Óptimo (Calculado)", "Ecuación Personalizada"))

    if model_type == "Modelo Óptimo (Calculado)":
        m = m_opt
        b = b_opt
        r2 = r2_opt
        y_pred = y_pred_opt
    else:
        st.sidebar.write("Ingresa los valores para $y = mx + b$:")
        m = st.sidebar.number_input("Pendiente (m):", value=m_opt, step=0.1, format="%.4f")
        b = st.sidebar.number_input("Intersección (b):", value=b_opt, step=0.1, format="%.4f")
        
        y_pred = m * df["X"].values + b
        r2 = r2_score(Y, y_pred)
    
    # Calcular Residuos
    df_results = df.copy()
    df_results["Predicción (Y_hat)"] = y_pred
    df_results["Residuo (Error)"] = df_results["Y"] - df_results["Predicción (Y_hat)"]
    
    # --- VISUALIZACIÓN DE RESULTADOS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Ecuación de la Recta</div>
            <div class="metric-value">Y = {m:.4f}X {b:+.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Coef. Determinación (R²)</div>
            <div class="metric-value">{r2:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Promedio de Residuos (Abs)</div>
            <div class="metric-value">{np.mean(np.abs(df_results["Residuo (Error)"])):.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    col_plot, col_data = st.columns([3, 2])
    
    # Gráfica
    with col_plot:
        st.subheader("Gráfica de Dispersión y Tendencia")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x="X", y="Y", data=df_results, ax=ax, color='#1f77b4', s=60, label="Datos Reales")
        
        # Puntos de la línea
        x_vals = np.linspace(df["X"].min(), df["X"].max(), 100)
        y_vals = m * x_vals + b
        
        ax.plot(x_vals, y_vals, color='#ff7f0e', linewidth=2, label=f"Tendencia: Y = {m:.2f}X {b:+.2f}")
        
        # Opcional: mostrar residuos
        for i in range(len(df_results)):
            ax.plot([df_results["X"].iloc[i], df_results["X"].iloc[i]], 
                    [df_results["Y"].iloc[i], df_results["Predicción (Y_hat)"].iloc[i]], 
                    color='gray', linestyle='--', alpha=0.5)

        ax.set_title("Análisis de Regresión Lineal", fontsize=14, pad=10)
        ax.set_xlabel("Variable Independiente (X)")
        ax.set_ylabel("Variable Dependiente (Y)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        sns.despine()
        
        st.pyplot(fig)
        
        # Guardar gráfico en memoria para PDF
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        img_bytes = img_buffer.getvalue()

    # Tabla de Resultados
    with col_data:
        st.subheader("Datos y Cálculos")
        st.dataframe(df_results, use_container_width=True, height=350)
    
    st.markdown("---")
    st.header("3. Exportar Informes")
    
    col_out1, col_out2 = st.columns(2)
    
    # EXPORTAR EXCEL
    with col_out1:
        st.markdown("**Descargar Excel:** Un archivo .xlsx que incluye la tabla con predicciones, residuos y métricas calculadas.")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_results.to_excel(writer, index=False, sheet_name='Resultados')
            
            # Crear tabla de métricas
            metrics_df = pd.DataFrame({
                "Métrica": ["Pendiente (m)", "Intersección (b)", "R-Cuadrado (R²)", "Ecuación", "Tipo de Modelo"],
                "Valor": [m, b, r2, f"Y = {m:.4f}X {b:+.4f}", model_type]
            })
            metrics_df.to_excel(writer, index=False, sheet_name='Métricas')
        
        excel_data = output.getvalue()
        
        st.download_button(
            label="📊 Descargar Reporte en Excel",
            data=excel_data,
            file_name="Reporte_Regresion_Lineal.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # EXPORTAR PDF
    with col_out2:
        st.markdown("**Descargar PDF:** Un informe visual que incluye la gráfica renderizada, las métricas y la tabla completa de datos.")
        
        # FPDF generator
        def create_pdf(df_res, m_val, b_val, r2_val, img_bytes_buf, m_type):
            pdf = FPDF()
            pdf.add_page()
            
            # Título
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Reporte de Analisis de Regresion Lineal', 0, 1, 'C')
            pdf.ln(5)
            
            # Métricas
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, '1. Metricas del Modelo', 0, 1)
            
            pdf.set_font('Arial', '', 11)
            pdf.cell(0, 8, f'Tipo de Modelo: {m_type}', 0, 1)
            pdf.cell(0, 8, f'Ecuacion de la Recta: Y = {m_val:.4f} * X {b_val:+.4f}', 0, 1)
            pdf.cell(0, 8, f'Pendiente (m): {m_val:.4f}', 0, 1)
            pdf.cell(0, 8, f'Interseccion (b): {b_val:.4f}', 0, 1)
            pdf.cell(0, 8, f'Coeficiente de Determinacion (R2): {r2_val:.4f}', 0, 1)
            pdf.ln(5)
            
            # Gráfica
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, '2. Visualizacion', 0, 1)
            
            # Guardar temporalmente la imagen para FPDF
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_bytes_buf)
                tmp_path = tmp_file.name
            
            pdf.image(tmp_path, x=15, w=180)
            os.unlink(tmp_path)
            
            pdf.add_page()
            
            # Tabla de Datos
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, '3. Tabla de Datos (Completa)', 0, 1)
            pdf.ln(2)
            
            # Encabezados de tabla
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(40, 8, 'X', 1, 0, 'C')
            pdf.cell(40, 8, 'Y', 1, 0, 'C')
            pdf.cell(50, 8, 'Prediccion (Y_hat)', 1, 0, 'C')
            pdf.cell(50, 8, 'Residuo (Error)', 1, 1, 'C')
            
            # Filas de la tabla
            pdf.set_font('Arial', '', 10)
            for idx, row in df_res.iterrows():
                pdf.cell(40, 8, f'{row["X"]:.4f}', 1, 0, 'C')
                pdf.cell(40, 8, f'{row["Y"]:.4f}', 1, 0, 'C')
                pdf.cell(50, 8, f'{row["Predicción (Y_hat)"]:.4f}', 1, 0, 'C')
                pdf.cell(50, 8, f'{row["Residuo (Error)"]:.4f}', 1, 1, 'C')
            
            # Fix unicode issues by encoding to latin-1
            # FPDF (standard) only handles latin-1 natively without custom fonts
            # we replaced accented characters with unaccented variants to prevent issues
            
            return bytes(pdf.output())

        try:
            pdf_data = create_pdf(df_results, m, b, r2, img_bytes, model_type)
            
            st.download_button(
                label="📄 Descargar Reporte en PDF",
                data=pdf_data,
                file_name="Reporte_Regresion_Lineal.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generando PDF: {e}")

else:
    st.info("👈 Por favor, ingresa o sube datos con al menos 2 puntos para comenzar el análisis.")
