import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as plgo
import matplotlib.pyplot as plt
import io
from fpdf import FPDF

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
        height: 100%;
    }
    .metric-value {
        font-size: 20px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    .metric-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Análisis de Regresión Lineal: Comparación de Métodos")
st.markdown("Herramienta para comparar Regresión Lineal Simple (LSR), Regresión Ortogonal Convencional (GOR) y GOR Propuesto.")

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
            
            st.sidebar.markdown("Selecciona las columnas para X e Y:")
            x_col = st.sidebar.selectbox("Columna X (Variable Independiente)", df_raw.columns)
            y_col = st.sidebar.selectbox("Columna Y (Variable Dependiente)", df_raw.columns)
            
            df = df_raw[[x_col, y_col]].rename(columns={x_col: "X", y_col: "Y"}).dropna()
            
        except Exception as e:
            st.sidebar.error(f"Error al leer el archivo: {e}")

if df is not None and len(df) >= 2:
    st.sidebar.markdown("---")
    st.sidebar.header("2. Parámetros Globales")
    eta = st.sidebar.number_input("Valor de Eta (λ) para GOR:", value=1.0, step=0.1, format="%.4f")

    # --- SECCIÓN DE CÁLCULOS GLOBALES (OCULTA) ---
    X = df["X"].values
    Y = df["Y"].values
    n = len(df)
    
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    Sxx = np.sum((X - x_mean)**2)
    Syy = np.sum((Y - y_mean)**2)
    Sxy = np.sum((X - x_mean)*(Y - y_mean))

    # 1. LSR (Simple Linear Regression)
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    m_lsr = float(model.coef_[0])
    b_lsr = float(model.intercept_)
    y_pred_lsr = m_lsr * X + b_lsr

    # 2. GOR Convencional (Das et al.)
    if Sxy != 0:
        beta1_num = (Syy - eta * Sxx) + np.sqrt((Syy - eta * Sxx)**2 + 4 * eta * Sxy**2)
        beta1_den = 2 * Sxy
        m_gor = beta1_num / beta1_den
    else:
        m_gor = 0
    b_gor = y_mean - m_gor * x_mean
    y_pred_gor = m_gor * X + b_gor

    # Proyecciones Ortogonales verdaderas (X_t, Y_t)
    X_t = (eta * X + m_gor * (Y - b_gor)) / (eta + m_gor**2)
    Y_t = b_gor + m_gor * X_t
    Y_t_mean = np.mean(Y_t)

    # 3. GOR Propuesto (Insesgado)
    if Sxx != 0:
        m_prop = np.sum((X - x_mean) * (Y_t - Y_t_mean)) / Sxx
    else:
        m_prop = 0
    b_prop = Y_t_mean - m_prop * x_mean
    y_pred_prop = m_prop * X + b_prop

    # Construir DataFrame de Resultados
    df_results = df.copy()
    df_results["Y_hat (LSR)"] = y_pred_lsr
    df_results["Residuo (LSR)"] = Y - y_pred_lsr
    
    df_results["Y_hat (GOR Conv)"] = y_pred_gor
    df_results["Residuo (GOR Conv)"] = Y - y_pred_gor
    df_results["X_t (GOR)"] = X_t
    df_results["Y_t (GOR)"] = Y_t
    
    df_results["Y_hat (GOR Prop)"] = y_pred_prop
    df_results["Residuo (GOR Prop)"] = Y - y_pred_prop

    # Función para calcular Error Estándar y R2
    def calc_metrics(y_pred, m):
        sse = np.sum((Y - y_pred)**2)
        mse = sse / (n - 2) if n > 2 else 0
        sigma = np.sqrt(mse) if n > 2 else 0
        
        se_m = sigma / np.sqrt(Sxx) if Sxx != 0 else 0
        se_b = sigma * np.sqrt(1/n + (x_mean**2)/Sxx) if Sxx != 0 else 0
        r2 = 1 - (sse / Syy) if Syy != 0 else 0
        return se_m, se_b, r2

    se_m_lsr, se_b_lsr, r2_lsr = calc_metrics(y_pred_lsr, m_lsr)
    se_m_gor, se_b_gor, r2_gor = calc_metrics(y_pred_gor, m_gor)
    se_m_prop, se_b_prop, r2_prop = calc_metrics(y_pred_prop, m_prop)

    # --- DASHBOARD E INTERFAZ PRINCIPAL ---
    tab1, tab2, tab3 = st.tabs(["📊 Análisis Comparativo", "📋 Datos y Proyecciones", "📚 Explicación de Fórmulas"])

    with tab1:
        # Métrica Principal
        diff_pendiente = abs(m_lsr - m_prop) / abs(m_lsr) * 100 if m_lsr != 0 else 0
        st.info(f"**Métrica Principal:** La diferencia de pendiente entre LSR y GOR Propuesto es del **{diff_pendiente:.2f}%**.")

        # Gráfico Dinámico con Plotly
        st.subheader("Gráfico Dinámico de Regresiones")
        
        fig_plotly = plgo.Figure()

        # Puntos Reales
        fig_plotly.add_trace(plgo.Scatter(
            x=X, y=Y, mode='markers', name='Datos Observados',
            marker=dict(color='black', size=8),
            hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>'
        ))

        # Líneas de Tendencia
        x_line = np.linspace(min(X), max(X), 100)
        
        fig_plotly.add_trace(plgo.Scatter(
            x=x_line, y=m_lsr * x_line + b_lsr, mode='lines', name='LSR (OLS)',
            line=dict(color='blue', width=2),
            hovertemplate='X: %{x:.2f}<br>Y (LSR): %{y:.2f}<extra></extra>'
        ))
        
        fig_plotly.add_trace(plgo.Scatter(
            x=x_line, y=m_gor * x_line + b_gor, mode='lines', name='GOR Convencional',
            line=dict(color='orange', width=2, dash='dash'),
            hovertemplate='X: %{x:.2f}<br>Y (GOR): %{y:.2f}<extra></extra>'
        ))
        
        fig_plotly.add_trace(plgo.Scatter(
            x=x_line, y=m_prop * x_line + b_prop, mode='lines', name='GOR Propuesto',
            line=dict(color='green', width=2, dash='dot'),
            hovertemplate='X: %{x:.2f}<br>Y (Prop): %{y:.2f}<extra></extra>'
        ))

        fig_plotly.update_layout(
            title='Comparación de Métodos de Regresión',
            xaxis_title='Variable Independiente (X)',
            yaxis_title='Variable Dependiente (Y)',
            hovermode='closest',
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_plotly, use_container_width=True)

        # Tabla Comparativa de Parámetros
        st.subheader("Tabla de Parámetros")
        
        comp_data = {
            "Método": ["LSR (OLS)", "GOR Convencional", "GOR Propuesto"],
            "Ecuación": [f"Y = {m_lsr:.4f}X {b_lsr:+.4f}", f"Y = {m_gor:.4f}X {b_gor:+.4f}", f"Y = {m_prop:.4f}X {b_prop:+.4f}"],
            "Pendiente (m)": [m_lsr, m_gor, m_prop],
            "Intercepción (b)": [b_lsr, b_gor, b_prop],
            "Error Estándar (m)": [se_m_lsr, se_m_gor, se_m_prop],
            "Error Estándar (b)": [se_b_lsr, se_b_gor, se_b_prop],
            "R²": [r2_lsr, r2_gor, r2_prop]
        }
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp.style.format({
            "Pendiente (m)": "{:.4f}",
            "Intercepción (b)": "{:.4f}",
            "Error Estándar (m)": "{:.4f}",
            "Error Estándar (b)": "{:.4f}",
            "R²": "{:.4f}"
        }), use_container_width=True)

    with tab2:
        st.subheader("Datos Originales y Proyecciones")
        st.markdown("Tabla detallada con cada dato original, sus predicciones ($\\hat{Y}$) y los residuos según cada modelo.")
        st.dataframe(df_results, use_container_width=True)

        st.markdown("---")
        st.subheader("Exportar Resultados")
        col_out1, col_out2 = st.columns(2)
        
        with col_out1:
            # EXPORTAR EXCEL
            st.markdown("**Descargar Excel:**")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_results.to_excel(writer, index=False, sheet_name='Resultados_Completos')
                df_comp.to_excel(writer, index=False, sheet_name='Comparacion_Modelos')
            excel_data = output.getvalue()
            st.download_button(
                label="📊 Descargar Reporte en Excel",
                data=excel_data,
                file_name="Reporte_Comparacion_Regresiones.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        with col_out2:
            # EXPORTAR PDF
            st.markdown("**Descargar PDF:**")
            
            def create_pdf():
                # Crear imagen de Matplotlib estática en background para el PDF
                fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5))
                ax_pdf.scatter(X, Y, color='black', label='Datos')
                x_vals = np.linspace(min(X), max(X), 100)
                ax_pdf.plot(x_vals, m_lsr * x_vals + b_lsr, color='blue', label='LSR')
                ax_pdf.plot(x_vals, m_gor * x_vals + b_gor, color='orange', linestyle='--', label='GOR Conv')
                ax_pdf.plot(x_vals, m_prop * x_vals + b_prop, color='green', linestyle=':', label='GOR Prop')
                ax_pdf.legend()
                ax_pdf.set_title("Comparacion de Metodos")
                
                img_buffer = io.BytesIO()
                fig_pdf.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                plt.close(fig_pdf)
                
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, 'Reporte de Comparacion de Regresiones', 0, 1, 'C')
                pdf.ln(5)
                
                # Imagen
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(img_buffer.getvalue())
                    tmp_path = tmp_file.name
                pdf.image(tmp_path, x=15, w=180)
                os.unlink(tmp_path)
                
                pdf.ln(5)
                # Métricas Principales (LSR vs Propuesto)
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, '1. Resumen de Modelos', 0, 1)
                pdf.set_font('Arial', '', 10)
                
                for i, row in df_comp.iterrows():
                    pdf.cell(0, 6, f"{row['Método']}: {row['Ecuación']} | R2: {row['R²']:.4f}", 0, 1)
                
                return bytes(pdf.output())

            try:
                pdf_data = create_pdf()
                st.download_button(
                    label="📄 Descargar Reporte en PDF",
                    data=pdf_data,
                    file_name="Reporte_Comparacion_Regresiones.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Error generando PDF: {e}")

    with tab3:
        st.subheader("📚 Fórmulas y Cálculos Detallados")
        
        st.markdown("""
        Esta sección muestra las fórmulas utilizadas para el cálculo de cada uno de los métodos. 
        Los cálculos para la Regresión Ortogonal Generalizada siguen el procedimiento de *Das et al. (2018)*.
        """)
        
        with st.expander("1. LSR (Regresión Lineal Simple)"):
            st.latex(rf"m_{{LSR}} = \frac{{S_{{xy}}}}{{S_{{xx}}}} = {m_lsr:.4f}")
            st.latex(rf"b_{{LSR}} = \bar{{Y}} - m_{{LSR}}\bar{{X}} = {b_lsr:.4f}")
            
        with st.expander("2. GOR Convencional"):
            st.markdown("- **Parámetro Eta (Lambda):**")
            st.latex(rf"\eta = \lambda = {eta:.4f}")
            
            st.markdown("- **Pendiente GOR ($\hat{\beta}_1$):**")
            st.latex(rf"\hat{{\beta}}_1 = \frac{{(S_{{yy}} - \eta S_{{xx}}) + \sqrt{{(S_{{yy}} - \eta S_{{xx}})^2 + 4\eta S_{{xy}}^2}}}}{{2S_{{xy}}}} = {m_gor:.4f}")
            
            st.markdown("- **Intersección GOR ($\hat{\beta}_0$):**")
            st.latex(rf"\hat{{\beta}}_0 = \bar{{Y}} - \hat{{\beta}}_1 \bar{{X}} = {b_gor:.4f}")
            
            st.markdown("- **Proyecciones Ortogonales Verdaderas ($X_t, Y_t$):**")
            st.latex(r"X_t = \frac{\hat{\beta}_1(Y_{obs} - \hat{\beta}_0) + \eta X_{obs}}{\eta + \hat{\beta}_1^2}")
            st.latex(r"Y_t = \hat{\beta}_0 + \hat{\beta}_1 X_t")
            
        with st.expander("3. GOR Propuesto (Insesgado)"):
            st.markdown("- **Pendiente Propuesta ($c_1$):**")
            st.latex(rf"c_1 = \frac{{\sum (X_{{obs,i}} - \bar{{X}}_{{obs}})(Y_{{t,i}} - \bar{{Y}}_t)}}{{\sum (X_{{obs,i}} - \bar{{X}}_{{obs}})^2}} = {m_prop:.4f}")
            
            st.markdown("- **Intersección Propuesta ($c_2$):**")
            st.latex(rf"c_2 = \bar{{Y}}_t - c_1 \bar{{X}}_{{obs}} = {b_prop:.4f}")

else:
    st.info("👈 Por favor, ingresa o sube datos con al menos 2 puntos para comenzar el análisis comparativo.")
