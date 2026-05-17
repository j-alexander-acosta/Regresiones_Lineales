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
    .latex-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Plataforma Educativa de Regresión Lineal: Comparación de Métodos")
st.markdown("Herramienta orientada a estudiantes de ingeniería y sismología para comprender y comparar: Regresión Lineal por Mínimos Cuadrados Ordinarios (SLR), Regresión Ortogonal Generalizada (GOR) Convencional y GOR Propuesto.")

# --- BARRA LATERAL: ENTRADA DE DATOS ---
st.sidebar.header("1. Entrada de Datos")
data_source = st.sidebar.radio("Selecciona el origen de los datos:", ("Ingreso Manual", "Subir Archivo (CSV/Excel)"))

df = None

if data_source == "Ingreso Manual":
    st.sidebar.markdown("Edita la tabla inferior para agregar tus puntos (X, Y):")
    # Tabla base con 19 muestras
    default_data = pd.DataFrame({
        "X": [4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2],
        "Y": [4.7, 4.6, 5.1, 5.2, 5.0, 5.4, 5.3, 5.5, 5.4, 5.6, 5.8, 5.9, 5.7, 6.1, 6.0, 6.2, 6.4, 6.3, 6.6]
    })
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
    eta = st.sidebar.number_input("Valor de Eta (η) para GOR:", value=0.2000, step=0.1, format="%.4f")

    # --- SECCIÓN DE CÁLCULOS GLOBALES ---
    # Cálculo estricto de dimensiones (n)
    n = len(df)
    X = df["X"].values
    Y = df["Y"].values
    
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    
    # Sumas de cuadrados y productos cruzados
    Sxx = np.sum((X - x_mean)**2)
    Syy = np.sum((Y - y_mean)**2)
    Sxy = np.sum((X - x_mean)*(Y - y_mean))

    # ==================================================
    # MÓDULO 1: SLR (Standard Linear Regression / OLS)
    # ==================================================
    # Pendiente (m_slr) y corte (b_slr)
    m_slr = Sxy / Sxx if Sxx != 0 else 0
    b_slr = y_mean - m_slr * x_mean
    y_pred_slr = m_slr * X + b_slr
    
    # Métricas de Error SLR
    sse_slr = np.sum((Y - y_pred_slr)**2)
    s2_e_slr = sse_slr / (n - 2) if n > 2 else 0 # Varianza Residual
    se_m_slr = np.sqrt(s2_e_slr / Sxx) if Sxx != 0 and n > 2 else 0 # Error estándar de la pendiente
    se_b_slr = se_m_slr * np.sqrt(np.sum(X**2) / n) if n > 0 else 0 # Error estándar de la intercepción
    rmse_slr = np.sqrt(sse_slr / n)
    r2_slr = 1 - (sse_slr / Syy) if Syy != 0 else 0

    # ==================================================
    # MÓDULO 2: GOR Convencional
    # ==================================================
    # Pendiente GOR (m_gor)
    if Sxy != 0:
        beta1_num = (Syy - eta * Sxx) + np.sqrt((Syy - eta * Sxx)**2 + 4 * eta * Sxy**2)
        beta1_den = 2 * Sxy
        m_gor = beta1_num / beta1_den
    else:
        m_gor = 0
    b_gor = y_mean - m_gor * x_mean
    y_pred_gor = m_gor * X + b_gor

    # Proyecciones Ortogonales verdaderas (X_t, Y_t)
    X_t = (m_gor * (Y - b_gor) + eta * X) / (eta + m_gor**2)
    Y_t = b_gor + m_gor * X_t
    
    # Métricas de Error GOR Convencional
    # Varianza Residual Ortogonal
    s2_e_gor = (1 / (n - 2)) * np.sum(((Y - b_gor - m_gor * X)**2) / (m_gor**2 + eta)) if n > 2 else 0
    sigma_gor = np.sqrt(s2_e_gor) # Desviación Estándar Residual Ortogonal (Error Típico GOR)
    
    # Pseudo-métricas para la tabla comparativa (Ajuste sobre Y observado)
    sse_gor = np.sum((Y - y_pred_gor)**2)
    s2_e_pseudo_gor = sse_gor / (n - 2) if n > 2 else 0
    se_m_gor = np.sqrt(s2_e_pseudo_gor / Sxx) if Sxx != 0 and n > 2 else 0
    se_b_gor = se_m_gor * np.sqrt(np.sum(X**2) / n) if n > 0 else 0
    rmse_gor = np.sqrt(sse_gor / n)
    r2_gor = 1 - (sse_gor / Syy) if Syy != 0 else 0

    # ==================================================
    # MÓDULO 3: GOR Propuesto (Ranjit Das et al.)
    # ==================================================
    Y_t_mean = np.mean(Y_t)
    # Pendiente Propuesta (m_prop) e intersección (b_prop)
    if Sxx != 0:
        m_prop = np.sum((X - x_mean) * (Y_t - Y_t_mean)) / Sxx
    else:
        m_prop = 0
    b_prop = Y_t_mean - m_prop * x_mean
    y_pred_prop = m_prop * X + b_prop

    # Métricas de Error GOR Propuesto (comparado con datos reales observados Y)
    sse_prop = np.sum((Y - y_pred_prop)**2)
    s2_e_prop = sse_prop / (n - 2) if n > 2 else 0
    se_m_prop = np.sqrt(s2_e_prop / Sxx) if Sxx != 0 and n > 2 else 0
    se_b_prop = se_m_prop * np.sqrt(np.sum(X**2) / n) if n > 0 else 0
    rmse_prop = np.sqrt(sse_prop / n)
    r2_prop = 1 - (sse_prop / Syy) if Syy != 0 else 0

    # Construir DataFrame de Resultados
    df_results = df.copy()
    df_results["Y_est (SLR)"] = y_pred_slr
    df_results["Residuo (SLR)"] = Y - y_pred_slr
    
    df_results["Y_est (GOR Conv)"] = y_pred_gor
    df_results["Residuo (GOR Conv)"] = Y - y_pred_gor
    df_results["X_t (GOR)"] = X_t
    df_results["Y_t (GOR)"] = Y_t
    
    df_results["Y_est (GOR Prop)"] = y_pred_prop
    df_results["Residuo (GOR Prop)"] = Y - y_pred_prop


    # --- DASHBOARD COMPARATIVO CONSOLIDADO ---
    st.header("📊 Dashboard Comparativo Consolidado")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        # Gráfico Dinámico con Plotly
        fig_plotly = plgo.Figure()

        # Puntos Reales (Negro)
        fig_plotly.add_trace(plgo.Scatter(
            x=X, y=Y, mode='markers', name='Datos Observados',
            marker=dict(color='black', size=8),
            hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>'
        ))

        # Líneas de Tendencia (Continuas y nítidas)
        x_line = np.linspace(min(X), max(X), 100)
        
        fig_plotly.add_trace(plgo.Scatter(
            x=x_line, y=m_slr * x_line + b_slr, mode='lines', name='SLR',
            line=dict(color='blue', width=2),
            hovertemplate='X: %{x:.2f}<br>Y (SLR): %{y:.2f}<extra></extra>'
        ))
        
        fig_plotly.add_trace(plgo.Scatter(
            x=x_line, y=m_gor * x_line + b_gor, mode='lines', name='GOR Convencional',
            line=dict(color='orange', width=2),
            hovertemplate='X: %{x:.2f}<br>Y (GOR): %{y:.2f}<extra></extra>'
        ))
        
        fig_plotly.add_trace(plgo.Scatter(
            x=x_line, y=m_prop * x_line + b_prop, mode='lines', name='GOR Propuesto',
            line=dict(color='green', width=2),
            hovertemplate='X: %{x:.2f}<br>Y (Prop): %{y:.2f}<extra></extra>'
        ))

        fig_plotly.update_layout(
            title='Ajuste de los Modelos a los Datos Observados',
            xaxis_title='Variable Independiente (X)',
            yaxis_title='Variable Dependiente (Y)',
            hovermode='closest',
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_plotly, use_container_width=True)

    with col2:
        # Tabla Comparativa de Parámetros
        comp_data = {
            "Método": ["SLR", "GOR Conv.", "GOR Prop."],
            "Pendiente (m)": [m_slr, m_gor, m_prop],
            "Intercepción (b)": [b_slr, b_gor, b_prop],
            "SE(m)": [se_m_slr, se_m_gor, se_m_prop],
            "SE(b)": [se_b_slr, se_b_gor, se_b_prop],
            "RMSE": [rmse_slr, rmse_gor, rmse_prop],
            "R²": [r2_slr, r2_gor, r2_prop]
        }
        df_comp = pd.DataFrame(comp_data)
        st.markdown("**Comparativa de Métricas de los Modelos**")
        st.dataframe(df_comp.style.format({
            "Pendiente (m)": "{:.4f}",
            "Intercepción (b)": "{:.4f}",
            "SE(m)": "{:.4f}",
            "SE(b)": "{:.4f}",
            "RMSE": "{:.4f}",
            "R²": "{:.4f}"
        }), use_container_width=True)
        
        diff_pendiente = abs(m_slr - m_prop) / abs(m_slr) * 100 if m_slr != 0 else 0
        st.info(f"**Insight:** La diferencia de pendiente entre SLR y GOR Propuesto es del **{diff_pendiente:.2f}%**.")


    st.markdown("---")

    # --- MÓDULOS EDUCATIVOS ---
    st.header("📖 Módulos Educativos y Fundamentos Matemáticos")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. SLR (Mínimos Cuadrados)", 
        "2. GOR Convencional", 
        "3. GOR Propuesto", 
        "📥 Datos y Exportación"
    ])

    with tab1:
        st.subheader("Módulo 1: Regresión Lineal por Mínimos Cuadrados Ordinarios (SLR / OLS)")
        st.markdown("""
        **Enfoque Educativo:** 
        Este método asume que la variable independiente ($X$) es perfecta y **no tiene error de medición**, asignando toda la incertidumbre o error al eje dependiente ($Y$). Por tanto, minimiza las distancias verticales (residuos) entre los puntos y la recta.
        """)
        
        st.markdown("### Fórmulas Matemáticas y Resultados")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Pendiente ($m$):**")
            st.latex(r"m = \frac{n\sum(XY) - (\sum X)(\sum Y)}{n\sum X^2 - (\sum X)^2} \text{ o } m = \frac{S_{xy}}{S_{xx}}")
            st.latex(rf"\Rightarrow m = {m_slr:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Varianza Residual ($s_e^2$):**")
            st.latex(r"s_e^2 = \frac{\sum(Y_{obs} - Y_{pred})^2}{n-2}")
            st.latex(rf"\Rightarrow s_e^2 = {s2_e_slr:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_f2:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Intersección ($b$):**")
            st.latex(r"b = \bar{Y} - m\bar{X}")
            st.latex(rf"\Rightarrow b = {b_slr:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Error Estándar de la Pendiente ($SE_m$):**")
            st.latex(r"SE_m = \sqrt{\frac{s_e^2}{\sum(X_i - \bar{X})^2}}")
            st.latex(rf"\Rightarrow SE_m = {se_m_slr:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
        st.markdown("**Error Estándar de la Intercepción ($SE_b$):**")
        st.latex(r"SE_b = SE_m \sqrt{\frac{\sum X_i^2}{n}}")
        st.latex(rf"\Rightarrow SE_b = {se_b_slr:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)


    with tab2:
        st.subheader("Módulo 2: Regresión Ortogonal Generalizada (GOR Convencional)")
        st.markdown("""
        **Enfoque Educativo:** 
        En la naturaleza (como en sismología o ciencias físicas), **ambos instrumentos miden con error**. 
        El GOR supera la limitación del SLR al minimiza la distancia *perpendicular* (ortogonal) a la recta, ponderada por la relación de varianzas de error de ambos ejes, denotada por $\eta$ (Eta).
        """)
        
        st.markdown("### Fórmulas Matemáticas y Resultados")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Parámetro de Relación de Varianzas ($\eta$):**")
            st.latex(r"\eta = \frac{\sigma^2_{\varepsilon y}}{\sigma^2_{\varepsilon x}}")
            st.latex(rf"\Rightarrow \eta = {eta:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Pendiente GOR ($\hat{\beta}_1$):**")
            st.latex(r"\hat{\beta}_1 = \frac{(S_{yy} - \eta S_{xx}) + \sqrt{(S_{yy} - \eta S_{xx})^2 + 4 \eta S_{xy}^2}}{2 S_{xy}}")
            st.latex(rf"\Rightarrow \hat{{\beta}}_1 = {m_gor:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Proyecciones Ortogonales Verdaderas (puntos corregidos $X_t, Y_t$):**")
            st.latex(r"X_t = \frac{\hat{\beta}_1(Y_{obs} - \hat{\beta}_0) + \eta X_{obs}}{\eta + \hat{\beta}_1^2}")
            st.latex(r"Y_t = \hat{\beta}_0 + \hat{\beta}_1 X_t")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_f2:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Intersección GOR ($\hat{\beta}_0$):**")
            st.latex(r"\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}")
            st.latex(rf"\Rightarrow \hat{{\beta}}_0 = {b_gor:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Varianza Residual Ortogonal ($\hat{\sigma}^2$):**")
            st.latex(r"\hat{\sigma}^2 = \frac{1}{n-2} \sum_{i=1}^{n} \frac{(Y_i - \hat{\beta}_0 - \hat{\beta}_1 X_i)^2}{\hat{\beta}_1^2 + \eta}")
            st.latex(rf"\Rightarrow \hat{{\sigma}}^2 = {s2_e_gor:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Desviación Estándar Residual Ortogonal (Error Típico GOR):**")
            st.latex(r"\hat{\sigma} = \sqrt{\hat{\sigma}^2}")
            st.latex(rf"\Rightarrow \hat{{\sigma}} = {sigma_gor:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.subheader("Módulo 3: Regresión Ortogonal Propuesta (Ranjit Das et al.)")
        st.markdown("""
        **Enfoque Educativo:** 
        Esta es la innovación del modelo propuesto por Das. Utiliza las **proyecciones ortogonales verdaderas ($Y_t$)** calculadas en el GOR convencional para ajustar una recta lineal insesgada final contra los valores observados de $X$.
        Esto facilita enormemente su aplicación directa en sistemas operativos de monitoreo.
        """)
        
        st.markdown("### Fórmulas Matemáticas y Resultados")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Pendiente Propuesta ($c_1$):**")
            st.latex(r"c_1 = \frac{\sum (X_{obs,i} - \bar{X}_{obs})(Y_{t,i} - \bar{Y}_t)}{\sum (X_{obs,i} - \bar{X}_{obs})^2}")
            st.latex(rf"\Rightarrow c_1 = {m_prop:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Ecuación Predictiva Final:**")
            st.latex(r"Y_{t\_propuesto} = c_1 X_{obs} + c_2")
            st.markdown(f"**Ecuación calculada:** $Y = {m_prop:.4f}X + ({b_prop:.4f})$")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_f2:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Intersección Propuesta ($c_2$):**")
            st.latex(r"c_2 = \bar{Y}_t - c_1 \bar{X}_{obs}")
            st.latex(rf"\Rightarrow c_2 = {b_prop:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**RMSE (Ajuste final contra datos reales $Y_{obs}$):**")
            st.latex(r"RMSE = \sqrt{\frac{\sum (Y_{obs} - Y_{t\_propuesto})^2}{n}}")
            st.latex(rf"\Rightarrow RMSE = {rmse_prop:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.subheader("Datos Originales y Proyecciones")
        st.markdown("Tabla detallada con cada dato original, sus predicciones y los residuos según cada modelo.")
        st.dataframe(df_results, use_container_width=True)

        st.markdown("---")
        st.subheader("Exportar Resultados")
        col_out1, col_out2 = st.columns(2)
        
        with col_out1:
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
            st.markdown("**Descargar PDF:**")
            
            def create_pdf():
                # Crear imagen de Matplotlib estática en background para el PDF
                fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5))
                ax_pdf.scatter(X, Y, color='black', label='Datos Observados')
                x_vals = np.linspace(min(X), max(X), 100)
                ax_pdf.plot(x_vals, m_slr * x_vals + b_slr, color='blue', label='SLR')
                ax_pdf.plot(x_vals, m_gor * x_vals + b_gor, color='orange', label='GOR Conv')
                ax_pdf.plot(x_vals, m_prop * x_vals + b_prop, color='green', label='GOR Prop')
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
                
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(img_buffer.getvalue())
                    tmp_path = tmp_file.name
                pdf.image(tmp_path, x=15, w=180)
                os.unlink(tmp_path)
                
                pdf.ln(5)
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, '1. Resumen de Modelos', 0, 1)
                pdf.set_font('Arial', '', 10)
                
                for i, row in df_comp.iterrows():
                    eq = f"Y = {row['Pendiente (m)']:.4f}X + {row['Intercepción (b)']:.4f}"
                    pdf.cell(0, 6, f"{row['Método']}: {eq} | RMSE: {row['RMSE']:.4f} | R2: {row['R²']:.4f}", 0, 1)
                
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

else:
    st.info("👈 Por favor, ingresa o sube datos con al menos 2 puntos para comenzar el análisis comparativo.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 14px;'>"
    "Desarrollado y mantenido por <b>Alexander Acosta</b> "
    "(<a href='https://github.com/j-alexander-acosta' target='_blank' style='color: #1f77b4; text-decoration: none;'>@j-alexander-acosta</a>)"
    "</p>", 
    unsafe_allow_html=True
)
