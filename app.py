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
    /* Ocultar el botón de Deploy */
    .stDeployButton, .stAppDeployButton {
        display: none !important;
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

    st.sidebar.markdown("---")
    st.sidebar.header("3. Regresión No Lineal")
    nl_model = st.sidebar.selectbox(
        "Selecciona el modelo no lineal:",
        ("Exponencial (Y = a + b * e^(c*X))", "Logarítmico (Y = a + b * ln(X))", "Potencial (Y = a * X^b)", "Cuadrático (Y = a*X^2 + b*X + c)")
    )




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

    # ==================================================
    # MÓDULO 4: Regresión No Lineal
    # ==================================================
    nl_valid = True
    nl_error_msg = ""
    y_pred_nl = None
    nl_equation = ""
    a_nl, b_nl, c_nl = 0, 0, 0
    A_nl = 0

    if nl_model.startswith("Exponencial"):
        # Ajuste no lineal y = a + b * e^(c * x) utilizando curve_fit de scipy
        try:
            y_range = np.max(Y) - np.min(Y)
            if y_range == 0:
                y_range = 1.0

            # Estimación inicial robusta (a0, b0, c0) basada en la tendencia general
            if Sxy >= 0:
                a_0 = np.min(Y) - 0.1 * y_range - 1e-4
                diff = Y - a_0
                ln_diff = np.log(diff)
                c_0 = np.sum((X - x_mean) * (ln_diff - np.mean(ln_diff))) / Sxx if Sxx != 0 else 0.1
                A_0 = np.mean(ln_diff) - c_0 * x_mean
                b_0 = np.exp(A_0)
            else:
                a_0 = np.max(Y) + 0.1 * y_range + 1e-4
                diff = a_0 - Y
                ln_diff = np.log(diff)
                c_0 = np.sum((X - x_mean) * (ln_diff - np.mean(ln_diff))) / Sxx if Sxx != 0 else -0.1
                A_0 = np.mean(ln_diff) - c_0 * x_mean
                b_0 = -np.exp(A_0)

            # Función del modelo
            def exp_model_func(x, a, b, c):
                return a + b * np.exp(c * x)

            # Ajuste con curve_fit
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(exp_model_func, X, Y, p0=[a_0, b_0, c_0], maxfev=10000)
            a_nl, b_nl, c_nl = popt[0], popt[1], popt[2]

            y_pred_nl = exp_model_func(X, a_nl, b_nl, c_nl)
            
            # Formatear la ecuación final para presentación
            sign_b = "+" if b_nl >= 0 else "-"
            abs_b = abs(b_nl)
            nl_equation = rf"Y = {a_nl:.4f} {sign_b} {abs_b:.4f} \cdot e^{{{c_nl:.4f} X}}"
        except Exception as e:
            nl_valid = False
            nl_error_msg = f"No se pudo ajustar el modelo Exponencial de 3 parámetros. Error: {e}"




    elif nl_model.startswith("Logarítmico"):
        if np.any(X <= 0):
            nl_valid = False
            nl_error_msg = "El modelo Logarítmico requiere que todos los valores de X sean estrictamente mayores a cero (X > 0) para calcular ln(X)."
        else:
            ln_X = np.log(X)
            ln_X_mean = np.mean(ln_X)
            S_lnX_lnX = np.sum((ln_X - ln_X_mean)**2)
            S_lnX_Y = np.sum((ln_X - ln_X_mean) * (Y - y_mean))
            b_nl = S_lnX_Y / S_lnX_lnX if S_lnX_lnX != 0 else 0
            a_nl = y_mean - b_nl * ln_X_mean
            y_pred_nl = a_nl + b_nl * np.log(X)
            nl_equation = rf"Y = {a_nl:.4f} + {b_nl:.4f} \cdot \ln(X)"

    elif nl_model.startswith("Potencial"):
        if np.any(X <= 0) or np.any(Y <= 0):
            nl_valid = False
            nl_error_msg = "El modelo Potencial requiere que todos los valores tanto de X como de Y sean estrictamente mayores a cero (X > 0, Y > 0) para aplicar ln(X) y ln(Y)."
        else:
            ln_X = np.log(X)
            ln_Y = np.log(Y)
            ln_X_mean = np.mean(ln_X)
            ln_Y_mean = np.mean(ln_Y)
            S_lnX_lnX = np.sum((ln_X - ln_X_mean)**2)
            S_lnX_lnY = np.sum((ln_X - ln_X_mean) * (ln_Y - ln_Y_mean))
            b_nl = S_lnX_lnY / S_lnX_lnX if S_lnX_lnX != 0 else 0
            A_nl = ln_Y_mean - b_nl * ln_X_mean
            a_nl = np.exp(A_nl)
            y_pred_nl = a_nl * (X ** b_nl)
            nl_equation = rf"Y = {a_nl:.4f} \cdot X^{{{b_nl:.4f}}}"

    elif nl_model.startswith("Cuadrático"):
        if n < 3:
            nl_valid = False
            nl_error_msg = "El modelo Cuadrático requiere al menos 3 puntos de datos para calcular un ajuste único. Por favor, agrega más puntos en la barra lateral."
        else:
            coefs = np.polyfit(X, Y, 2)
            a_nl, b_nl, c_nl = coefs[0], coefs[1], coefs[2]
            y_pred_nl = a_nl * (X**2) + b_nl * X + c_nl
            nl_equation = rf"Y = {a_nl:.4f} X^2 + {b_nl:.4f} X + ({c_nl:.4f})"

    if nl_valid:
        sse_nl = np.sum((Y - y_pred_nl)**2)
        rmse_nl = np.sqrt(sse_nl / n)
        r2_nl = 1 - (sse_nl / Syy) if Syy != 0 else 0
    else:
        sse_nl = None
        rmse_nl = None
        r2_nl = None

    # ==================================================
    # MÓDULO 5: Power Law Model (Y = a * X^b)
    # ==================================================
    power_valid = True
    power_error_msg = ""
    y_pred_power = None
    power_equation = ""
    a_power, b_power = 0, 0
    A_power = 0

    if np.any(X <= 0) or np.any(Y <= 0):
        power_valid = False
        power_error_msg = "El modelo Power Law (Y = a * X^b) requiere que todos los valores tanto de X como de Y sean estrictamente mayores a cero para poder aplicar la transformación logarítmica dual ln(X) y ln(Y)."
    else:
        ln_X = np.log(X)
        ln_Y = np.log(Y)
        ln_X_mean = np.mean(ln_X)
        ln_Y_mean = np.mean(ln_Y)
        S_lnX_lnX = np.sum((ln_X - ln_X_mean)**2)
        S_lnX_lnY = np.sum((ln_X - ln_X_mean) * (ln_Y - ln_Y_mean))
        
        b_power = S_lnX_lnY / S_lnX_lnX if S_lnX_lnX != 0 else 0
        A_power = ln_Y_mean - b_power * ln_X_mean
        a_power = np.exp(A_power)
        
        y_pred_power = a_power * (X ** b_power)
        power_equation = rf"Y = {a_power:.4f} \cdot X^{{{b_power:.4f}}}"
        
        sse_power = np.sum((Y - y_pred_power)**2)
        rmse_power = np.sqrt(sse_power / n)
        r2_power = 1 - (sse_power / Syy) if Syy != 0 else 0

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

    if nl_valid:
        df_results[f"Y_est ({nl_model.split(' ')[0]})"] = y_pred_nl
        df_results[f"Residuo ({nl_model.split(' ')[0]})"] = Y - y_pred_nl

    if power_valid:
        df_results["Y_est (Power Law)"] = y_pred_power
        df_results["Residuo (Power Law)"] = Y - y_pred_power




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

        if nl_valid:
            x_line_nl = np.linspace(min(X), max(X), 200)
            if nl_model.startswith("Exponencial"):
                y_line_nl = a_nl + b_nl * np.exp(c_nl * x_line_nl)



            elif nl_model.startswith("Logarítmico"):
                x_line_safe = np.maximum(x_line_nl, 1e-9)
                y_line_nl = a_nl + b_nl * np.log(x_line_safe)
            elif nl_model.startswith("Potencial"):
                x_line_safe = np.maximum(x_line_nl, 1e-9)
                y_line_nl = a_nl * (x_line_safe ** b_nl)
            elif nl_model.startswith("Cuadrático"):
                y_line_nl = a_nl * (x_line_nl ** 2) + b_nl * x_line_nl + c_nl
            
            fig_plotly.add_trace(plgo.Scatter(
                x=x_line_nl, y=y_line_nl, mode='lines', name=f"{nl_model.split(' ')[0]} (No Lin.)",
                line=dict(color='purple', width=2.5, dash='dashdot'),
                hovertemplate='X: %{x:.2f}<br>Y (No Lin.): %{y:.2f}<extra></extra>'
            ))

        if power_valid:
            x_line_power = np.linspace(min(X), max(X), 200)
            x_line_power_safe = np.maximum(x_line_power, 1e-9)
            y_line_power = a_power * (x_line_power_safe ** b_power)
            
            fig_plotly.add_trace(plgo.Scatter(
                x=x_line_power, y=y_line_power, mode='lines', name="Power Law",
                line=dict(color='coral', width=2.5, dash='dash'),
                hovertemplate='X: %{x:.2f}<br>Y (Power Law): %{y:.2f}<extra></extra>'
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
        if nl_valid:
            comp_data["Método"].append(f"{nl_model.split(' ')[0]} (No Lin.)")
            comp_data["Pendiente (m)"].append(np.nan)
            comp_data["Intercepción (b)"].append(np.nan)
            comp_data["SE(m)"].append(np.nan)
            comp_data["SE(b)"].append(np.nan)
            comp_data["RMSE"].append(rmse_nl)
            comp_data["R²"].append(r2_nl)

        if power_valid:
            comp_data["Método"].append("Power Law")
            comp_data["Pendiente (m)"].append(np.nan)
            comp_data["Intercepción (b)"].append(np.nan)
            comp_data["SE(m)"].append(np.nan)
            comp_data["SE(b)"].append(np.nan)
            comp_data["RMSE"].append(rmse_power)
            comp_data["R²"].append(r2_power)


        df_comp = pd.DataFrame(comp_data)
        st.markdown("**Comparativa de Métricas de los Modelos**")
        st.dataframe(df_comp.style.format({
            "Pendiente (m)": "{:.4f}",
            "Intercepción (b)": "{:.4f}",
            "SE(m)": "{:.4f}",
            "SE(b)": "{:.4f}",
            "RMSE": "{:.4f}",
            "R²": "{:.4f}"
        }, na_rep="-"), use_container_width=True)
        
        diff_pendiente = abs(m_slr - m_prop) / abs(m_slr) * 100 if m_slr != 0 else 0
        st.info(f"**Insight:** La diferencia de pendiente entre SLR y GOR Propuesto es del **{diff_pendiente:.2f}%**.")



    st.markdown("---")

    # --- MÓDULOS EDUCATIVOS ---
    st.header("📖 Módulos Educativos y Fundamentos Matemáticos")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. SLR (Mínimos Cuadrados)", 
        "2. GOR Convencional", 
        "3. GOR Propuesto", 
        "4. Regresión No Lineal", 
        "5. Power Law Model",
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
        st.subheader("Módulo 4: Regresión No Lineal")
        if not nl_valid:
            st.warning(nl_error_msg)
            st.markdown("""
            **Nota Educativa:**
            Los modelos no lineales (como el Exponencial, Logarítmico y Potencial) se resuelven aplicando una transformación logarítmica para linealizar la relación. 
            Dado que la función logaritmo natural $\\ln(z)$ está definida únicamente para $z > 0$, el conjunto de datos actual contiene valores no permitidos para este tipo de regresión.
            
            *Sugerencia:* Si deseas explorar estos modelos, modifica tus datos en la barra lateral para que todos los valores correspondientes sean estrictamente positivos, o selecciona el **Modelo Cuadrático** (que no posee restricciones de dominio).
            """)
        else:
            st.markdown(f"""
            **Enfoque Educativo:**
            El modelo **{nl_model.split(' ')[0]}** describe una relación no lineal en el espacio original de datos.
            Para ajustarlo usando mínimos cuadrados, aplicamos una transformación a los datos (linealización), calculamos el ajuste sobre la recta transformada y luego aplicamos el operador inverso para recuperar los coeficientes originales.
            """)
            
            st.markdown("### Fórmulas Matemáticas y Resultados")
            col_f1, col_f2 = st.columns(2)
            
            if nl_model.startswith("Exponencial"):
                with col_f1:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Ecuación Original:**")
                    st.latex(r"Y = a + b \cdot e^{c X}")
                    st.markdown("Presencia de una constante aditiva ($a$) que actúa como asíntota.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Estimación Inicial Semilla ($a_0, b_0, c_0$):**")
                    if Sxy >= 0:
                        st.markdown("Relación positiva: asíntota inferior $a_0 < \min(Y)$")
                        st.latex(r"\ln(Y - a_0) = \ln(b) + c X")
                    else:
                        st.markdown("Relación negativa: asíntota superior $a_0 > \max(Y)$")
                        st.latex(r"\ln(a_0 - Y) = \ln(-b) + c X")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_f2:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Coeficientes del Ajuste No Lineal:**")
                    st.latex(rf"\Rightarrow a = {a_nl:.4f}")
                    st.latex(rf"\Rightarrow b = {b_nl:.4f}")
                    st.latex(rf"\Rightarrow c = {c_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Método de Ajuste:**")
                    st.markdown("Ajuste por mínimos cuadrados no lineales iterativos (algoritmo Levenberg-Marquardt) para minimizar los residuos en el espacio real de datos.")
                    st.markdown("</div>", unsafe_allow_html=True)



                    
            elif nl_model.startswith("Logarítmico"):
                with col_f1:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Ecuación Original y Linealización:**")
                    st.latex(r"Y = a + b \ln(X)")
                    st.markdown("Definiendo $X' = \\ln(X)$, ajustamos la recta:")
                    st.latex(r"Y = a + b X'")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Coeficientes del Modelo:**")
                    st.latex(rf"\Rightarrow a = {a_nl:.4f}")
                    st.latex(rf"\Rightarrow b = {b_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_f2:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Pendiente ($b$):**")
                    st.latex(r"b = \frac{\sum (\ln(X_i) - \bar{\ln(X)})(Y_i - \bar{Y})}{\sum (\ln(X_i) - \bar{\ln(X)})^2}")
                    st.latex(rf"\Rightarrow b = {b_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Intersección ($a$):**")
                    st.latex(r"a = \bar{Y} - b \bar{\ln(X)}")
                    st.latex(rf"\Rightarrow a = {a_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            elif nl_model.startswith("Potencial"):
                with col_f1:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Ecuación Original y Linealización:**")
                    st.latex(r"Y = a \cdot X^b \implies \ln(Y) = \ln(a) + b \ln(X)")
                    st.markdown("Definiendo $X' = \\ln(X)$, $Y' = \\ln(Y)$ y $A = \\ln(a)$, ajustamos:")
                    st.latex(r"Y' = A + b X'")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Recuperación de Coeficientes:**")
                    st.latex(rf"b = {b_nl:.4f}")
                    st.latex(r"a = e^A")
                    st.latex(rf"\Rightarrow a = e^{{{A_nl:.4f}}} = {a_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_f2:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Pendiente Linealizada ($b$):**")
                    st.latex(r"b = \frac{\sum (\ln(X_i) - \bar{\ln(X)})(\ln(Y_i) - \bar{\ln(Y)})}{\sum (\ln(X_i) - \bar{\ln(X)})^2}")
                    st.latex(rf"\Rightarrow b = {b_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Intersección Linealizada ($A$):**")
                    st.latex(r"A = \bar{\ln(Y)} - b \bar{\ln(X)}")
                    st.latex(rf"\Rightarrow A = {A_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            elif nl_model.startswith("Cuadrático"):
                with col_f1:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Ecuación del Polinomio de Segundo Grado:**")
                    st.latex(r"Y = a X^2 + b X + c")
                    st.markdown("A pesar de ser curvilíneo, es **lineal en sus parámetros** ($a, b, c$).")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Coeficientes Calculados:**")
                    st.latex(rf"\Rightarrow a = {a_nl:.4f}")
                    st.latex(rf"\Rightarrow b = {b_nl:.4f}")
                    st.latex(rf"\Rightarrow c = {c_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_f2:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown("**Sistema de Ecuaciones Normales:**")
                    st.latex(r"\begin{pmatrix} \sum X_i^4 & \sum X_i^3 & \sum X_i^2 \\ \sum X_i^3 & \sum X_i^2 & \sum X_i \\ \sum X_i^2 & \sum X_i & n \end{pmatrix} \begin{pmatrix} a \\ b \\ c \end{pmatrix} = \begin{pmatrix} \sum X_i^2 Y_i \\ \sum X_i Y_i \\ \sum Y_i \end{pmatrix}")
                    st.markdown("Se resuelve directamente por álgebra matricial lineal.")
                    st.markdown("</div>", unsafe_allow_html=True)

            st.success(f"**Ecuación ajustada:** ${nl_equation}$")
            
            # Comparación específica de métricas
            st.markdown("### Comparativa de Métricas: SLR vs No Lineal")
            metrics_nl_data = {
                "Métrica": ["Suma de Errores al Cuadrado (SSE)", "Error Cuadrático Medio (RMSE)", "Coeficiente de Determinación (R²)"],
                "SLR (Lineal)": [sse_slr, rmse_slr, r2_slr],
                f"No Lineal ({nl_model.split(' ')[0]})": [sse_nl, rmse_nl, r2_nl]
            }
            df_metrics_nl = pd.DataFrame(metrics_nl_data)
            st.dataframe(df_metrics_nl.style.format({
                "SLR (Lineal)": "{:.4f}",
                f"No Lineal ({nl_model.split(' ')[0]})": "{:.4f}"
            }), use_container_width=True)
            
            # Calculadora de predicción interactiva
            st.markdown("---")
            st.markdown("### 🧮 Calculadora de Predicciones Interactiva")
            x_input = st.number_input(f"Ingresa un valor de X para calcular Y ({nl_model.split(' ')[0]}):", value=float(np.mean(X)), format="%.4f")
            
            y_pred_calc = None
            if nl_model.startswith("Exponencial"):
                y_pred_calc = a_nl + b_nl * np.exp(c_nl * x_input)
                sign_b = "+" if b_nl >= 0 else "-"
                abs_b = abs(b_nl)
                calc_latex = rf"Y = {a_nl:.4f} {sign_b} {abs_b:.4f} \cdot e^{{{c_nl:.4f} \cdot {x_input:.4f}}} = {y_pred_calc:.4f}"



            elif nl_model.startswith("Logarítmico"):
                if x_input <= 0:
                    st.error("Error: Para el modelo Logarítmico, el valor de entrada X debe ser estrictamente mayor a cero.")
                else:
                    y_pred_calc = a_nl + b_nl * np.log(x_input)
                    calc_latex = rf"Y = {a_nl:.4f} + {b_nl:.4f} \cdot \ln({x_input:.4f}) = {y_pred_calc:.4f}"
            elif nl_model.startswith("Potencial"):
                if x_input <= 0:
                    st.error("Error: Para el modelo Potencial, el valor de entrada X debe ser estrictamente mayor a cero.")
                else:
                    y_pred_calc = a_nl * (x_input ** b_nl)
                    calc_latex = rf"Y = {a_nl:.4f} \cdot {x_input:.4f}^{{{b_nl:.4f}}} = {y_pred_calc:.4f}"
            elif nl_model.startswith("Cuadrático"):
                y_pred_calc = a_nl * (x_input**2) + b_nl * x_input + c_nl
                calc_latex = rf"Y = {a_nl:.4f} \cdot ({x_input:.4f})^2 + {b_nl:.4f} \cdot ({x_input:.4f}) + ({c_nl:.4f}) = {y_pred_calc:.4f}"
                
            if y_pred_calc is not None:
                st.latex(calc_latex)
                st.info(f"**Resultado:** Para $X = {x_input:.4f}$, el valor de $Y$ estimado por el modelo no lineal es **{y_pred_calc:.4f}**.")

    with tab5:
        st.subheader("Módulo 5: Power Law Model (Modelo de Ley de Potencia)")
        if not power_valid:
            st.warning(power_error_msg)
            st.markdown("""
            **Nota Educativa:**
            El modelo de Ley de Potencia ($Y = a X^b$) requiere una transformación logarítmica dual en ambos ejes.
            Dado que la función logaritmo natural $\\ln(z)$ está definida únicamente para $z > 0$, el conjunto de datos actual contiene valores no permitidos (negativos o cero) en las variables.
            
            *Sugerencia:* Ajusta tus puntos de datos en la barra lateral para que todos los valores de X e Y sean estrictamente positivos ($>0$) para poder realizar este ajuste.
            """)
        else:
            st.markdown("""
            **Enfoque Educativo:**
            El modelo de **Ley de Potencia (Power Law)** es uno de los más importantes en sismología y ciencias naturales. 
            Describe relaciones de escala donde un cambio relativo en una cantidad produce un cambio relativo proporcional en la otra, independiente de la escala de tamaño de las cantidades.
            
            *Ejemplos en sismología:*
            - **Ley de Omori** para la frecuencia de réplicas tras un gran terremoto: $n(t) \propto t^{-p}$.
            - **Escalamiento del momento sísmico** con la longitud de ruptura de la falla: $M_0 \propto L^b$.
            - **Ley de Gutenberg-Richter** (expresada en número acumulativo de sismos contra tamaño/energía).
            """)
            
            st.markdown("### Fórmulas Matemáticas y Resultados")
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown("**Ecuación Original y Linealización:**")
                st.latex(r"Y = a \cdot X^b \implies \ln(Y) = \ln(a) + b \ln(X)")
                st.markdown("Definiendo $X' = \\ln(X)$, $Y' = \\ln(Y)$ y $A = \\ln(a)$, ajustamos:")
                st.latex(r"Y' = A + b X'")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown("**Recuperación de Coeficientes:**")
                st.latex(rf"b = {b_power:.4f}")
                st.latex(r"a = e^A")
                st.latex(rf"\Rightarrow a = e^{{{A_power:.4f}}} = {a_power:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_f2:
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown("**Pendiente Linealizada ($b$):**")
                st.latex(r"b = \frac{\sum (\ln(X_i) - \bar{\ln(X)})(\ln(Y_i) - \bar{\ln(Y)})}{\sum (\ln(X_i) - \bar{\ln(X)})^2}")
                st.latex(rf"\Rightarrow b = {b_power:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown("**Intersección Linealizada ($A$):**")
                st.latex(r"A = \bar{\ln(Y)} - b \bar{\ln(X)}")
                st.latex(rf"\Rightarrow A = {A_power:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)

            st.success(f"**Ecuación ajustada:** ${power_equation}$")
            
            # Comparativa de métricas
            st.markdown("### Comparativa de Métricas: SLR vs Power Law")
            metrics_power_data = {
                "Métrica": ["Suma de Errores al Cuadrado (SSE)", "Error Cuadrático Medio (RMSE)", "Coeficiente de Determinación (R²)"],
                "SLR (Lineal)": [sse_slr, rmse_slr, r2_slr],
                "Power Law (No Lineal)": [sse_power, rmse_power, r2_power]
            }
            df_metrics_power = pd.DataFrame(metrics_power_data)
            st.dataframe(df_metrics_power.style.format({
                "SLR (Lineal)": "{:.4f}",
                "Power Law (No Lineal)": "{:.4f}"
            }), use_container_width=True)
            
            # Calculadora de predicción interactiva
            st.markdown("---")
            st.markdown("### 🧮 Calculadora de Predicciones Interactiva")
            x_input = st.number_input("Ingresa un valor de X para calcular Y (Power Law):", value=float(np.mean(X)), format="%.4f")
            
            if x_input <= 0:
                st.error("Error: Para el modelo Power Law, el valor de entrada X debe ser estrictamente mayor a cero.")
            else:
                y_pred_calc = a_power * (x_input ** b_power)
                calc_latex = rf"Y = {a_power:.4f} \cdot {x_input:.4f}^{{{b_power:.4f}}} = {y_pred_calc:.4f}"
                st.latex(calc_latex)
                st.info(f"**Resultado:** Para $X = {x_input:.4f}$, el valor de $Y$ estimado por el modelo de Ley de Potencia es **{y_pred_calc:.4f}**.")

    with tab6:
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
                
                if nl_valid:
                    if nl_model.startswith("Exponencial"):
                        y_vals_nl = a_nl + b_nl * np.exp(c_nl * x_vals)

                    elif nl_model.startswith("Logarítmico"):
                        x_vals_safe = np.maximum(x_vals, 1e-9)
                        y_vals_nl = a_nl + b_nl * np.log(x_vals_safe)
                    elif nl_model.startswith("Potencial"):
                        x_vals_safe = np.maximum(x_vals, 1e-9)
                        y_vals_nl = a_nl * (x_vals_safe ** b_nl)
                    elif nl_model.startswith("Cuadrático"):
                        y_vals_nl = a_nl * (x_vals ** 2) + b_nl * x_vals + c_nl
                    
                    ax_pdf.plot(x_vals, y_vals_nl, color='purple', linestyle='-.', label=f"No Lineal ({nl_model.split(' ')[0]})")
                
                if power_valid:
                    x_vals_safe = np.maximum(x_vals, 1e-9)
                    y_vals_power = a_power * (x_vals_safe ** b_power)
                    ax_pdf.plot(x_vals, y_vals_power, color='coral', linestyle='--', label='Power Law')

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
                    if "No Lin." in row['Método']:
                        clean_eq = nl_equation.replace('\\cdot', '*').replace('\\ln', 'ln').replace('\\bar', '').replace('\\hat', '').replace('{', '').replace('}', '')
                        eq = clean_eq
                    elif "Power Law" in row['Método']:
                        clean_eq = power_equation.replace('\\cdot', '*').replace('{', '').replace('}', '')
                        eq = clean_eq
                    else:
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
