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
data_source = st.sidebar.radio("Selecciona el origen de los datos:", ("Ingreso Manual", "Subir Archivo (.csv, .xlsx)"))

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
        ("Exponencial (Y = a + b e^(cx))", "Logarítmico (Y = a + b ln(x))", "Potencial (Y = a x^b)", "Cuadrático (Y = ax^2 + bx + c)")
    )
    
    a_val = 0.0
    if nl_model.startswith("Exponencial"):
        Y_temp = df["Y"].values
        min_y = float(np.min(Y_temp))
        default_a = 5.0 if min_y > 5.0 else float(np.round(min_y - 1.0, 2))
        a_val = st.sidebar.number_input("Valor de la asíntota 'a':", value=default_a, step=0.1, format="%.4f")
        if a_val >= min_y:
            st.sidebar.error("⚠️ La asíntota 'a' debe ser estrictamente menor que el valor mínimo de Y (Y > a) para calcular ln(y - a).")




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
    # Pendiente (m_slr) e intercepción (c_slr)
    m_slr = Sxy / Sxx if Sxx != 0 else 0
    c_slr = y_mean - m_slr * x_mean
    y_pred_slr = m_slr * X + c_slr
    
    # Métricas de Error SLR
    sse_slr = np.sum((Y - y_pred_slr)**2)
    s2_e_slr = sse_slr / (n - 2) if n > 2 else 0 # Varianza Residual
    se_m_slr = np.sqrt(s2_e_slr / Sxx) if Sxx != 0 and n > 2 else 0 # Error estándar de la pendiente
    se_c_slr = se_m_slr * np.sqrt(np.sum(X**2) / n) if n > 0 else 0 # Error estándar de la intercepción
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
    # MÓDULO 4: Regresión No Lineal (Cálculos Globales para Todos los Modelos)
    # ==================================================
    
    # 4.1 Modelo Exponencial con Asíntota (Y = a + b * e^(cx))
    exp_asymp_valid = True
    exp_asymp_error_msg = ""
    a_nl_exp, b_nl_exp, c_nl_exp = 0.0, 0.0, 0.0
    A_nl_exp = 0.0
    y_minus_a = None
    y_trans = None
    sum_x = 0.0
    sum_x2 = 0.0
    sum_y_trans = 0.0
    sum_x_y_trans = 0.0
    r2_linearized_exp = 0.0
    y_pred_nl_exp = None
    sse_nl_exp = None
    rmse_nl_exp = None
    r2_nl_exp = None
    nl_equation_exp = ""

    try:
        a_nl_exp = a_val
        min_y = np.min(Y)
        if a_nl_exp >= min_y:
            exp_asymp_valid = False
            exp_asymp_error_msg = f"La asíntota a ({a_nl_exp:.4f}) debe ser estrictamente menor que el valor mínimo de Y ({min_y:.4f}) para calcular ln(y - a)."
        else:
            y_minus_a = Y - a_nl_exp
            y_trans = np.log(y_minus_a)
            
            sum_x = np.sum(X)
            sum_x2 = np.sum(X**2)
            sum_y_trans = np.sum(y_trans)
            sum_x_y_trans = np.sum(X * y_trans)
            
            den = n * sum_x2 - sum_x**2
            if den != 0:
                c_nl_exp = (n * sum_x_y_trans - sum_x * sum_y_trans) / den
                A_nl_exp = (sum_y_trans - c_nl_exp * sum_x) / n
            else:
                c_nl_exp = 0.0
                A_nl_exp = np.mean(y_trans)
            
            b_nl_exp = np.exp(A_nl_exp)
            y_pred_nl_exp = a_nl_exp + b_nl_exp * np.exp(c_nl_exp * X)
            
            y_trans_mean = np.mean(y_trans)
            S_y_trans_y_trans = np.sum((y_trans - y_trans_mean)**2)
            y_trans_pred = c_nl_exp * X + A_nl_exp
            sse_linearized_exp = np.sum((y_trans - y_trans_pred)**2)
            r2_linearized_exp = 1.0 - (sse_linearized_exp / S_y_trans_y_trans) if S_y_trans_y_trans != 0 else 1.0
            
            sse_nl_exp = np.sum((Y - y_pred_nl_exp)**2)
            rmse_nl_exp = np.sqrt(sse_nl_exp / n)
            r2_nl_exp = 1 - (sse_nl_exp / Syy) if Syy != 0 else 0
            
            sign_b = "+" if b_nl_exp >= 0 else "-"
            nl_equation_exp = rf"y = {a_nl_exp:.4f} {sign_b} {abs(b_nl_exp):.4f}e^{{{c_nl_exp:.4f}x}}"
    except Exception as e:
        exp_asymp_valid = False
        exp_asymp_error_msg = f"Error en los cálculos del modelo Exponencial con Asíntota: {e}"

    # 4.2 Modelo Logarítmico (Y = a + b * ln(x))
    log_valid = True
    log_error_msg = ""
    a_nl_log, b_nl_log = 0.0, 0.0
    y_pred_nl_log = None
    sse_nl_log = None
    rmse_nl_log = None
    r2_nl_log = None
    nl_equation_log = ""

    if np.any(X <= 0):
        log_valid = False
        log_error_msg = "El modelo Logarítmico requiere que todos los valores de X sean estrictamente mayores a cero (X > 0) para calcular ln(X)."
    else:
        try:
            ln_X = np.log(X)
            ln_X_mean = np.mean(ln_X)
            S_lnX_lnX = np.sum((ln_X - ln_X_mean)**2)
            S_lnX_Y = np.sum((ln_X - ln_X_mean) * (Y - y_mean))
            b_nl_log = S_lnX_Y / S_lnX_lnX if S_lnX_lnX != 0 else 0
            a_nl_log = y_mean - b_nl_log * ln_X_mean
            y_pred_nl_log = a_nl_log + b_nl_log * np.log(X)
            nl_equation_log = rf"y = {a_nl_log:.4f} + {b_nl_log:.4f}\ln(x)"
            
            sse_nl_log = np.sum((Y - y_pred_nl_log)**2)
            rmse_nl_log = np.sqrt(sse_nl_log / n)
            r2_nl_log = 1 - (sse_nl_log / Syy) if Syy != 0 else 0
        except Exception as e:
            log_valid = False
            log_error_msg = f"Error en los cálculos del modelo Logarítmico: {e}"

    # 4.3 Modelo Potencial / Power Law (Y = a * x^b) usando ln
    pot_valid = True
    pot_error_msg = ""
    a_nl_pot, b_nl_pot = 0.0, 0.0
    A_nl_pot = 0.0
    y_pred_nl_pot = None
    sse_nl_pot = None
    rmse_nl_pot = None
    r2_nl_pot = None
    nl_equation_pot = ""

    if np.any(X <= 0) or np.any(Y <= 0):
        pot_valid = False
        pot_error_msg = "El modelo Potencial requiere que todos los valores tanto de X como de Y sean estrictamente mayores a cero (X > 0, Y > 0) para aplicar ln(X) y ln(Y)."
    else:
        try:
            ln_X = np.log(X)
            ln_Y = np.log(Y)
            ln_X_mean = np.mean(ln_X)
            ln_Y_mean = np.mean(ln_Y)
            S_lnX_lnX = np.sum((ln_X - ln_X_mean)**2)
            S_lnX_lnY = np.sum((ln_X - ln_X_mean) * (ln_Y - ln_Y_mean))
            b_nl_pot = S_lnX_lnY / S_lnX_lnX if S_lnX_lnX != 0 else 0
            A_nl_pot = ln_Y_mean - b_nl_pot * ln_X_mean
            a_nl_pot = np.exp(A_nl_pot)
            y_pred_nl_pot = a_nl_pot * (X ** b_nl_pot)
            nl_equation_pot = rf"y = {a_nl_pot:.4f}x^{{{b_nl_pot:.4f}}}"
            
            sse_nl_pot = np.sum((Y - y_pred_nl_pot)**2)
            rmse_nl_pot = np.sqrt(sse_nl_pot / n)
            r2_nl_pot = 1 - (sse_nl_pot / Syy) if Syy != 0 else 0
        except Exception as e:
            pot_valid = False
            pot_error_msg = f"Error en los cálculos del modelo Potencial: {e}"

    # 4.4 Modelo Cuadrático (Y = ax^2 + bx + c)
    quad_valid = True
    quad_error_msg = ""
    a_nl_quad, b_nl_quad, c_nl_quad = 0.0, 0.0, 0.0
    y_pred_nl_quad = None
    sse_nl_quad = None
    rmse_nl_quad = None
    r2_nl_quad = None
    nl_equation_quad = ""

    if n < 3:
        quad_valid = False
        quad_error_msg = "El modelo Cuadrático requiere al menos 3 puntos de datos para calcular un ajuste único. Por favor, agrega más puntos en la barra lateral."
    else:
        try:
            coefs = np.polyfit(X, Y, 2)
            a_nl_quad, b_nl_quad, c_nl_quad = coefs[0], coefs[1], coefs[2]
            y_pred_nl_quad = a_nl_quad * (X**2) + b_nl_quad * X + c_nl_quad
            nl_equation_quad = rf"y = {a_nl_quad:.4f}x^2 + {b_nl_quad:.4f}x + {c_nl_quad:.4f}"
            
            sse_nl_quad = np.sum((Y - y_pred_nl_quad)**2)
            rmse_nl_quad = np.sqrt(sse_nl_quad / n)
            r2_nl_quad = 1 - (sse_nl_quad / Syy) if Syy != 0 else 0
        except Exception as e:
            quad_valid = False
            quad_error_msg = f"Error en los cálculos del modelo Cuadrático: {e}"

    # Asignar variables del modelo no lineal seleccionado en la barra lateral (para Tab 4)
    nl_valid = True
    nl_error_msg = ""
    y_pred_nl = None
    nl_equation = ""
    a_nl, b_nl, c_nl = 0.0, 0.0, 0.0
    A_nl = 0.0
    sse_nl = None
    rmse_nl = None
    r2_nl = None

    if nl_model.startswith("Exponencial"):
        nl_valid = exp_asymp_valid
        nl_error_msg = exp_asymp_error_msg
        a_nl, b_nl, c_nl = a_nl_exp, b_nl_exp, c_nl_exp
        A_nl = A_nl_exp
        y_pred_nl = y_pred_nl_exp
        nl_equation = nl_equation_exp
        sse_nl, rmse_nl, r2_nl = sse_nl_exp, rmse_nl_exp, r2_nl_exp
        # variables específicas para la tabla de sumatorias en tab 4
        if exp_asymp_valid:
            y_minus_a = Y - a_nl_exp
            y_trans = np.log(y_minus_a)
            sum_x = np.sum(X)
            sum_x2 = np.sum(X**2)
            sum_y_trans = np.sum(y_trans)
            sum_x_y_trans = np.sum(X * y_trans)
            r2_linearized = r2_linearized_exp
    elif nl_model.startswith("Logarítmico"):
        nl_valid = log_valid
        nl_error_msg = log_error_msg
        a_nl, b_nl = a_nl_log, b_nl_log
        y_pred_nl = y_pred_nl_log
        nl_equation = nl_equation_log
        sse_nl, rmse_nl, r2_nl = sse_nl_log, rmse_nl_log, r2_nl_log
    elif nl_model.startswith("Potencial"):
        nl_valid = pot_valid
        nl_error_msg = pot_error_msg
        a_nl, b_nl = a_nl_pot, b_nl_pot
        A_nl = A_nl_pot
        y_pred_nl = y_pred_nl_pot
        nl_equation = nl_equation_pot
        sse_nl, rmse_nl, r2_nl = sse_nl_pot, rmse_nl_pot, r2_nl_pot
    elif nl_model.startswith("Cuadrático"):
        nl_valid = quad_valid
        nl_error_msg = quad_error_msg
        a_nl, b_nl, c_nl = a_nl_quad, b_nl_quad, c_nl_quad
        y_pred_nl = y_pred_nl_quad
        nl_equation = nl_equation_quad
        sse_nl, rmse_nl, r2_nl = sse_nl_quad, rmse_nl_quad, r2_nl_quad

    # ==================================================
    # MÓDULO 5: Power Law Model (Duplicate de Potencial para Tab 5)
    # ==================================================
    power_valid = pot_valid
    power_error_msg = pot_error_msg
    a_power, b_power = a_nl_pot, b_nl_pot
    A_power = A_nl_pot
    y_pred_power = y_pred_nl_pot
    power_equation = nl_equation_pot
    sse_power = sse_nl_pot
    rmse_power = rmse_nl_pot
    r2_power = r2_nl_pot
    x_trans_power = np.log(X) if pot_valid else None
    y_trans_power = np.log(Y) if pot_valid else None
    sum_x_trans_power = np.sum(x_trans_power) if pot_valid else 0.0
    sum_x2_trans_power = np.sum(x_trans_power**2) if pot_valid else 0.0
    sum_y_trans_power = np.sum(y_trans_power) if pot_valid else 0.0
    sum_xy_trans_power = np.sum(x_trans_power * y_trans_power) if pot_valid else 0.0
    r2_linearized_power = 0.0
    if pot_valid:
        y_trans_power_mean = np.mean(y_trans_power)
        S_y_trans_power_y_trans_power = np.sum((y_trans_power - y_trans_power_mean)**2)
        y_trans_power_pred = b_power * x_trans_power + A_power
        sse_linearized_power = np.sum((y_trans_power - y_trans_power_pred)**2)
        r2_linearized_power = 1.0 - (sse_linearized_power / S_y_trans_power_y_trans_power) if S_y_trans_power_y_trans_power != 0 else 1.0

    # ==================================================
    # MÓDULO 6: Función Potencia (base 10) (Cálculos Globales)
    # ==================================================
    fun_pot_valid = True
    fun_pot_error_msg = ""
    a_correct, b_correct = 0.0, 0.0
    a_err, b_err = 0.0, 0.0
    m_log, c_log = 0.0, 0.0
    r2_lin_pot10 = 0.0
    X_log, Y_log = None, None
    X_log_mean, Y_log_mean = 0.0, 0.0
    S_xx_log, S_xy_log, S_yy_log = 0.0, 0.0, 0.0
    y_pred_pot10_correct = None
    y_pred_pot10_incorrect = None
    sse_pot10_correct, rmse_pot10_correct, r2_pot10_correct = None, None, None
    sse_pot10_incorrect, rmse_pot10_incorrect, r2_pot10_incorrect = None, None, None

    if np.any(X <= 0) or np.any(Y <= 0):
        fun_pot_valid = False
        fun_pot_error_msg = "⚠️ El modelo Función Potencia requiere que todos los valores de X e Y sean estrictamente mayores a cero (>0) para aplicar el logaritmo base 10 (log10)."
    else:
        try:
            X_log = np.log10(X)
            Y_log = np.log10(Y)
            X_log_mean = np.mean(X_log)
            Y_log_mean = np.mean(Y_log)

            S_xx_log = np.sum((X_log - X_log_mean)**2)
            S_xy_log = np.sum((X_log - X_log_mean) * (Y_log - Y_log_mean))
            S_yy_log = np.sum((Y_log - Y_log_mean)**2)

            m_log = S_xy_log / S_xx_log if S_xx_log != 0 else 0.0
            c_log = Y_log_mean - m_log * X_log_mean

            r2_lin_pot10 = (S_xy_log**2) / (S_xx_log * S_yy_log) if (S_xx_log * S_yy_log) != 0 else 0.0

            a_correct = 10**c_log
            b_correct = m_log

            a_err = 10**c_log
            b_err = 10**m_log

            y_pred_pot10_correct = a_correct * (X ** b_correct)
            sse_pot10_correct = np.sum((Y - y_pred_pot10_correct)**2)
            rmse_pot10_correct = np.sqrt(sse_pot10_correct / n)
            r2_pot10_correct = 1 - (sse_pot10_correct / Syy) if Syy != 0 else 0

            y_pred_pot10_incorrect = a_err * (X ** b_err)
            sse_pot10_incorrect = np.sum((Y - y_pred_pot10_incorrect)**2)
            rmse_pot10_incorrect = np.sqrt(sse_pot10_incorrect / n)
            r2_pot10_incorrect = 1 - (sse_pot10_incorrect / Syy) if Syy != 0 else 0
        except Exception as e:
            fun_pot_valid = False
            fun_pot_error_msg = f"Error en los cálculos de Función Potencia (base 10): {e}"

    # ==================================================
    # MÓDULO 7: Función Exponencial Simple (ln) (Cálculos Globales)
    # ==================================================
    fun_exp_valid = True
    fun_exp_error_msg = ""
    b_coeff, ln_a = 0.0, 0.0
    r2_lin_exp = 0.0
    a_coeff = 0.0
    Y_log_exp = None
    X_mean_exp, Y_log_mean_exp = 0.0, 0.0
    S_xx_exp, S_xy_log_exp, S_yy_log_exp = 0.0, 0.0, 0.0
    y_pred_exp_simple = None
    sse_exp_simple, rmse_exp_simple, r2_exp_simple = None, None, None

    if np.any(Y <= 0):
        fun_exp_valid = False
        fun_exp_error_msg = "⚠️ El modelo Exponencial requiere que todos los valores de Y sean estrictamente mayores a cero (>0) para aplicar el logaritmo natural (ln)."
    else:
        try:
            Y_log_exp = np.log(Y)
            X_mean_exp = np.mean(X)
            Y_log_mean_exp = np.mean(Y_log_exp)

            S_xx_exp = np.sum((X - X_mean_exp)**2)
            S_xy_log_exp = np.sum((X - X_mean_exp) * (Y_log_exp - Y_log_mean_exp))
            S_yy_log_exp = np.sum((Y_log_exp - Y_log_mean_exp)**2)

            b_coeff = S_xy_log_exp / S_xx_exp if S_xx_exp != 0 else 0.0
            ln_a = Y_log_mean_exp - b_coeff * X_mean_exp

            r2_lin_exp = (S_xy_log_exp**2) / (S_xx_exp * S_yy_log_exp) if (S_xx_exp * S_yy_log_exp) != 0 else 0.0
            a_coeff = np.exp(ln_a)

            y_pred_exp_simple = a_coeff * np.exp(b_coeff * X)
            sse_exp_simple = np.sum((Y - y_pred_exp_simple)**2)
            rmse_exp_simple = np.sqrt(sse_exp_simple / n)
            r2_exp_simple = 1 - (sse_exp_simple / Syy) if Syy != 0 else 0
        except Exception as e:
            fun_exp_valid = False
            fun_exp_error_msg = f"Error en los cálculos de Función Exponencial: {e}"

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

    if exp_asymp_valid:
        df_results["Y_est (No Lin. Exponencial Asintota)"] = y_pred_nl_exp
        df_results["Residuo (No Lin. Exponencial Asintota)"] = Y - y_pred_nl_exp

    if log_valid:
        df_results["Y_est (No Lin. Logaritmico)"] = y_pred_nl_log
        df_results["Residuo (No Lin. Logaritmico)"] = Y - y_pred_nl_log

    if pot_valid:
        df_results["Y_est (No Lin. Potencial / Power Law)"] = y_pred_nl_pot
        df_results["Residuo (No Lin. Potencial / Power Law)"] = Y - y_pred_nl_pot

    if quad_valid:
        df_results["Y_est (No Lin. Cuadratico)"] = y_pred_nl_quad
        df_results["Residuo (No Lin. Cuadratico)"] = Y - y_pred_nl_quad

    if fun_pot_valid:
        df_results["Y_est (Fun. Potencia log10 - Correcto)"] = y_pred_pot10_correct
        df_results["Residuo (Fun. Potencia log10 - Correcto)"] = Y - y_pred_pot10_correct
        df_results["Y_est (Fun. Potencia log10 - Incorrecto)"] = y_pred_pot10_incorrect
        df_results["Residuo (Fun. Potencia log10 - Incorrecto)"] = Y - y_pred_pot10_incorrect

    if fun_exp_valid:
        df_results["Y_est (Fun. Exponencial Simple ln)"] = y_pred_exp_simple
        df_results["Residuo (Fun. Exponencial Simple ln)"] = Y - y_pred_exp_simple




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
            x=x_line, y=m_slr * x_line + c_slr, mode='lines', name='SLR',
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
            "Intercepción (c)": [c_slr, b_gor, b_prop],
            "SE(m)": [se_m_slr, se_m_gor, se_m_prop],
            "SE(c)": [se_c_slr, se_b_gor, se_b_prop],
            "RMSE": [rmse_slr, rmse_gor, rmse_prop],
            "R²": [r2_slr, r2_gor, r2_prop]
        }
        if exp_asymp_valid:
            comp_data["Método"].append("No Lin. Exponencial Asintota")
            comp_data["Pendiente (m)"].append(np.nan)
            comp_data["Intercepción (c)"].append(np.nan)
            comp_data["SE(m)"].append(np.nan)
            comp_data["SE(c)"].append(np.nan)
            comp_data["RMSE"].append(rmse_nl_exp)
            comp_data["R²"].append(r2_nl_exp)

        if log_valid:
            comp_data["Método"].append("No Lin. Logarítmico")
            comp_data["Pendiente (m)"].append(np.nan)
            comp_data["Intercepción (c)"].append(np.nan)
            comp_data["SE(m)"].append(np.nan)
            comp_data["SE(c)"].append(np.nan)
            comp_data["RMSE"].append(rmse_nl_log)
            comp_data["R²"].append(r2_nl_log)

        if pot_valid:
            comp_data["Método"].append("No Lin. Potencial / Power Law")
            comp_data["Pendiente (m)"].append(np.nan)
            comp_data["Intercepción (c)"].append(np.nan)
            comp_data["SE(m)"].append(np.nan)
            comp_data["SE(c)"].append(np.nan)
            comp_data["RMSE"].append(rmse_nl_pot)
            comp_data["R²"].append(r2_nl_pot)

        if quad_valid:
            comp_data["Método"].append("No Lin. Cuadrático")
            comp_data["Pendiente (m)"].append(np.nan)
            comp_data["Intercepción (c)"].append(np.nan)
            comp_data["SE(m)"].append(np.nan)
            comp_data["SE(c)"].append(np.nan)
            comp_data["RMSE"].append(rmse_nl_quad)
            comp_data["R²"].append(r2_nl_quad)

        if fun_pot_valid:
            comp_data["Método"].append("Fun. Potencia log10 (Correcto)")
            comp_data["Pendiente (m)"].append(np.nan)
            comp_data["Intercepción (c)"].append(np.nan)
            comp_data["SE(m)"].append(np.nan)
            comp_data["SE(c)"].append(np.nan)
            comp_data["RMSE"].append(rmse_pot10_correct)
            comp_data["R²"].append(r2_pot10_correct)

            comp_data["Método"].append("Fun. Potencia log10 (Incorrecto)")
            comp_data["Pendiente (m)"].append(np.nan)
            comp_data["Intercepción (c)"].append(np.nan)
            comp_data["SE(m)"].append(np.nan)
            comp_data["SE(c)"].append(np.nan)
            comp_data["RMSE"].append(rmse_pot10_incorrect)
            comp_data["R²"].append(r2_pot10_incorrect)

        if fun_exp_valid:
            comp_data["Método"].append("Fun. Exponencial Simple ln")
            comp_data["Pendiente (m)"].append(np.nan)
            comp_data["Intercepción (c)"].append(np.nan)
            comp_data["SE(m)"].append(np.nan)
            comp_data["SE(c)"].append(np.nan)
            comp_data["RMSE"].append(rmse_exp_simple)
            comp_data["R²"].append(r2_exp_simple)


        df_comp = pd.DataFrame(comp_data)
        st.markdown("**Comparativa de Métricas de los Modelos**")
        st.dataframe(df_comp.style.format({
            "Pendiente (m)": "{:.4f}",
            "Intercepción (c)": "{:.4f}",
            "SE(m)": "{:.4f}",
            "SE(c)": "{:.4f}",
            "RMSE": "{:.4f}",
            "R²": "{:.4f}"
        }, na_rep="-"), use_container_width=True)
        
        diff_pendiente = abs(m_slr - m_prop) / abs(m_slr) * 100 if m_slr != 0 else 0
        st.info(f"**Insight:** La diferencia de pendiente entre SLR y GOR Propuesto es del **{diff_pendiente:.2f}%**.")



    st.markdown("---")

    # --- MÓDULOS EDUCATIVOS ---
    st.header("📖 Módulos Educativos y Fundamentos Matemáticos")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "1. SLR (Mínimos Cuadrados)", 
        "2. GOR Convencional", 
        "3. GOR Propuesto", 
        "4. Regresión No Lineal", 
        "5. Power Law Model",
        "6. Función Potencia",
        "7. Función Exponencial",
        "📥 Datos y Exportación"
    ])



    with tab1:
        st.subheader("Módulo 1: Regresión Lineal por Mínimos Cuadrados Ordinarios (SLR / OLS)")
        
        # Concepto e Idea de la Diapositiva 3
        col_concept, col_example = st.columns([1.2, 1])
        with col_concept:
            st.markdown("""
            **Concepto de Regresión Lineal:**
            La regresión lineal es un método utilizado para modelar la relación entre variables usando una **línea recta**.
            
            **Ecuación del Modelo:**
            """)
            st.latex(r"y = mx + c")
            st.markdown("""
            **Significado de los Parámetros:**
            *   $x$: Variable independiente (independent variable)
            *   $y$: Variable dependiente (dependent variable)
            *   $m$: Pendiente / Tasa de cambio (slope / rate of change)
            *   $c$: Intercepción (intercept)
            """)
        with col_example:
            st.markdown("""
            **Idea Principal:**
            *   Asume que los datos cambian a una **tasa constante** (constant rate).
            *   La relación entre $x$ e $y$ es **lineal (línea recta)**.
            
            **Ejemplo Práctico:**
            Si la temperatura aumenta, las ventas de helados aumentan de forma constante $\Rightarrow$ tendencia de línea recta (straight-line trend).
            
            *Nota Educativa:* Este método clásico asume que la variable independiente ($X$) es perfecta y **no tiene error de medición**, asignando toda la incertidumbre o error al eje dependiente ($Y$), minimizando las distancias verticales (residuos).
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
            st.markdown("**Intersección ($c$):**")
            st.latex(r"c = \bar{Y} - m\bar{X}")
            st.latex(rf"\Rightarrow c = {c_slr:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown("**Error Estándar de la Pendiente ($SE_m$):**")
            st.latex(r"SE_m = \sqrt{\frac{s_e^2}{\sum(X_i - \bar{X})^2}}")
            st.latex(rf"\Rightarrow SE_m = {se_m_slr:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
        st.markdown("**Error Estándar de la Intercepción ($SE_c$):**")
        st.latex(r"SE_c = SE_m \sqrt{\frac{\sum X_i^2}{n}}")
        st.latex(rf"\Rightarrow SE_c = {se_c_slr:.4f}")
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
        
        # Conceptos de las Diapositivas 4 y 5
        st.markdown("### 📖 Fundamentos de Regresión No Lineal")
        col_nl_concept, col_nl_diff = st.columns([1.1, 1.2])
        
        with col_nl_concept:
            st.markdown("""
            **¿Qué es la Regresión No Lineal?**
            Se utiliza cuando la relación entre las variables **no es una línea recta**.
            
            **Modelo de Ejemplo (Exponencial):**
            """)
            st.latex(r"y = a + be^{cx}")
            st.markdown("""
            **Significado e Idea Principal:**
            *   La relación cambia a **diferentes tasas** (tasa de cambio no es constante).
            *   Los datos forman una **curva**, no una recta.
            *   La curva puede tomar forma exponencial, logarítmica, potencial, entre otras.
            
            **Ejemplos de Aplicación:**
            *   Decaimiento de terremotos (sismología).
            *   Crecimiento de poblaciones.
            *   Decaimiento radiactivo, etc.
            """)
            
        with col_nl_diff:
            st.markdown("**Tabla Comparativa (Diferencias):**")
            diff_df = pd.DataFrame({
                "Característica": ["Forma (Shape)", "Tasa de Cambio (Rate of change)", "Ecuación (Equation)", "Complejidad (Complexity)", "Ejemplo (Example)"],
                "Regresión Lineal": ["Línea recta (Straight line)", "Constante (Constant)", "y = mx + c", "Simple", "Salario vs Experiencia"],
                "Regresión No Lineal": ["Curva (Curve)", "Cambiante (Changing)", "y = a + be^(cx), etc.", "Más compleja (More complex)", "Sistemas de decaimiento/crecimiento"]
            })
            st.table(diff_df)
            
        st.markdown("---")

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
                st.markdown("### 📊 Proceso de Linealización y Ajuste del Modelo")
                
                # 1. Tabla de datos
                df_linearized = pd.DataFrame({
                    "x": X,
                    "Y": Y,
                    "y-a": y_minus_a,
                    "ln(y-a)": y_trans,
                    "Y estima": y_pred_nl,
                    "Residuo": Y - y_pred_nl
                })
                st.markdown("**Tabla de Cálculos Intermedios:**")
                st.dataframe(df_linearized.style.format({
                    "x": "{:g}",
                    "Y": "{:.2f}",
                    "y-a": "{:.2f}",
                    "ln(y-a)": "{:.6f}",
                    "Y estima": "{:.6f}",
                    "Residuo": "{:.6f}"
                }), use_container_width=True)
                
                # 2. Sumatorias e Parámetros
                col_math1, col_math2 = st.columns([1.1, 1])
                
                with col_math1:
                    st.markdown("**Sumatorias de Mínimos Cuadrados Linealizados:**")
                    st.markdown(f"""
                    *   $n$ (número de puntos) = **{n}**
                    *   $\sum x$ = **{sum_x:.4f}**
                    *   $\sum x^2$ = **{sum_x2:.4f}**
                    *   $\sum \ln(y-a)$ = **{sum_y_trans:.4f}**
                    *   $\sum x \ln(y-a)$ = **{sum_x_y_trans:.4f}**
                    """)
                    
                    st.markdown("**Fórmulas y Recuperación de Parámetros:**")
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.latex(r"c = \frac{n \sum (x \ln(y-a)) - (\sum x)(\sum \ln(y-a))}{n \sum x^2 - (\sum x)^2}")
                    st.latex(rf"\Rightarrow c \text{{ (pendiente)}} = {c_nl:.6f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.latex(r"A = \frac{\sum \ln(y-a) - c \sum x}{n}")
                    st.latex(rf"\Rightarrow A \text{{ (intersección)}} = {A_nl:.6f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.latex(r"b = e^A")
                    st.latex(rf"\Rightarrow b = e^{{{A_nl:.6f}}} = {b_nl:.6f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col_math2:
                    # Gráfico de Linealización
                    fig_lin = plgo.Figure()
                    
                    # Puntos linealizados
                    fig_lin.add_trace(plgo.Scatter(
                        x=X, y=y_trans, mode='markers', name='Datos Transformados',
                        marker=dict(color='#1f77b4', size=10, symbol='circle'),
                        hovertemplate='x: %{x}<br>ln(y-a): %{y:.6f}<extra></extra>'
                    ))
                    
                    # Recta linealizada
                    x_line = np.linspace(min(X), max(X), 100)
                    y_line_trans = c_nl * x_line + A_nl
                    
                    fig_lin.add_trace(plgo.Scatter(
                        x=x_line, y=y_line_trans, mode='lines', name='Ajuste Lineal',
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate='x: %{x:.2f}<br>ln(y-a) pred: %{y:.6f}<extra></extra>'
                    ))
                    
                    sign_A = "+" if A_nl >= 0 else "-"
                    abs_A = abs(A_nl)
                    fig_lin.update_layout(
                        title=f"Gráfico Linealizado: ln(y-a)<br>y' = {c_nl:.4f}x {sign_A} {abs_A:.4f}  |  R² = {r2_linearized:.4f}",
                        xaxis_title="x",
                        yaxis_title="ln(y-a)",
                        template='plotly_white',
                        showlegend=False,
                        height=350,
                        margin=dict(l=40, r=40, t=60, b=40)
                    )
                    st.plotly_chart(fig_lin, use_container_width=True)
                    
                st.markdown("### 📈 Comparación en el Espacio Original")
                st.markdown(f"**Ecuación Final Ajustada:** $y = {a_nl:.4f} {'+' if b_nl >= 0 else '-'} {abs(b_nl):.4f} \\cdot e^{{{c_nl:.4f} x}}$")



                    
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
                calc_latex = rf"y = {a_nl:.4f} {sign_b} {abs_b:.4f}e^{{{c_nl:.4f} \cdot {x_input:.4f}}} = {y_pred_calc:.4f}"



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
            
            st.markdown("### 📊 Proceso de Linealización y Ajuste del Modelo")
            
            # 1. Tabla de datos
            df_power_linearized = pd.DataFrame({
                "x": X,
                "Y": Y,
                "y-a": [np.nan] * len(X),
                "ln(y-a)": [np.nan] * len(X),
                "x*": x_trans_power,
                "y*": y_trans_power,
                "x* * y*": x_trans_power * y_trans_power,
                "(x*)^2": x_trans_power**2,
                "Y est Power": y_pred_power
            })
            st.markdown("**Tabla de Cálculos Intermedios:**")
            st.dataframe(df_power_linearized.style.format({
                "x": "{:g}",
                "Y": "{:.2f}",
                "y-a": "{:.2f}",
                "ln(y-a)": "{:.6f}",
                "x*": "{:.4f}",
                "y*": "{:.4f}",
                "x* * y*": "{:.4f}",
                "(x*)^2": "{:.4f}",
                "Y est Power": "{:.4f}"
            }, na_rep="-"), use_container_width=True)
            
            # 2. Sumatorias e Parámetros
            col_math1, col_math2 = st.columns([1.1, 1])
            
            with col_math1:
                st.markdown("**Sumatorias de Mínimos Cuadrados Linealizados:**")
                st.markdown(f"""
                *   $n$ (número de puntos) = **{n}**
                *   $\sum x^*$ = **{sum_x_trans_power:.4f}**
                *   $\sum (x^*)^2$ = **{sum_x2_trans_power:.4f}**
                *   $\sum y^*$ = **{sum_y_trans_power:.4f}**
                *   $\sum x^* \cdot y^*$ = **{sum_xy_trans_power:.4f}**
                """)
                
                st.markdown("**Fórmulas y Recuperación de Parámetros:**")
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.latex(r"b = \frac{n \sum (x^* y^*) - (\sum x^*)(\sum y^*)}{n \sum (x^*)^2 - (\sum x^*)^2}")
                st.latex(rf"\Rightarrow b \text{{ (exponente)}} = {b_power:.6f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.latex(r"A = \ln(a) = \frac{\sum y^* - b \sum x^*}{n}")
                st.latex(rf"\Rightarrow A = \ln(a) = {A_power:.6f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.latex(r"a = e^A")
                st.latex(rf"\Rightarrow a = e^{{{A_power:.6f}}} = {a_power:.6f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_math2:
                # Gráfico Y est Power
                fig_power = plgo.Figure()
                
                # Puntos reales
                fig_power.add_trace(plgo.Scatter(
                    x=X, y=Y, mode='markers', name='Datos Observados',
                    marker=dict(color='black', size=10, symbol='circle'),
                    hovertemplate='x: %{x}<br>Y: %{y:.2f}<extra></extra>'
                ))
                
                # Curva de potencia
                x_curve = np.linspace(min(X), max(X), 100)
                y_curve_power = a_power * (x_curve ** b_power)
                
                fig_power.add_trace(plgo.Scatter(
                    x=x_curve, y=y_curve_power, mode='lines', name='Ajuste Power Law',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='x: %{x:.2f}<br>Y estima: %{y:.4f}<extra></extra>'
                ))
                
                fig_power.update_layout(
                    title=f"Gráfico de Datos Exponenciales: Y est Power<br>y = {a_power:.4f} · x^{{{b_power:.4f}}}  |  R² = {r2_power:.4f}",
                    xaxis_title="x",
                    yaxis_title="Y",
                    template='plotly_white',
                    showlegend=False,
                    height=350,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_power, use_container_width=True)
                
            st.markdown("### 📈 Resumen del Modelo")
            st.success(f"**Ecuación Final Ajustada:** $y = {a_power:.6f} \\cdot x^{{{b_power:.6f}}}$")
            
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
                calc_latex = rf"y = {a_power:.4f} \cdot {x_input:.4f}^{{{b_power:.4f}}} = {y_pred_calc:.4f}"
                st.latex(calc_latex)
                st.info(f"**Resultado:** Para $X = {x_input:.4f}$, el valor de $Y$ estimado por el modelo de Ley de Potencia es **{y_pred_calc:.4f}**.")

    with tab6:
        st.subheader("Módulo 6: Función Potencia")
        st.markdown("""
        Este módulo está diseñado específicamente para analizar el modelo de **Función Potencia ($y = a \cdot x^b$)** recreando el comportamiento de tus hojas de cálculo. 
        Utiliza el conjunto de datos de la barra lateral para validar el proceso de linealización logarítmica dual y observar un análisis crítico de los resultados.
        """)
        
        X_mod = df["X"].values
        Y_mod = df["Y"].values

        if not fun_pot_valid:
            st.error(fun_pot_error_msg)
            st.info("Sugerencia: Ajusta tus datos de la barra lateral para cumplir con esta condición de dominio.")
        else:
            r2_lin = r2_lin_pot10
            df_mod_calc = pd.DataFrame({
                "x": X,
                "y": Y,
                "Log Y": Y_log,
                "Log X": X_log
            })

            st.markdown("#### 📊 Tabla de Cálculos Intermedios (Base log10)")
            st.dataframe(df_mod_calc.style.format({
                "x": "{:g}",
                "y": "{:g}",
                "Log Y": "{:.5f}",
                "Log X": "{:.5f}"
            }), use_container_width=True)

            col_coef1, col_coef2 = st.columns(2)
            with col_coef1:
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown("**Regresión Linealizada (Log-Log):**")
                st.latex(rf"\log_{{10}}(y) = {m_log:.4f} \cdot \log_{{10}}(x) + {c_log:.4f}")
                st.latex(rf"R^2 = {r2_lin:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_coef2:
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown("**Parámetros Ajustados:**")
                st.markdown(f"- **Log a** (Intercepción): `{c_log:.4f}`")
                st.markdown(f"- **Log b** (Pendiente): `{m_log:.4f}`")
                st.markdown(f"- **a** ($10^{{Log a}}$): `{a_correct:.4f}`")
                st.markdown(f"- **b matemático (correcto, pendiente $m$):** `{b_correct:.4f}`")
                st.markdown(f"- **b incorrecto ($10^{{m}}$):** `{b_err:.4f}`")
                st.markdown("</div>", unsafe_allow_html=True)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_lin_power = plgo.Figure()
                fig_lin_power.add_trace(plgo.Scatter(
                    x=X_log, y=Y_log, mode='markers', name='Datos Transformados',
                    marker=dict(color='#1f77b4', size=10, symbol='circle'),
                    hovertemplate='Log X: %{x:.5f}<br>Log Y: %{y:.5f}<extra></extra>'
                ))
                x_log_line = np.linspace(min(X_log), max(X_log), 100)
                y_log_line = m_log * x_log_line + c_log
                fig_lin_power.add_trace(plgo.Scatter(
                    x=x_log_line, y=y_log_line, mode='lines', name='Línea de Tendencia',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    hovertemplate='Log X: %{x:.5f}<br>Log Y Pred: %{y:.5f}<extra></extra>'
                ))
                fig_lin_power.update_layout(
                    title=f"Gráfica en Espacio Linealizado<br>y = {m_log:.4f}x + {c_log:.4f} | R² = {r2_lin:.4f}",
                    xaxis_title="Log X",
                    yaxis_title="Log Y",
                    template='plotly_white',
                    showlegend=False,
                    height=380,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_lin_power, use_container_width=True)

            with col_g2:
                fig_orig_power = plgo.Figure()
                fig_orig_power.add_trace(plgo.Scatter(
                    x=X_mod, y=Y_mod, mode='markers', name='Datos Observados',
                    marker=dict(color='black', size=10, symbol='circle'),
                    hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>'
                ))

                x_orig_line = np.linspace(min(X_mod), max(X_mod), 200)
                y_orig_correct = a_correct * (x_orig_line ** b_correct)
                
                fig_orig_power.add_trace(plgo.Scatter(
                    x=x_orig_line, y=y_orig_correct, mode='lines', name='Curva Correcta (Verde)',
                    line=dict(color='green', width=2.5),
                    hovertemplate='X: %{x:.2f}<br>Y (Correcto): %{y:.4f}<extra></extra>'
                ))

                try:
                    y_orig_err = a_err * (x_orig_line ** b_err)
                    if not np.any(np.isinf(y_orig_err)) and np.max(y_orig_err) < 1000000:
                        fig_orig_power.add_trace(plgo.Scatter(
                            x=x_orig_line, y=y_orig_err, mode='lines', name='Curva con Despeje Erróneo (Rojo)',
                            line=dict(color='red', width=2.5, dash='dot'),
                            hovertemplate='X: %{x:.2f}<br>Y (Despeje Erróneo): %{y:.4f}<extra></extra>'
                        ))
                except Exception:
                    pass

                fig_orig_power.update_layout(
                    title="Ajuste en Espacio Original (X vs Y)",
                    xaxis_title="X",
                    yaxis_title="Y",
                    template='plotly_white',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    height=380,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_orig_power, use_container_width=True)

            st.warning("💡 **Análisis Educativo y Crítico del Modelo Potencia:**")
            st.markdown(f"""
            Al linealizar el modelo potencial $y = a x^b$ mediante logaritmos base 10, obtenemos:
            $$\log_{{10}}(y) = \log_{{10}}(a) + b \log_{{10}}(x)$$
            Comparando esto con la ecuación de una recta $Y^* = m X^* + c$, identificamos que:
            - La **intercepción ($c$)** representa $\log_{{10}}(a)$, por lo que recuperamos $a = 10^c$: $\log_{{10}}(a) = {c_log:.4f} \Rightarrow a = 10^{{{c_log:.4f}}} = {a_correct:.4f}$.
            - La **pendiente ($m$)** representa **directamente** el exponente $b$, por lo que $b = m$. En la regresión, la pendiente es **{m_log:.4f}**, por lo que la ecuación matemática correcta es **$y = {a_correct:.4f} x^{{{b_correct:.4f}}}$**.

            **⚠️ El error común de despeje:**
            Un error frecuente al recuperar los parámetros consiste en aplicar la transformación exponencial a la pendiente de manera incorrecta: $b_{{\text{{err}}}} = 10^{{m}} = 10^{{{m_log:.4f}}} = {b_err:.4f}$.
            Esto resulta en la ecuación distorsionada $y = {a_err:.4f} x^{{{b_err:.4f}}}$.
            
            Como puedes ver en la gráfica en el **Espacio Original**, la curva resultante del despeje erróneo (en color rojo punteado) crece de manera explosiva y no se ajusta en absoluto a los datos reales para valores mayores de X, mientras que la curva matemáticamente correcta (en color verde) pasa exactamente por los puntos experimentales. ¡Este es un excelente ejemplo de por qué es importante comprender la teoría de la linealización!
            """)

            st.markdown("### 📖 Formulación Matemática")
            st.markdown(r"""
            Para ajustar el modelo de ley de potencia $y = a x^b$:
            1. **Transformación:** Aplicamos logaritmo en base 10 a ambas variables:
               $$x_i^* = \log_{10}(x_i), \quad y_i^* = \log_{10}(y_i)$$
            2. **Regresión lineal:** Ajustamos la recta $y_i^* = m x_i^* + c$ usando Mínimos Cuadrados Ordinarios:
               $$m = \frac{n \sum (x_i^* y_i^*) - (\sum x_i^*)(\sum y_i^*)}{n \sum (x_i^*)^2 - (\sum x_i^*)^2}$$
               $$c = \bar{y}^* - m \bar{x}^*$$
            3. **Despeje de parámetros:**
               $$a = 10^c, \quad b = m \quad \text{(Despeje Matemático Correcto)}$$
               $$a = 10^c, \quad b = 10^m \quad \text{(Despeje Incorrecto Común)}$$
            """)

    with tab7:
        st.subheader("Módulo 7: Función Exponencial")
        st.markdown("""
        Este módulo está diseñado específicamente para analizar el modelo de **Función Exponencial ($y = a \cdot e^{bx}$)** recreando el comportamiento de tus hojas de cálculo. 
        Utiliza el conjunto de datos de la barra lateral para validar el proceso de linealización semi-logarítmica y observar un análisis crítico de los resultados.
        """)
        
        X_mod = df["X"].values
        Y_mod = df["Y"].values

        if not fun_exp_valid:
            st.error(fun_exp_error_msg)
            st.info("Sugerencia: Ajusta tus datos de la barra lateral para cumplir con esta condición de dominio.")
        else:
            r2_lin = r2_lin_exp
            df_mod_calc = pd.DataFrame({
                "x": X,
                "y": Y,
                "Ln Y": Y_log_exp
            })

            st.markdown("#### 📊 Tabla de Cálculos Intermedios (Base logaritmo natural)")
            st.dataframe(df_mod_calc.style.format({
                "x": "{:g}",
                "y": "{:g}",
                "Ln Y": "{:.8f}"
            }), use_container_width=True)

            col_coef1, col_coef2 = st.columns(2)
            with col_coef1:
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown("**Regresión Linealizada (Semi-Log):**")
                st.latex(rf"\ln(y) = {b_coeff:.4f} \cdot x + {ln_a:.4f}")
                st.latex(rf"R^2 = {r2_lin:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_coef2:
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown("**Parámetros Ajustados:**")
                st.markdown(f"- **Ln(a)** (Intercepción): `{ln_a:.4f}`")
                st.markdown(f"- **b** (Pendiente / Exponente): `{b_coeff:.4f}`")
                st.markdown(f"- **A** (Coeficiente $a = e^{{Ln(a)}}$): `{a_coeff:.4f}`")
                st.markdown(f"**Ecuación final:** $y = {a_coeff:.4f} \\cdot e^{{{b_coeff:.4f} x}}$")
                st.markdown("</div>", unsafe_allow_html=True)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_lin_exp = plgo.Figure()
                fig_lin_exp.add_trace(plgo.Scatter(
                    x=X_mod, y=Y_log, mode='markers', name='Datos Transformados',
                    marker=dict(color='#1f77b4', size=10, symbol='circle'),
                    hovertemplate='X: %{x}<br>ln(Y): %{y:.5f}<extra></extra>'
                ))
                x_line = np.linspace(min(X_mod), max(X_mod), 100)
                y_line = b_coeff * x_line + ln_a
                fig_lin_exp.add_trace(plgo.Scatter(
                    x=x_line, y=y_line, mode='lines', name='Línea de Tendencia',
                    line=dict(color='#ff7f0e', width=2),
                    hovertemplate='X: %{x:.2f}<br>ln(Y) Pred: %{y:.5f}<extra></extra>'
                ))
                fig_lin_exp.update_layout(
                    title=f"Gráfica en Espacio Linealizado<br>y = {b_coeff:.4f}x + {ln_a:.4f} | R² = {r2_lin:.4f}",
                    xaxis_title="x",
                    yaxis_title="Ln Y",
                    template='plotly_white',
                    showlegend=False,
                    height=380,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_lin_exp, use_container_width=True)

            with col_g2:
                fig_orig_exp = plgo.Figure()
                fig_orig_exp.add_trace(plgo.Scatter(
                    x=X_mod, y=Y_mod, mode='markers', name='Datos Observados',
                    marker=dict(color='black', size=10, symbol='circle'),
                    hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>'
                ))

                x_orig_line = np.linspace(min(X_mod), max(X_mod), 200)
                y_orig_fit = a_coeff * np.exp(b_coeff * x_orig_line)

                fig_orig_exp.add_trace(plgo.Scatter(
                    x=x_orig_line, y=y_orig_fit, mode='lines', name='Curva Exponencial',
                    line=dict(color='green', width=2.5),
                    hovertemplate='X: %{x:.2f}<br>Y estima: %{y:.4f}<extra></extra>'
                ))

                fig_orig_exp.update_layout(
                    title="Ajuste en Espacio Original (X vs Y)",
                    xaxis_title="X",
                    yaxis_title="Y",
                    template='plotly_white',
                    showlegend=False,
                    height=380,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_orig_exp, use_container_width=True)

            st.info("💡 **Análisis Educativo y Crítico del Modelo Exponencial:**")
            st.markdown(f"""
            Al linealizar el modelo exponencial $y = a e^{{bx}}$ usando el logaritmo natural, obtenemos:
            $$\ln(y) = \ln(a) + b x$$
            Comparando esto con la ecuación de una recta $Y^* = b X + A$, observamos que:
            - La **intercepción ($A$)** en el eje vertical es $\ln(a)$, de donde recuperamos el término $a = e^A$: $\ln(a) = {ln_a:.4f} \Rightarrow a = e^{{{ln_a:.4f}}} = {a_coeff:.4f}$.
            - La **pendiente ($b$)** representa **directamente** el exponente en el factor de escala $e^{{bx}}$, que es **{b_coeff:.4f}**.
            
            Este cálculo resulta en la ecuación final ajustada:
            $$y = {a_coeff:.4f} \cdot e^{{{b_coeff:.4f} x}}$$
            La curva verde ajustada en el espacio original representa con precisión la tendencia de los datos experimentales sin experimentar desviaciones.
            """)

            st.markdown("### 📖 Formulación Matemática")
            st.markdown(r"""
            Para ajustar el modelo exponencial $y = a e^{bx}$:
            1. **Transformación:** Aplicamos logaritmo natural únicamente a la variable dependiente $y$:
               $$y_i^* = \ln(y_i)$$
            2. **Regresión lineal:** Ajustamos la recta $y_i^* = b x_i + \ln(a)$ usando Mínimos Cuadrados Ordinarios:
               $$b = \frac{n \sum (x_i y_i^*) - (\sum x_i)(\sum y_i^*)}{n \sum x_i^2 - (\sum x_i)^2}$$
               $$\ln(a) = \bar{y}^* - b \bar{x}$$
            3. **Despeje de parámetros:**
               $$a = e^{\ln(a)}, \quad b = \text{pendiente}$$
            """)

    with tab8:
        st.subheader("Datos Originales y Proyecciones")
        st.markdown("Tabla detallada con cada dato original, sus predicciones y los residuos según cada modelo.")
        st.dataframe(df_results, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Guía de Fórmulas Dinámicas en Excel")
        st.markdown("""
        El reporte Excel descargable contiene fórmulas dinámicas que calculan automáticamente las estimaciones y los residuos. Esto permite cambiar los valores de entrada en Excel y actualizar el análisis en tiempo real. 
        
        A continuación se muestran las fórmulas aplicadas en la primera fila de datos (Fila 6 en Excel, donde la columna **B** es **X** y la columna **C** es **Y**):
        """)
        
        # Build guide table matching models_details
        models_details_guide = []
        models_details_guide.append({
            "name": "SLR",
            "pred": lambda r: f"= {m_slr:.6f} * B{r} + {c_slr:.6f}",
        })
        models_details_guide.append({
            "name": "GOR Conv",
            "pred": lambda r: f"= {m_gor:.6f} * B{r} + {b_gor:.6f}",
        })
        models_details_guide.append({
            "name": "GOR Prop",
            "pred": lambda r: f"= {m_prop:.6f} * B{r} + {b_prop:.6f}",
        })
        if exp_asymp_valid:
            models_details_guide.append({
                "name": "Exp Asíntota",
                "pred": lambda r: f"= {a_val:.6f} + {b_nl_exp:.6f} * EXP({c_nl_exp:.6f} * B{r})",
            })
        if log_valid:
            models_details_guide.append({
                "name": "Logarítmico",
                "pred": lambda r: f"= {a_nl_log:.6f} + {b_nl_log:.6f} * LN(B{r})",
            })
        if pot_valid:
            models_details_guide.append({
                "name": "Potencial / Power Law",
                "pred": lambda r: f"= {a_nl_pot:.6f} * (B{r} ^ {b_nl_pot:.6f})",
            })
        if quad_valid:
            models_details_guide.append({
                "name": "Cuadrático",
                "pred": lambda r: f"= {a_nl_quad:.6f} * (B{r} ^ 2) + {b_nl_quad:.6f} * B{r} + {c_nl_quad:.6f}",
            })
        if fun_pot_valid:
            models_details_guide.append({
                "name": "Fun Potencia (Correcto)",
                "pred": lambda r: f"= {a_correct:.6f} * (B{r} ^ {b_correct:.6f})",
            })
            models_details_guide.append({
                "name": "Fun Potencia (Incorrecto)",
                "pred": lambda r: f"= {a_err:.6f} * (B{r} ^ {b_err:.6f})",
            })
        if fun_exp_valid:
            models_details_guide.append({
                "name": "Fun Exponencial",
                "pred": lambda r: f"= {a_coeff:.6f} * EXP({b_coeff:.6f} * B{r})",
            })

        from openpyxl.utils import get_column_letter

        col_letters = {}
        c_idx = 4 # column 4 is D
        for m in models_details_guide:
            pred_letter = get_column_letter(c_idx)
            c_idx += 1
            res_letter = get_column_letter(c_idx)
            c_idx += 1
            col_letters[m["name"]] = (pred_letter, res_letter)

        guide_data = {
            "Modelo / Método": [],
            "Fórmula de Estimación (Columna en Excel)": [],
            "Fórmula de Residuo (Columna en Excel)": []
        }

        for m in models_details_guide:
            pred_col, res_col = col_letters[m["name"]]
            guide_data["Modelo / Método"].append(m["name"])
            guide_data["Fórmula de Estimación (Columna en Excel)"].append(f"`{m['pred'](6)}` (Columna {pred_col})")
            guide_data["Fórmula de Residuo (Columna en Excel)"].append(f"`= C6 - {pred_col}6` (Columna {res_col})")

        df_guide = pd.DataFrame(guide_data)
        st.table(df_guide)

        st.markdown("---")
        st.subheader("Exportar Resultados")
        col_out1, col_out2 = st.columns(2)
        
        with col_out1:
            st.markdown("**Descargar Excel:**")
            
            def generate_excel_with_formulas():
                import openpyxl
                from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
                from openpyxl.utils import get_column_letter

                wb = openpyxl.Workbook()
                default_sheet = wb.active
                wb.remove(default_sheet)

                # Styles
                title_font = Font(name="Calibri", size=16, bold=True, color="1F497D")
                subtitle_font = Font(name="Calibri", size=11, italic=True, color="595959")
                header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
                bold_font = Font(name="Calibri", size=11, bold=True)
                regular_font = Font(name="Calibri", size=11)
                
                header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
                thin_border_side = Side(border_style="thin", color="D3D3D3")
                thin_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
                
                align_center = Alignment(horizontal="center", vertical="center")
                align_left = Alignment(horizontal="left", vertical="center")
                align_right = Alignment(horizontal="right", vertical="center")

                # --- SHEET 1: Resumen de Modelos ---
                ws_summary = wb.create_sheet(title="Resumen de Modelos")
                ws_summary.views.sheetView[0].showGridLines = True
                
                ws_summary.cell(row=2, column=2, value="Reporte Comparativo de Regresiones").font = title_font
                ws_summary.cell(row=3, column=2, value="Análisis estadístico y fórmulas dinámicas para Excel").font = subtitle_font
                
                headers_summary = ["Método", "Ecuación Matemática", "Fórmula Excel Sugerida", "RMSE", "R²", "Estado"]
                for col_idx, header in enumerate(headers_summary, start=2):
                    cell = ws_summary.cell(row=5, column=col_idx, value=header)
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = align_center
                    cell.border = thin_border
                
                methods_list = [
                    {"name": "Regresión Lineal Simple (SLR)", "math": "y = m*x + c", "formula": "= m * X + c", "valid": True, "rmse": rmse_slr, "r2": r2_slr},
                    {"name": "GOR Convencional", "math": "y = m*x + c", "formula": "= m * X + c", "valid": True, "rmse": rmse_gor, "r2": r2_gor},
                    {"name": "GOR Propuesto", "math": "y = m*x + c", "formula": "= m * X + c", "valid": True, "rmse": rmse_prop, "r2": r2_prop},
                    {"name": "No Lin. Exponencial Asíntota", "math": "y = a + b*e^(cx)", "formula": "= a + b * EXP(c * X)", "valid": exp_asymp_valid, "rmse": rmse_nl_exp if exp_asymp_valid else np.nan, "r2": r2_nl_exp if exp_asymp_valid else np.nan},
                    {"name": "No Lin. Logarítmico", "math": "y = a + b*ln(x)", "formula": "= a + b * LN(X)", "valid": log_valid, "rmse": rmse_nl_log if log_valid else np.nan, "r2": r2_nl_log if log_valid else np.nan},
                    {"name": "No Lin. Potencial / Power Law", "math": "y = a*x^b", "formula": "= a * (X ^ b)", "valid": pot_valid, "rmse": rmse_nl_pot if pot_valid else np.nan, "r2": r2_nl_pot if pot_valid else np.nan},
                    {"name": "No Lin. Cuadrático", "math": "y = a*x^2 + b*x + c", "formula": "= a * (X ^ 2) + b * X + c", "valid": quad_valid, "rmse": rmse_nl_quad if quad_valid else np.nan, "r2": r2_nl_quad if quad_valid else np.nan},
                    {"name": "Función Potencia log10 (Correcto)", "math": "y = a*x^b", "formula": "= a * (X ^ b)", "valid": fun_pot_valid, "rmse": rmse_pot10_correct if fun_pot_valid else np.nan, "r2": r2_pot10_correct if fun_pot_valid else np.nan},
                    {"name": "Función Potencia log10 (Incorrecto)", "math": "y = a_err * x^b_err", "formula": "= a_err * (X ^ b_err)", "valid": fun_pot_valid, "rmse": rmse_pot10_incorrect if fun_pot_valid else np.nan, "r2": r2_pot10_incorrect if fun_pot_valid else np.nan},
                    {"name": "Función Exponencial Simple ln", "math": "y = a*e^(bx)", "formula": "= a * EXP(b * X)", "valid": fun_exp_valid, "rmse": rmse_exp_simple if fun_exp_valid else np.nan, "r2": r2_exp_simple if fun_exp_valid else np.nan},
                ]
                
                current_row = 6
                for m in methods_list:
                    ws_summary.cell(row=current_row, column=2, value=m["name"]).font = regular_font
                    ws_summary.cell(row=current_row, column=3, value=m["math"]).font = regular_font
                    ws_summary.cell(row=current_row, column=4, value=m["formula"]).font = regular_font
                    
                    if m["valid"]:
                        ws_summary.cell(row=current_row, column=5, value=m["rmse"]).number_format = "0.0000"
                        ws_summary.cell(row=current_row, column=6, value=m["r2"]).number_format = "0.0000"
                        ws_summary.cell(row=current_row, column=7, value="Ajustado").font = regular_font
                    else:
                        ws_summary.cell(row=current_row, column=5, value="-").alignment = align_center
                        ws_summary.cell(row=current_row, column=6, value="-").alignment = align_center
                        ws_summary.cell(row=current_row, column=7, value="No Válido (Dominio)").font = Font(name="Calibri", size=11, color="FF0000")
                        
                    for c in range(2, 8):
                        cell = ws_summary.cell(row=current_row, column=c)
                        cell.border = thin_border
                        if c in [5, 6]:
                            cell.alignment = align_right
                        elif c in [2, 3, 4]:
                            cell.alignment = align_left
                        else:
                            cell.alignment = align_center
                    current_row += 1
                    
                # Autosize columns
                for col in ws_summary.columns:
                    max_len = max(len(str(cell.value or '')) for cell in col)
                    col_letter = get_column_letter(col[0].column)
                    if col[0].column >= 2:
                        ws_summary.column_dimensions[col_letter].width = max(max_len + 3, 10)
                
                # --- SHEET 2: Resultados Detallados ---
                ws_details = wb.create_sheet(title="Resultados Detallados")
                ws_details.views.sheetView[0].showGridLines = True
                
                ws_details.cell(row=2, column=2, value="Resultados del Ajuste y Predicciones Dinámicas").font = title_font
                ws_details.cell(row=3, column=2, value="Las estimaciones y residuos se calculan automáticamente mediante fórmulas de Excel").font = subtitle_font
                
                models_details = []
                models_details.append({
                    "name": "SLR",
                    "pred": lambda r: f"= {m_slr:.12f} * B{r} + {c_slr:.12f}",
                })
                models_details.append({
                    "name": "GOR Conv",
                    "pred": lambda r: f"= {m_gor:.12f} * B{r} + {b_gor:.12f}",
                })
                models_details.append({
                    "name": "GOR Prop",
                    "pred": lambda r: f"= {m_prop:.12f} * B{r} + {b_prop:.12f}",
                })
                if exp_asymp_valid:
                    models_details.append({
                        "name": "Exp Asíntota",
                        "pred": lambda r: f"= {a_val:.12f} + {b_nl_exp:.12f} * EXP({c_nl_exp:.12f} * B{r})",
                    })
                if log_valid:
                    models_details.append({
                        "name": "Logarítmico",
                        "pred": lambda r: f"= {a_nl_log:.12f} + {b_nl_log:.12f} * LN(B{r})",
                    })
                if pot_valid:
                    models_details.append({
                        "name": "Potencial / Power Law",
                        "pred": lambda r: f"= {a_nl_pot:.12f} * (B{r} ^ {b_nl_pot:.12f})",
                    })
                if quad_valid:
                    models_details.append({
                        "name": "Cuadrático",
                        "pred": lambda r: f"= {a_nl_quad:.12f} * (B{r} ^ 2) + {b_nl_quad:.12f} * B{r} + {c_nl_quad:.12f}",
                    })
                if fun_pot_valid:
                    models_details.append({
                        "name": "Fun Potencia (Correcto)",
                        "pred": lambda r: f"= {a_correct:.12f} * (B{r} ^ {b_correct:.12f})",
                    })
                    models_details.append({
                        "name": "Fun Potencia (Incorrecto)",
                        "pred": lambda r: f"= {a_err:.12f} * (B{r} ^ {b_err:.12f})",
                    })
                if fun_exp_valid:
                    models_details.append({
                        "name": "Fun Exponencial",
                        "pred": lambda r: f"= {a_coeff:.12f} * EXP({b_coeff:.12f} * B{r})",
                    })
                
                # Write Headers
                ws_details.cell(row=5, column=2, value="X (Obs)").font = header_font
                ws_details.cell(row=5, column=2).fill = header_fill
                ws_details.cell(row=5, column=2).border = thin_border
                ws_details.cell(row=5, column=2).alignment = align_center
                
                ws_details.cell(row=5, column=3, value="Y (Obs)").font = header_font
                ws_details.cell(row=5, column=3).fill = header_fill
                ws_details.cell(row=5, column=3).border = thin_border
                ws_details.cell(row=5, column=3).alignment = align_center
                
                col_idx = 4
                for m in models_details:
                    cell_est = ws_details.cell(row=5, column=col_idx, value=f"Y_est ({m['name']})")
                    cell_est.font = header_font
                    cell_est.fill = header_fill
                    cell_est.border = thin_border
                    cell_est.alignment = align_center
                    col_idx += 1
                    
                    cell_res = ws_details.cell(row=5, column=col_idx, value=f"Residuo ({m['name']})")
                    cell_res.font = header_font
                    cell_res.fill = header_fill
                    cell_res.border = thin_border
                    cell_res.alignment = align_center
                    col_idx += 1

                # Write Data
                start_row = 6
                for i, (x_val, y_val) in enumerate(zip(X, Y)):
                    r = start_row + i
                    ws_details.cell(row=r, column=2, value=float(x_val)).number_format = "0.00"
                    ws_details.cell(row=r, column=3, value=float(y_val)).number_format = "0.00"
                    ws_details.cell(row=r, column=2).border = thin_border
                    ws_details.cell(row=r, column=3).border = thin_border
                    ws_details.cell(row=r, column=2).alignment = align_right
                    ws_details.cell(row=r, column=3).alignment = align_right
                    
                    c_idx = 4
                    for m in models_details:
                        pred_formula = m["pred"](r)
                        pred_letter = get_column_letter(c_idx)
                        
                        cell_pred = ws_details.cell(row=r, column=c_idx, value=pred_formula)
                        cell_pred.number_format = "0.0000"
                        cell_pred.border = thin_border
                        cell_pred.alignment = align_right
                        c_idx += 1
                        
                        res_formula = f"= C{r} - {pred_letter}{r}"
                        cell_res = ws_details.cell(row=r, column=c_idx, value=res_formula)
                        cell_res.number_format = "0.0000"
                        cell_res.border = thin_border
                        cell_res.alignment = align_right
                        c_idx += 1
                        
                for col in ws_details.columns:
                    max_len = max(len(str(cell.value or '')) for cell in col)
                    col_letter = get_column_letter(col[0].column)
                    if col[0].column >= 2:
                        ws_details.column_dimensions[col_letter].width = max(max_len + 3, 12)
                        
                # --- SHEET 3: Parámetros de Ajuste ---
                ws_coefs = wb.create_sheet(title="Parametros de Ajuste")
                ws_coefs.views.sheetView[0].showGridLines = True
                
                ws_coefs.cell(row=2, column=2, value="Coeficientes Calculados").font = title_font
                ws_coefs.cell(row=3, column=2, value="Valores de los parámetros y métricas para cada modelo ajustado").font = subtitle_font
                
                headers_coefs = ["Método", "Parámetro 1", "Valor 1", "Parámetro 2", "Valor 2", "Parámetro 3", "Valor 3"]
                for col_idx, header in enumerate(headers_coefs, start=2):
                    cell = ws_coefs.cell(row=5, column=col_idx, value=header)
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = align_center
                    cell.border = thin_border
                    
                coefs_data = []
                coefs_data.append(["SLR", "Pendiente (m)", m_slr, "Intersección (c)", c_slr, "", ""])
                coefs_data.append(["GOR Convencional", "Pendiente (m)", m_gor, "Intersección (c)", b_gor, "", ""])
                coefs_data.append(["GOR Propuesto", "Pendiente (m)", m_prop, "Intersección (c)", b_prop, "", ""])
                if exp_asymp_valid:
                    coefs_data.append(["No Lin. Exponencial Asíntota", "Asíntota (a)", a_val, "b (escala)", b_nl_exp, "c (tasa)", c_nl_exp])
                if log_valid:
                    coefs_data.append(["No Lin. Logarítmico", "a (intersección)", a_nl_log, "b (escala)", b_nl_log, "", ""])
                if pot_valid:
                    coefs_data.append(["No Lin. Potencial / Power Law", "a (escala)", a_nl_pot, "b (exponente)", b_nl_pot, "", ""])
                if quad_valid:
                    coefs_data.append(["No Lin. Cuadrático", "a (x^2)", a_nl_quad, "b (x)", b_nl_quad, "c (cte)", c_nl_quad])
                if fun_pot_valid:
                    coefs_data.append(["Función Potencia log10 (Correcto)", "a (escala)", a_correct, "b (exponente)", b_correct, "", ""])
                    coefs_data.append(["Función Potencia log10 (Incorrecto)", "a_err (escala)", a_err, "b_err (exponente)", b_err, "", ""])
                if fun_exp_valid:
                    coefs_data.append(["Función Exponencial Simple ln", "a (escala)", a_coeff, "b (exponente)", b_coeff, "", ""])

                current_row = 6
                for r_data in coefs_data:
                    ws_coefs.cell(row=current_row, column=2, value=r_data[0]).font = bold_font
                    
                    ws_coefs.cell(row=current_row, column=3, value=r_data[1]).font = regular_font
                    if isinstance(r_data[2], (int, float)):
                        ws_coefs.cell(row=current_row, column=4, value=r_data[2]).number_format = "0.000000"
                    else:
                        ws_coefs.cell(row=current_row, column=4, value=r_data[2])
                        
                    ws_coefs.cell(row=current_row, column=5, value=r_data[3]).font = regular_font
                    if isinstance(r_data[4], (int, float)):
                        ws_coefs.cell(row=current_row, column=6, value=r_data[4]).number_format = "0.000000"
                    else:
                        ws_coefs.cell(row=current_row, column=6, value=r_data[4])
                        
                    ws_coefs.cell(row=current_row, column=7, value=r_data[5]).font = regular_font
                    if isinstance(r_data[6], (int, float)):
                        ws_coefs.cell(row=current_row, column=8, value=r_data[6]).number_format = "0.000000"
                    else:
                        ws_coefs.cell(row=current_row, column=8, value=r_data[6])
                        
                    for col_idx in range(2, 9):
                        cell = ws_coefs.cell(row=current_row, column=col_idx)
                        cell.border = thin_border
                        if col_idx in [2, 3, 5, 7]:
                            cell.alignment = align_left
                        elif col_idx in [4, 6, 8]:
                            cell.alignment = align_right
                    current_row += 1

                for col in ws_coefs.columns:
                    max_len = max(len(str(cell.value or '')) for cell in col)
                    col_letter = get_column_letter(col[0].column)
                    if col[0].column >= 2:
                        ws_coefs.column_dimensions[col_letter].width = max(max_len + 3, 12)

                output = io.BytesIO()
                wb.save(output)
                return output.getvalue()

            try:
                excel_data = generate_excel_with_formulas()
                st.download_button(
                    label="📊 Descargar Reporte en Excel",
                    data=excel_data,
                    file_name="Reporte_Comparacion_Regresiones.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error generando Excel: {e}")
            
        with col_out2:
            st.markdown("**Descargar PDF:**")
            
            def create_pdf():
                # Crear imagen de Matplotlib estática en background para el PDF
                fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5))
                ax_pdf.scatter(X, Y, color='black', label='Datos Observados')
                x_vals = np.linspace(min(X), max(X), 100)
                ax_pdf.plot(x_vals, m_slr * x_vals + c_slr, color='blue', label='SLR')
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
                    method_name = row['Método']
                    if method_name == "SLR":
                        eq = f"y = {m_slr:.4f}x + {c_slr:.4f}"
                    elif method_name.startswith("GOR Conv"):
                        eq = f"y = {m_gor:.4f}x + {b_gor:.4f}"
                    elif method_name.startswith("GOR Prop"):
                        eq = f"y = {m_prop:.4f}x + {b_prop:.4f}"
                    elif method_name == "No Lin. Exponencial Asintota":
                        eq = nl_equation_exp
                    elif method_name == "No Lin. Logarítmico":
                        eq = nl_equation_log
                    elif method_name == "No Lin. Potencial / Power Law":
                        eq = nl_equation_pot
                    elif method_name == "No Lin. Cuadrático":
                        eq = nl_equation_quad
                    elif method_name == "Fun. Potencia log10 (Correcto)":
                        eq = f"y = {a_correct:.4f} * x^{{{b_correct:.4f}}}"
                    elif method_name == "Fun. Potencia log10 (Incorrecto)":
                        eq = f"y = {a_err:.4f} * x^{{{b_err:.4f}}}"
                    elif method_name == "Fun. Exponencial Simple ln":
                        eq = f"y = {a_coeff:.4f} * e^{{{b_coeff:.4f}x}}"
                    else:
                        eq = ""
                    clean_eq = eq.replace('\\cdot', '*').replace('\\ln', 'ln').replace('\\bar', '').replace('\\hat', '').replace('{', '').replace('}', '')
                    pdf.cell(0, 6, f"{method_name}: {clean_eq} | RMSE: {row['RMSE']:.4f} | R2: {row['R²']:.4f}", 0, 1)
                
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
