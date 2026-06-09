import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats
import plotly.graph_objects as plgo
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
from translations import TRANSLATIONS

# --- IDIOMA / LANGUAGE SELECTOR ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'es'

t = TRANSLATIONS[st.session_state.lang]

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title=t["page_title"], page_icon="📈", layout="wide")

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

st.title(t["st_title"])
st.markdown(t["st_desc"])

# --- BARRA LATERAL: ENTRADA DE DATOS ---
st.sidebar.markdown("### 🌐 Idioma / Language")
col_lang1, col_lang2, col_lang3 = st.sidebar.columns(3)
with col_lang1:
    if st.button("Español", use_container_width=True, type="primary" if st.session_state.lang == 'es' else "secondary"):
        st.session_state.lang = 'es'
        st.rerun()
with col_lang2:
    if st.button("English", use_container_width=True, type="primary" if st.session_state.lang == 'en' else "secondary"):
        st.session_state.lang = 'en'
        st.rerun()
with col_lang3:
    if st.button("हिन्दी", use_container_width=True, type="primary" if st.session_state.lang == 'hi' else "secondary"):
        st.session_state.lang = 'hi'
        st.rerun()

st.sidebar.markdown("---")
analysis_type = st.sidebar.selectbox(t["sb_analysis_type"], [t["sb_simple_regression"], t["sb_multiple_regression"]])
is_mlr = (analysis_type == t["sb_multiple_regression"])

st.sidebar.header(t["sb_header_data"])
data_source = st.sidebar.radio(t["sb_data_source"], (t["sb_manual"], t["sb_upload"]))

df = None

if not is_mlr:
    if data_source == t["sb_manual"]:
        st.sidebar.markdown(t["sb_manual_desc"])
        default_data = pd.DataFrame({
            "X": [4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2],
            "Y": [4.7, 4.6, 5.1, 5.2, 5.0, 5.4, 5.3, 5.5, 5.4, 5.6, 5.8, 5.9, 5.7, 6.1, 6.0, 6.2, 6.4, 6.3, 6.6]
        })
        df = st.sidebar.data_editor(default_data, num_rows="dynamic", use_container_width=True)
    else:
        uploaded_file = st.sidebar.file_uploader(t["sb_upload_desc"], type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    df_raw = pd.read_excel(uploaded_file)
                
                st.sidebar.markdown(t["sb_select_cols"])
                x_col = st.sidebar.selectbox(t["sb_col_x"], df_raw.columns)
                y_col = st.sidebar.selectbox(t["sb_col_y"], df_raw.columns)
                
                df = df_raw[[x_col, y_col]].rename(columns={x_col: "X", y_col: "Y"}).dropna()
                
            except Exception as e:
                st.sidebar.error(t["sb_err_file"].format(e))
else:
    if data_source == t["sb_manual"]:
        st.sidebar.markdown(t["sb_manual_desc"])
        default_data_mlr = pd.DataFrame({
            "logPOA": [1.91991, 1.91472, 1.56733, 1.93615, 1.84532, 1.46172, 1.73993, 1.34949, 1.23044, 1.73635, 1.58530, 1.85273, 1.68352, 2.03520, 1.60728],
            "M": [4.78507, 4.78507, 4.78507, 4.78507, 4.78507, 4.78507, 4.78507, 5.81963, 5.81963, 5.74574, 5.74574, 5.74574, 5.74574, 5.74574, 5.74574],
            "M^2": [22.8969, 22.8969, 22.8969, 22.8969, 22.8969, 22.8969, 22.8969, 33.8681, 33.8681, 33.0135, 33.0135, 33.0135, 33.0135, 33.0135, 33.0135],
            "logR": [1.79358, 1.81018, 1.83163, 1.85999, 1.90601, 1.91687, 1.92090, 2.24137, 2.40310, 2.07246, 1.93409, 2.07355, 2.10917, 2.07282, 2.17219],
            "R": [62.1692, 64.5929, 67.8788, 72.4423, 80.5395, 82.5788, 83.3484, 174.328, 252.991, 118.156, 85.9195, 118.454, 128.578, 118.255, 148.659]
        })
        df = st.sidebar.data_editor(default_data_mlr, num_rows="dynamic", use_container_width=True)
    else:
        uploaded_file = st.sidebar.file_uploader(t["sb_upload_desc"], type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.sidebar.error(t["sb_err_file"].format(e))

if df is not None and len(df) >= 2:
    if is_mlr:
        st.sidebar.markdown("---")
        st.sidebar.header(t["sb_select_cols"])
        y_col = st.sidebar.selectbox(t["sb_col_y"], df.columns, index=0)
        x_cols = st.sidebar.multiselect(t["sb_col_x_multiple"], [c for c in df.columns if c != y_col], default=[c for c in df.columns if c != y_col])
        
        if not x_cols:
            st.warning(t["mlr_err_no_x"])
            st.markdown("---")
            st.markdown(
                "<p style='text-align: center; color: gray; font-size: 14px;'>"
                "Desarrollado y mantenido por <b>Alexander Acosta</b> "
                "(<a href='https://github.com/j-alexander-acosta' target='_blank' style='color: #1f77b4; text-decoration: none;'>@j-alexander-acosta</a>)"
                "</p>", 
                unsafe_allow_html=True
            )
            st.stop()
            
        df_clean = df[[y_col] + x_cols].dropna()
        if len(df_clean) < len(x_cols) + 2:
            st.warning(t["mlr_err_insufficient_data"].format(len(x_cols) + 2, len(x_cols)))
            st.markdown("---")
            st.markdown(
                "<p style='text-align: center; color: gray; font-size: 14px;'>"
                "Desarrollado y mantenido por <b>Alexander Acosta</b> "
                "(<a href='https://github.com/j-alexander-acosta' target='_blank' style='color: #1f77b4; text-decoration: none;'>@j-alexander-acosta</a>)"
                "</p>", 
                unsafe_allow_html=True
            )
            st.stop()
            
        # Math Solver
        Y = df_clean[y_col].values
        X_vals = df_clean[x_cols].values
        N = len(Y)
        K = len(x_cols)
        
        X_mat = np.column_stack([np.ones(N), X_vals])
        
        try:
            XtX = np.dot(X_mat.T, X_mat)
            XtX_inv = np.linalg.inv(XtX)
            XtY = np.dot(X_mat.T, Y)
            beta = np.dot(XtX_inv, XtY)
            
            Y_pred = np.dot(X_mat, beta)
            residuals = Y - Y_pred
            
            Y_mean = np.mean(Y)
            SST = np.sum((Y - Y_mean)**2)
            SSE = np.sum(residuals**2)
            SSR = SST - SSE
            
            df_reg = K
            df_resid = N - K - 1
            df_total = N - 1
            
            MSR = SSR / df_reg if df_reg > 0 else 0
            MSE = SSE / df_resid if df_resid > 0 else 0
            
            F_stat = MSR / MSE if MSE > 0 else 0
            signif_f = stats.f.sf(F_stat, df_reg, df_resid) if df_resid > 0 else 1.0
            
            r2 = 1.0 - (SSE / SST) if SST > 0 else 0.0
            r2_adj = 1.0 - ((SSE / df_resid) / (SST / df_total)) if df_resid > 0 and df_total > 0 else 0.0
            se_reg = np.sqrt(MSE)
            
            cov_beta = MSE * XtX_inv
            se_beta = np.sqrt(np.diagonal(cov_beta))
            t_stats = beta / se_beta
            p_values = 2.0 * stats.t.sf(np.abs(t_stats), df_resid) if df_resid > 0 else np.ones(K+1)
            
            t_crit = stats.t.ppf(0.975, df_resid) if df_resid > 0 else 0.0
            ci_lower = beta - t_crit * se_beta
            ci_upper = beta + t_crit * se_beta
            
            multiple_r = np.sqrt(r2) if r2 >= 0 else 0.0
            Y_sorted = np.sort(Y)
            percentiles = (np.arange(1, N + 1) - 0.5) / N * 100
        except np.linalg.LinAlgError:
            st.error("Error: Colinealidad detectada en las variables independientes. Por favor, remueve alguna variable correlacionada.")
            st.stop()
            
        # UI
        st.header(t["mlr_title"])
        tab_stats, tab_formulas, tab_res_analysis, tab_prob_output, tab_normal, tab_plots = st.tabs([
            t["mlr_tab_stats"],
            t["mlr_tab_formulas"],
            t["mlr_tab_res_analysis"],
            t["mlr_tab_prob_output"],
            t["mlr_tab_normal"],
            t["mlr_tab_plots"]
        ])
        
        with tab_stats:
            st.subheader(t["mlr_stats_title"])
            stats_df = pd.DataFrame({
                t["table_col_method"]: [
                    t["mlr_multiple_r"],
                    t["mlr_r2"],
                    t["mlr_r2_adj"],
                    t["mlr_se"],
                    t["mlr_obs"]
                ],
                "Valor": [
                    f"{multiple_r:.8f}",
                    f"{r2:.8f}",
                    f"{r2_adj:.8f}",
                    f"{se_reg:.8f}",
                    f"{N}"
                ]
            })
            st.table(stats_df)
            
            st.subheader(t["mlr_anova_title"])
            anova_df = pd.DataFrame({
                "": [t["mlr_row_reg"], t["mlr_row_resid"], t["mlr_row_total"]],
                t["mlr_anova_df"]: [df_reg, df_resid, df_total],
                t["mlr_anova_ss"]: [f"{SSR:.6f}", f"{SSE:.6f}", f"{SST:.6f}"],
                t["mlr_anova_ms"]: [f"{MSR:.6f}", f"{MSE:.6f}", ""],
                t["mlr_anova_f"]: [f"{F_stat:.6f}" if df_reg > 0 else "", "", ""],
                t["mlr_anova_sig_f"]: [f"{signif_f:.3e}" if df_reg > 0 else "", "", ""]
            })
            st.table(anova_df)
            
            st.subheader(t["mlr_coefs_title"])
            coef_names = [t["mlr_intercept"]] + list(x_cols)
            coefs_df = pd.DataFrame({
                "": coef_names,
                t["mlr_col_coef"]: [f"{b:.8f}" for b in beta],
                t["mlr_col_se"]: [f"{se:.8f}" for se in se_beta],
                t["mlr_col_t"]: [f"{t_val:.8f}" for t_val in t_stats],
                t["mlr_col_p"]: [f"{p:.3e}" for p in p_values],
                t["mlr_col_low95"]: [f"{low:.8f}" for low in ci_lower],
                t["mlr_col_upp95"]: [f"{upp:.8f}" for upp in ci_upper]
            })
            st.table(coefs_df)
            
        with tab_formulas:
            st.markdown(r"""
### Estadísticas de la Regresión

| Estadística | Fórmula Visual (Cálculo) |
| :--- | :--- |
| **Coeficiente de correlación múltiple** | $R = \sqrt{R^2}$ |
| **Coeficiente de determinación ($R^2$)** | $R^2 = \frac{SSR}{SST} = \frac{\sum(\hat{y}_i - \bar{y})^2}{\sum(y_i - \bar{y})^2}$ |
| **$R^2$ ajustado** | $R^2_{adj} = 1 - (1 - R^2) \frac{(n - 1)}{(n - k - 1)}$ |
| **Error típico** | $S_e = \sqrt{\frac{SSE}{(n - k - 1)}} = \sqrt{\frac{\sum(y_i - \hat{y}_i)^2}{(n - k - 1)}}$ |
| **Observaciones** | $n = \text{Total Observaciones}$ |

<br>

### Análisis de Varianza (ANOVA)

| Fuente | Grados de libertad ($df$) | Suma de Cuadrados ($SS$) | Promedio de los Cuadrados ($MS$) | $F$ |
| :--- | :--- | :--- | :--- | :--- |
| **Regresión** | $k$ | $SSR = \sum(\hat{y}_i - \bar{y})^2$ | $MSR = \frac{SSR}{k}$ | $F = \frac{MSR}{MSE}$ |
| **Residuos** | $n - k - 1$ | $SSE = \sum(y_i - \hat{y}_i)^2$ | $MSE = \frac{SSE}{n - k - 1}$ | |
| **Total** | $n - 1$ | $SST = \sum(y_i - \bar{y})^2$ | | |

<br>

### Parámetros y Coeficientes

| Estadística | Fórmula Visual (Cálculo) |
| :--- | :--- |
| **Coeficientes ($\hat{\beta}$)** | $\hat{\beta} = (X^T X)^{-1} X^T Y$ |
| **Error típico ($SE(\hat{\beta}_j)$)** | $SE(\hat{\beta}_j) = \sqrt{MSE \cdot [(X^T X)^{-1}]_{jj}}$ |
| **Estadístico $t$** | $t = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}$ |
| **Probabilidad (p-valor)** | $p = 2 \cdot (1 - P(T \le \lvert t \rvert))$ |
| **Intervalo de Confianza (95%)** | $\hat{\beta}_j \pm t_{\alpha/2, \text{df}} \cdot SE(\hat{\beta}_j)$ |

<br>

**Leyenda:**
- **SSR, SST, SSE:** Suma de Cuadrados (Regresión, Total, Error/Residuo)
- **MSR, MSE:** Promedio de los Cuadrados (Regresión, Residuo)
- **n:** total de observaciones
- **k:** número de predictores (variables independientes)
- **$X$:** Matriz de variables independientes (incluyendo columna de unos)
- **$Y$:** Vector de la variable dependiente
- **$(X^T X)^{-1}$:** Matriz inversa de $X^T X$
- **$[ \cdot ]_{jj}$:** Elemento de la diagonal principal de la matriz
- **$t_{\alpha/2, \text{df}}$:** Valor crítico $t$ de Student con $\alpha = 0.05$
""", unsafe_allow_html=True)
            
            
        with tab_res_analysis:
            st.subheader(t["mlr_res_analysis_title"])
            
            res_df = pd.DataFrame({
                t["mlr_observation"]: np.arange(1, N + 1),
                t["mlr_predicted_value"].format(y_col): Y_pred,
                t["mlr_residual_val"]: residuals
            })
            st.dataframe(
                res_df.style.format({
                    t["mlr_predicted_value"].format(y_col): "{:.8f}",
                    t["mlr_residual_val"]: "{:.8f}"
                }),
                use_container_width=True,
                hide_index=True
            )
            
        with tab_prob_output:
            st.subheader(t["mlr_tab_prob_output"])
            
            prob_df = pd.DataFrame({
                t["mlr_prob_observation"]: np.arange(1, N + 1),
                t["mlr_prob_percentile"]: percentiles,
                t["mlr_prob_ordered_y"].format(y_col): Y_sorted
            })
            st.dataframe(
                prob_df.style.format({
                    t["mlr_prob_percentile"]: "{:.8f}",
                    t["mlr_prob_ordered_y"].format(y_col): "{:.8f}"
                }),
                use_container_width=True,
                hide_index=True
            )
            
        with tab_normal:
            fig_norm = plgo.Figure()
            fig_norm.add_trace(plgo.Scatter(
                x=percentiles, y=Y_sorted, mode='markers+lines',
                marker=dict(color='#1f77b4', size=8),
                line=dict(color='#1f77b4', width=1.5),
                hovertemplate='Percentil: %{x:.2f}%<br>Y: %{y:.4f}<extra></extra>'
            ))
            fig_norm.update_layout(
                title=t["mlr_normal_plot_title"],
                xaxis_title=t["mlr_normal_x_label"],
                yaxis_title=t["mlr_normal_y_label"],
                template='plotly_white',
                height=450
            )
            st.plotly_chart(fig_norm, use_container_width=True)
            
        with tab_plots:
            for j, x_col in enumerate(x_cols):
                st.markdown(f"### Variable X: **{x_col}**")
                col_p1, col_p2 = st.columns(2)
                x_vals_col = df_clean[x_col].values
                
                with col_p1:
                    fig_res = plgo.Figure()
                    fig_res.add_trace(plgo.Scatter(
                        x=x_vals_col, y=residuals, mode='markers',
                        marker=dict(color='#1f77b4', size=8),
                        hovertemplate='X: %{x}<br>Residuo: %{y:.4f}<extra></extra>'
                    ))
                    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_res.update_layout(
                        title=t["mlr_res_plot_title"].format(x_col),
                        xaxis_title=x_col,
                        yaxis_title=t["mlr_res_y_label"],
                        template='plotly_white',
                        height=350
                    )
                    st.plotly_chart(fig_res, use_container_width=True)
                    
                with col_p2:
                    fig_fit = plgo.Figure()
                    fig_fit.add_trace(plgo.Scatter(
                        x=x_vals_col, y=Y, mode='markers', name=t["mlr_observed"],
                        marker=dict(color='#1f77b4', size=8),
                        hovertemplate='X: %{x}<br>Obs: %{y:.4f}<extra></extra>'
                    ))
                    sort_idx = np.argsort(x_vals_col)
                    fig_fit.add_trace(plgo.Scatter(
                        x=x_vals_col[sort_idx], y=Y_pred[sort_idx], mode='lines+markers', name=t["mlr_predicted"],
                        line=dict(color='orange', width=2),
                        marker=dict(color='orange', size=6),
                        hovertemplate='X: %{x}<br>Pred: %{y:.4f}<extra></extra>'
                    ))
                    fig_fit.update_layout(
                        title=t["mlr_fit_plot_title"].format(x_col),
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        template='plotly_white',
                        height=350
                    )
                    st.plotly_chart(fig_fit, use_container_width=True)
            
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: gray; font-size: 14px;'>"
            "Desarrollado y mantenido por <b>Alexander Acosta</b> "
            "(<a href='https://github.com/j-alexander-acosta' target='_blank' style='color: #1f77b4; text-decoration: none;'>@j-alexander-acosta</a>)"
            "</p>", 
            unsafe_allow_html=True
        )
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header(t["sb_header_params"])
    eta = st.sidebar.number_input(t["sb_eta"], value=0.2000, step=0.1, format="%.4f")

    st.sidebar.markdown("---")
    st.sidebar.header(t["sb_header_nl"])
    
    nl_options = [
        t["nl_models"]["exp"],
        t["nl_models"]["log"],
        t["nl_models"]["pot"],
        t["nl_models"]["quad"]
    ]
    nl_model = st.sidebar.selectbox(
        t["sb_select_nl"],
        nl_options
    )
    
    nl_mapping = {
        t["nl_models"]["exp"]: "Exponential",
        t["nl_models"]["log"]: "Logarithmic",
        t["nl_models"]["pot"]: "Power",
        t["nl_models"]["quad"]: "Quadratic"
    }
    selected_nl_key = nl_mapping[nl_model]
    
    nl_short_names = {
        "Exponential": t["nl_models"]["exp"].split(' ')[0],
        "Logarithmic": t["nl_models"]["log"].split(' ')[0],
        "Power": t["nl_models"]["pot"].split(' ')[0],
        "Quadratic": t["nl_models"]["quad"].split(' ')[0]
    }
    selected_nl_short = nl_short_names[selected_nl_key]
    
    a_val = 0.0
    if selected_nl_key == "Exponential":
        Y_temp = df["Y"].values
        min_y = float(np.min(Y_temp))
        default_a = 5.0 if min_y > 5.0 else float(np.round(min_y - 1.0, 2))
        a_val = st.sidebar.number_input(t["sb_asymptote"], value=default_a, step=0.1, format="%.4f")
        if a_val >= min_y:
            st.sidebar.error(t["sb_asymptote_err"])




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
            if st.session_state.lang == 'es':
                exp_asymp_error_msg = f"La asíntota a ({a_nl_exp:.4f}) debe ser estrictamente menor que el valor mínimo de Y ({min_y:.4f}) para calcular ln(y - a)."
            elif st.session_state.lang == 'hi':
                exp_asymp_error_msg = f"एसिम्पटोट a ({a_nl_exp:.4f}) ln(y - a) की गणना करने के लिए Y के न्यूनतम मान ({min_y:.4f}) से कड़ाई से कम होना चाहिए।"
            else:
                exp_asymp_error_msg = f"Asymptote a ({a_nl_exp:.4f}) must be strictly less than the minimum value of Y ({min_y:.4f}) to calculate ln(y - a)."
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
        if st.session_state.lang == 'es':
            exp_asymp_error_msg = f"Error en los cálculos del modelo Exponencial con Asíntota: {e}"
        elif st.session_state.lang == 'hi':
            exp_asymp_error_msg = f"एसिम्पटोट मॉडल गणनाओं के साथ घातांकीय में त्रुटि: {e}"
        else:
            exp_asymp_error_msg = f"Error in Exponential with Asymptote model calculations: {e}"

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
        if st.session_state.lang == 'es':
            log_error_msg = "El modelo Logarítmico requiere que todos los valores de X sean estrictamente mayores a cero (X > 0) para calcular ln(X)."
        elif st.session_state.lang == 'hi':
            log_error_msg = "लघुगणकीय मॉडल के लिए सभी X मान शून्य से अधिक होने चाहिए (X > 0) ताकि ln(X) की गणना की जा सके।"
        else:
            log_error_msg = "Logarithmic model requires all X values to be strictly greater than zero (X > 0) to calculate ln(X)."
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
            if st.session_state.lang == 'es':
                log_error_msg = f"Error en los cálculos del modelo Logarítmico: {e}"
            elif st.session_state.lang == 'hi':
                log_error_msg = f"लघुगणकीय मॉडल की गणना में त्रुटि: {e}"
            else:
                log_error_msg = f"Error in Logarithmic model calculations: {e}"

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
        if st.session_state.lang == 'es':
            pot_error_msg = "El modelo Potencial requiere que todos los valores tanto de X como de Y sean estrictamente mayores a cero (X > 0, Y > 0) para aplicar ln(X) y ln(Y)."
        elif st.session_state.lang == 'hi':
            pot_error_msg = "घातीय मॉडल को ln(X) और ln(Y) लागू करने के लिए X और Y दोनों के सभी मानों को शून्य से कड़ाई से अधिक (X > 0, Y > 0) होना आवश्यक है।"
        else:
            pot_error_msg = "Power model requires all values of both X and Y to be strictly greater than zero (X > 0, Y > 0) to apply ln(X) and ln(Y)."
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
            if st.session_state.lang == 'es':
                pot_error_msg = f"Error en los cálculos del modelo Potencial: {e}"
            elif st.session_state.lang == 'hi':
                pot_error_msg = f"घातीय (पावर) मॉडल की गणना में त्रुटि: {e}"
            else:
                pot_error_msg = f"Error in Power model calculations: {e}"

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
        if st.session_state.lang == 'es':
            quad_error_msg = "El modelo Cuadrático requiere al menos 3 puntos de datos para calcular un ajuste único. Por favor, agrega más puntos en la barra lateral."
        elif st.session_state.lang == 'hi':
            quad_error_msg = "द्विघात मॉडल को एक विशिष्ट फिट की गणना करने के लिए कम से कम 3 डेटा बिंदुओं की आवश्यकता होती है। कृपया साइडबार में और बिंदु जोड़ें।"
        else:
            quad_error_msg = "Quadratic model requires at least 3 data points to calculate a unique fit. Please add more points in the sidebar."
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
            if st.session_state.lang == 'es':
                quad_error_msg = f"Error en los cálculos del modelo Cuadrático: {e}"
            elif st.session_state.lang == 'hi':
                quad_error_msg = f"द्विघात मॉडल की गणना में त्रुटि: {e}"
            else:
                quad_error_msg = f"Error in Quadratic model calculations: {e}"

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

    if selected_nl_key == "Exponential":
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
    elif selected_nl_key == "Logarithmic":
        nl_valid = log_valid
        nl_error_msg = log_error_msg
        a_nl, b_nl = a_nl_log, b_nl_log
        y_pred_nl = y_pred_nl_log
        nl_equation = nl_equation_log
        sse_nl, rmse_nl, r2_nl = sse_nl_log, rmse_nl_log, r2_nl_log
    elif selected_nl_key == "Power":
        nl_valid = pot_valid
        nl_error_msg = pot_error_msg
        a_nl, b_nl = a_nl_pot, b_nl_pot
        A_nl = A_nl_pot
        y_pred_nl = y_pred_nl_pot
        nl_equation = nl_equation_pot
        sse_nl, rmse_nl, r2_nl = sse_nl_pot, rmse_nl_pot, r2_nl_pot
    elif selected_nl_key == "Quadratic":
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
        if st.session_state.lang == 'es':
            fun_pot_error_msg = "⚠️ El modelo Función Potencia requiere que todos los valores de X e Y sean estrictamente mayores a cero (>0) para aplicar el logaritmo base 10 (log10)."
        elif st.session_state.lang == 'hi':
            fun_pot_error_msg = "⚠️ पावर फंक्शन मॉडल को बेस 10 लघुगणक (log10) लागू करने के लिए X और Y के सभी मानों को शून्य से कड़ाई से अधिक (>0) होना आवश्यक है।"
        else:
            fun_pot_error_msg = "⚠️ The Power Function model requires all X and Y values to be strictly greater than zero (>0) to apply base 10 logarithm (log10)."
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
            if st.session_state.lang == 'es':
                fun_pot_error_msg = f"Error en los cálculos de Función Potencia (base 10): {e}"
            elif st.session_state.lang == 'hi':
                fun_pot_error_msg = f"पावर फंक्शन (बेस 10) की गणना में त्रुटि: {e}"
            else:
                fun_pot_error_msg = f"Error in Power Function (base 10) calculations: {e}"

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
        if st.session_state.lang == 'es':
            fun_exp_error_msg = "⚠️ El modelo Exponencial requiere que todos los valores de Y sean estrictamente mayores a cero (>0) para aplicar el logaritmo natural (ln)."
        elif st.session_state.lang == 'hi':
            fun_exp_error_msg = "⚠️ घातांकीय मॉडल को प्राकृतिक लघुगणक (ln) लागू करने के लिए Y के सभी मानों को शून्य से कड़ाई से अधिक (>0) होना आवश्यक है।"
        else:
            fun_exp_error_msg = "⚠️ The Exponential model requires all Y values to be strictly greater than zero (>0) to apply natural logarithm (ln)."
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
            if st.session_state.lang == 'es':
                fun_exp_error_msg = f"Error en los cálculos de Función Exponencial: {e}"
            elif st.session_state.lang == 'hi':
                fun_exp_error_msg = f"घातांकीय फंक्शन की गणना में त्रुटि: {e}"
            else:
                fun_exp_error_msg = f"Error in Exponential Function calculations: {e}"

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
    st.header(t["db_title"])
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        # Gráfico Dinámico con Plotly
        fig_plotly = plgo.Figure()

        # Puntos Reales (Negro)
        fig_plotly.add_trace(plgo.Scatter(
            x=X, y=Y, mode='markers', name=t["chart_obs"],
            marker=dict(color='black', size=8),
            hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>'
        ))

        # Líneas de Tendencia (Continuas y nítidas)
        x_line = np.linspace(min(X), max(X), 100)
        
        fig_plotly.add_trace(plgo.Scatter(
            x=x_line, y=m_slr * x_line + c_slr, mode='lines', name=t["chart_slr"],
            line=dict(color='blue', width=2),
            hovertemplate='X: %{x:.2f}<br>Y (SLR): %{y:.2f}<extra></extra>'
        ))
        
        fig_plotly.add_trace(plgo.Scatter(
            x=x_line, y=m_gor * x_line + b_gor, mode='lines', name=t["chart_gor_conv"],
            line=dict(color='orange', width=2),
            hovertemplate='X: %{x:.2f}<br>Y (GOR): %{y:.2f}<extra></extra>'
        ))
        
        fig_plotly.add_trace(plgo.Scatter(
            x=x_line, y=m_prop * x_line + b_prop, mode='lines', name=t["chart_gor_prop"],
            line=dict(color='green', width=2),
            hovertemplate='X: %{x:.2f}<br>Y (Prop): %{y:.2f}<extra></extra>'
        ))

        if nl_valid:
            x_line_nl = np.linspace(min(X), max(X), 200)
            if selected_nl_key == "Exponential":
                y_line_nl = a_nl + b_nl * np.exp(c_nl * x_line_nl)
            elif selected_nl_key == "Logarithmic":
                x_line_safe = np.maximum(x_line_nl, 1e-9)
                y_line_nl = a_nl + b_nl * np.log(x_line_safe)
            elif selected_nl_key == "Power":
                x_line_safe = np.maximum(x_line_nl, 1e-9)
                y_line_nl = a_nl * (x_line_safe ** b_nl)
            elif selected_nl_key == "Quadratic":
                y_line_nl = a_nl * (x_line_nl ** 2) + b_nl * x_line_nl + c_nl
            
            fig_plotly.add_trace(plgo.Scatter(
                x=x_line_nl, y=y_line_nl, mode='lines', name=t["chart_nlin"].format(selected_nl_short),
                line=dict(color='purple', width=2.5, dash='dashdot'),
                hovertemplate='X: %{x:.2f}<br>Y (No Lin.): %{y:.2f}<extra></extra>'
            ))

        if power_valid:
            x_line_power = np.linspace(min(X), max(X), 200)
            x_line_power_safe = np.maximum(x_line_power, 1e-9)
            y_line_power = a_power * (x_line_power_safe ** b_power)
            
            fig_plotly.add_trace(plgo.Scatter(
                x=x_line_power, y=y_line_power, mode='lines', name=t["chart_power"],
                line=dict(color='coral', width=2.5, dash='dash'),
                hovertemplate='X: %{x:.2f}<br>Y (Power Law): %{y:.2f}<extra></extra>'
            ))

        fig_plotly.update_layout(
            title=t["chart_title"],
            xaxis_title=t["chart_x"],
            yaxis_title=t["chart_y"],
            hovermode='closest',
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_plotly, use_container_width=True)

    with col2:
        # Tabla Comparativa de Parámetros
        comp_data = {
            t["table_col_method"]: [t["methods_names"]["SLR"], t["methods_names"]["GOR Conv"], t["methods_names"]["GOR Prop"]],
            t["table_col_slope"]: [m_slr, m_gor, m_prop],
            t["table_col_intercept"]: [c_slr, b_gor, b_prop],
            t["table_col_se_m"]: [se_m_slr, se_m_gor, se_m_prop],
            t["table_col_se_c"]: [se_c_slr, se_b_gor, se_b_prop],
            t["table_col_rmse"]: [rmse_slr, rmse_gor, rmse_prop],
            t["table_col_r2"]: [r2_slr, r2_gor, r2_prop]
        }
        if exp_asymp_valid:
            comp_data[t["table_col_method"]].append(t["methods_names"]["No Lin. Exponencial Asintota"])
            comp_data[t["table_col_slope"]].append(np.nan)
            comp_data[t["table_col_intercept"]].append(np.nan)
            comp_data[t["table_col_se_m"]].append(np.nan)
            comp_data[t["table_col_se_c"]].append(np.nan)
            comp_data[t["table_col_rmse"]].append(rmse_nl_exp)
            comp_data[t["table_col_r2"]].append(r2_nl_exp)

        if log_valid:
            comp_data[t["table_col_method"]].append(t["methods_names"]["No Lin. Logarítmico"])
            comp_data[t["table_col_slope"]].append(np.nan)
            comp_data[t["table_col_intercept"]].append(np.nan)
            comp_data[t["table_col_se_m"]].append(np.nan)
            comp_data[t["table_col_se_c"]].append(np.nan)
            comp_data[t["table_col_rmse"]].append(rmse_nl_log)
            comp_data[t["table_col_r2"]].append(r2_nl_log)

        if pot_valid:
            comp_data[t["table_col_method"]].append(t["methods_names"]["No Lin. Potencial / Power Law"])
            comp_data[t["table_col_slope"]].append(np.nan)
            comp_data[t["table_col_intercept"]].append(np.nan)
            comp_data[t["table_col_se_m"]].append(np.nan)
            comp_data[t["table_col_se_c"]].append(np.nan)
            comp_data[t["table_col_rmse"]].append(rmse_nl_pot)
            comp_data[t["table_col_r2"]].append(r2_nl_pot)

        if quad_valid:
            comp_data[t["table_col_method"]].append(t["methods_names"]["No Lin. Cuadrático"])
            comp_data[t["table_col_slope"]].append(np.nan)
            comp_data[t["table_col_intercept"]].append(np.nan)
            comp_data[t["table_col_se_m"]].append(np.nan)
            comp_data[t["table_col_se_c"]].append(np.nan)
            comp_data[t["table_col_rmse"]].append(rmse_nl_quad)
            comp_data[t["table_col_r2"]].append(r2_nl_quad)

        if fun_pot_valid:
            comp_data[t["table_col_method"]].append(t["methods_names"]["Fun. Potencia log10 (Correcto)"])
            comp_data[t["table_col_slope"]].append(np.nan)
            comp_data[t["table_col_intercept"]].append(np.nan)
            comp_data[t["table_col_se_m"]].append(np.nan)
            comp_data[t["table_col_se_c"]].append(np.nan)
            comp_data[t["table_col_rmse"]].append(rmse_pot10_correct)
            comp_data[t["table_col_r2"]].append(r2_pot10_correct)

            comp_data[t["table_col_method"]].append(t["methods_names"]["Fun. Potencia log10 (Incorrecto)"])
            comp_data[t["table_col_slope"]].append(np.nan)
            comp_data[t["table_col_intercept"]].append(np.nan)
            comp_data[t["table_col_se_m"]].append(np.nan)
            comp_data[t["table_col_se_c"]].append(np.nan)
            comp_data[t["table_col_rmse"]].append(rmse_pot10_incorrect)
            comp_data[t["table_col_r2"]].append(r2_pot10_incorrect)

        if fun_exp_valid:
            comp_data[t["table_col_method"]].append(t["methods_names"]["Fun. Exponencial Simple ln"])
            comp_data[t["table_col_slope"]].append(np.nan)
            comp_data[t["table_col_intercept"]].append(np.nan)
            comp_data[t["table_col_se_m"]].append(np.nan)
            comp_data[t["table_col_se_c"]].append(np.nan)
            comp_data[t["table_col_rmse"]].append(rmse_exp_simple)
            comp_data[t["table_col_r2"]].append(r2_exp_simple)

        df_comp = pd.DataFrame(comp_data)
        st.markdown(t["table_title"])
        st.dataframe(df_comp.style.format({
            t["table_col_slope"]: "{:.4f}",
            t["table_col_intercept"]: "{:.4f}",
            t["table_col_se_m"]: "{:.4f}",
            t["table_col_se_c"]: "{:.4f}",
            t["table_col_rmse"]: "{:.4f}",
            t["table_col_r2"]: "{:.4f}"
        }, na_rep="-"), use_container_width=True)
        
        diff_pendiente = abs(m_slr - m_prop) / abs(m_slr) * 100 if m_slr != 0 else 0
        st.info(t["insight_text"].format(diff_pendiente))

    st.markdown("---")

    # --- MÓDULOS EDUCATIVOS ---
    st.header(t["edu_header"])
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(t["tab_titles"])



    with tab1:
        st.subheader(t["t1_title"])
        
        # Concepto e Idea de la Diapositiva 3
        col_concept, col_example = st.columns([1.2, 1])
        with col_concept:
            st.markdown(t["t1_concept_hdr"])
            st.markdown(t["t1_concept"])
            
            st.markdown(t["t1_eq_hdr"])
            st.latex(r"y = mx + c")
            st.markdown(t["t1_params_hdr"])
            st.markdown(t["t1_params"])
        with col_example:
            st.markdown(t["t1_idea_hdr"])
            st.markdown(t["t1_idea"])

        st.markdown(t["math_results"])
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t1_f_slope"])
            st.latex(r"m = \frac{n\sum(XY) - (\sum X)(\sum Y)}{n\sum X^2 - (\sum X)^2} \text{ o } m = \frac{S_{xy}}{S_{xx}}")
            st.latex(rf"\Rightarrow m = {m_slr:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t1_f_var"])
            st.latex(r"s_e^2 = \frac{\sum(Y_{obs} - Y_{pred})^2}{n-2}")
            st.latex(rf"\Rightarrow s_e^2 = {s2_e_slr:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_f2:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t1_f_intercept"])
            st.latex(r"c = \bar{Y} - m\bar{X}")
            st.latex(rf"\Rightarrow c = {c_slr:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t1_f_se_m"])
            st.latex(r"SE_m = \sqrt{\frac{s_e^2}{\sum(X_i - \bar{X})^2}}")
            st.latex(rf"\Rightarrow SE_m = {se_m_slr:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
        st.markdown(t["t1_f_se_c"])
        st.latex(r"SE_c = SE_m \sqrt{\frac{\sum X_i^2}{n}}")
        st.latex(rf"\Rightarrow SE_c = {se_c_slr:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)


    with tab2:
        st.subheader(t["t2_title"])
        st.markdown(t["t2_edu_focus"])
        
        st.markdown(t["math_results"])
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t2_f_eta"])
            st.latex(r"\eta = \frac{\sigma^2_{\varepsilon y}}{\sigma^2_{\varepsilon x}}")
            st.latex(rf"\Rightarrow \eta = {eta:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t2_f_slope"])
            st.latex(r"\hat{\beta}_1 = \frac{(S_{yy} - \eta S_{xx}) + \sqrt{(S_{yy} - \eta S_{xx})^2 + 4 \eta S_{xy}^2}}{2 S_{xy}}")
            st.latex(rf"\Rightarrow \hat{{\beta}}_1 = {m_gor:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t2_f_projections"])
            st.latex(r"X_t = \frac{\hat{\beta}_1(Y_{obs} - \hat{\beta}_0) + \eta X_{obs}}{\eta + \hat{\beta}_1^2}")
            st.latex(r"Y_t = \hat{\beta}_0 + \hat{\beta}_1 X_t")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_f2:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t2_f_intercept"])
            st.latex(r"\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}")
            st.latex(rf"\Rightarrow \hat{{\beta}}_0 = {b_gor:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t2_f_var"])
            st.latex(r"\hat{\sigma}^2 = \frac{1}{n-2} \sum_{i=1}^{n} \frac{(Y_i - \hat{\beta}_0 - \hat{\beta}_1 X_i)^2}{\hat{\beta}_1^2 + \eta}")
            st.latex(rf"\Rightarrow \hat{{\sigma}}^2 = {s2_e_gor:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t2_f_se"])
            st.latex(r"\hat{\sigma} = \sqrt{\hat{\sigma}^2}")
            st.latex(rf"\Rightarrow \hat{{\sigma}} = {sigma_gor:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.subheader(t["t3_title"])
        st.markdown(t["t3_edu_focus"])
        
        st.markdown(t["math_results"])
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t3_f_slope"])
            st.latex(r"c_1 = \frac{\sum (X_{obs,i} - \bar{X}_{obs})(Y_{t,i} - \bar{Y}_t)}{\sum (X_{obs,i} - \bar{X}_{obs})^2}")
            st.latex(rf"\Rightarrow c_1 = {m_prop:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t3_f_eq"])
            st.latex(r"Y_{t\_propuesto} = c_1 X_{obs} + c_2")
            st.markdown(t["t3_f_eq_calc"].format(m_prop, b_prop))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_f2:
            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t3_f_intercept"])
            st.latex(r"c_2 = \bar{Y}_t - c_1 \bar{X}_{obs}")
            st.latex(rf"\Rightarrow c_2 = {b_prop:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
            st.markdown(t["t3_f_rmse"])
            st.latex(r"RMSE = \sqrt{\frac{\sum (Y_{obs} - Y_{t\_propuesto})^2}{n}}")
            st.latex(rf"\Rightarrow RMSE = {rmse_prop:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.subheader(t["t4_title"])
        
        # Conceptos de las Diapositivas 4 y 5
        st.markdown(t["t4_fund_hdr"])
        col_nl_concept, col_nl_diff = st.columns([1.1, 1.2])
        
        with col_nl_concept:
            st.markdown(t["t4_what_is"])
            st.markdown(t["t4_exp_model"])
            st.latex(r"y = a + be^{cx}")
            st.markdown(t["t4_meaning"])
            st.markdown(t["t4_examples"])
            
        with col_nl_diff:
            st.markdown(t["t4_table_hdr"])
            if st.session_state.lang == 'es':
                diff_df = pd.DataFrame({
                    "Característica": ["Forma (Shape)", "Tasa de Cambio (Rate of change)", "Ecuación (Equation)", "Complejidad (Complexity)", "Ejemplo (Example)"],
                    "Regresión Lineal": ["Línea recta (Straight line)", "Constante (Constant)", "y = mx + c", "Simple", "Salario vs Experiencia"],
                    "Regresión No Lineal": ["Curva (Curve)", "Cambiante (Changing)", "y = a + be^(cx), etc.", "Más compleja (More complex)", "Sistemas de decaimiento/crecimiento"]
                })
            elif st.session_state.lang == 'hi':
                diff_df = pd.DataFrame({
                    "विशेषता (Feature)": ["आकार (Shape)", "परिवर्तन की दर (Rate of change)", "समीकरण (Equation)", "जटिलता (Complexity)", "उदाहरण (Example)"],
                    "रेखीय प्रतिगमन (Linear Regression)": ["सीधी रेखा (Straight line)", "स्थिर (Constant)", "y = mx + c", "सरल (Simple)", "वेतन बनाम अनुभव (Salary vs Experience)"],
                    "गैर-रेखीय प्रतिगमन (Non-Linear Regression)": ["वक्र (Curve)", "परिवर्तनशील (Changing)", "y = a + be^(cx), etc.", "अधिक जटिल (More complex)", "क्षय/वृद्धि प्रणालियां (Decay/growth systems)"]
                })
            else:
                diff_df = pd.DataFrame({
                    "Feature": ["Shape", "Rate of change", "Equation", "Complexity", "Example"],
                    "Linear Regression": ["Straight line", "Constant", "y = mx + c", "Simple", "Salary vs Experience"],
                    "Non-Linear Regression": ["Curve", "Changing", "y = a + be^(cx), etc.", "More complex", "Decay/growth systems"]
                })
            st.table(diff_df)
            
        st.markdown("---")

        if not nl_valid:
            st.warning(nl_error_msg)
            st.markdown(t["t4_warn_note"])
        else:
            st.markdown(t["t4_edu_focus"].format(selected_nl_short))
            
            st.markdown(t["t4_process_hdr"])
            col_f1, col_f2 = st.columns(2)
            
            if selected_nl_key == "Exponential":
                cols_lin = ["x", "Y", "y-a", "ln(y-a)", "Y est", "Residual"] if st.session_state.lang == 'en' else (["x", "Y", "y-a", "ln(y-a)", "Y अनुमानित", "अवशेष"] if st.session_state.lang == 'hi' else ["x", "Y", "y-a", "ln(y-a)", "Y estima", "Residuo"])
                df_linearized = pd.DataFrame({
                    cols_lin[0]: X,
                    cols_lin[1]: Y,
                    cols_lin[2]: y_minus_a,
                    cols_lin[3]: y_trans,
                    cols_lin[4]: y_pred_nl,
                    cols_lin[5]: Y - y_pred_nl
                })
                st.markdown(t["t4_table_interm"])
                st.dataframe(df_linearized.style.format({
                    cols_lin[0]: "{:g}",
                    cols_lin[1]: "{:.2f}",
                    cols_lin[2]: "{:.2f}",
                    cols_lin[3]: "{:.6f}",
                    cols_lin[4]: "{:.6f}",
                    cols_lin[5]: "{:.6f}"
                }), use_container_width=True)
                
                # 2. Sumatorias e Parámetros
                col_math1, col_math2 = st.columns([1.1, 1])
                
                with col_math1:
                    st.markdown(t["t4_sums_hdr"])
                    st.markdown(f"""
                    *   {t['t4_n_points'].format(n)}
                    *   {t['t4_sum_x'].format(sum_x)}
                    *   {t['t4_sum_x2'].format(sum_x2)}
                    *   {t['t4_sum_ln_ya'].format(sum_y_trans)}
                    *   {t['t4_sum_x_ln_ya'].format(sum_x_y_trans)}
                    """)
                    
                    st.markdown(t["t4_formulas_params"])
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
                        x=X, y=y_trans, mode='markers', name=t["t4_graph_lin_dots"],
                        marker=dict(color='#1f77b4', size=10, symbol='circle'),
                        hovertemplate='x: %{x}<br>ln(y-a): %{y:.6f}<extra></extra>'
                    ))
                    
                    # Recta linealizada
                    x_line = np.linspace(min(X), max(X), 100)
                    y_line_trans = c_nl * x_line + A_nl
                    
                    fig_lin.add_trace(plgo.Scatter(
                        x=x_line, y=y_line_trans, mode='lines', name=t["t4_graph_lin_fit"],
                        line=dict(color='red', width=2, dash='dash'),
                        hovertemplate='x: %{x:.2f}<br>ln(y-a) pred: %{y:.6f}<extra></extra>'
                    ))
                    
                    sign_A = "+" if A_nl >= 0 else "-"
                    abs_A = abs(A_nl)
                    fig_lin.update_layout(
                        title=t["t4_graph_lin_title"].format(c_nl, sign_A, abs_A, r2_linearized),
                        xaxis_title=t["t4_graph_lin_x"],
                        yaxis_title=t["t4_graph_lin_y"],
                        template='plotly_white',
                        showlegend=False,
                        height=350,
                        margin=dict(l=40, r=40, t=60, b=40)
                    )
                    st.plotly_chart(fig_lin, use_container_width=True)
                    
                st.markdown(t["t4_space_orig_hdr"])
                st.markdown(t["t4_eq_adjusted"].format(a_nl, '+' if b_nl >= 0 else '-', abs(b_nl), c_nl))
                    
            elif selected_nl_key == "Logarithmic":
                with col_f1:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown(t["t4_eq_orig_lin"])
                    st.latex(r"Y = a + b \ln(X)")
                    if st.session_state.lang == 'es':
                        st.markdown("Definiendo $X' = \\ln(X)$, ajustamos la recta:")
                    elif st.session_state.lang == 'hi':
                        st.markdown("$X' = \\ln(X)$ परिभाषित करते हुए, हम रेखा को फिट करते हैं:")
                    else:
                        st.markdown("Defining $X' = \\ln(X)$, we fit the line:")
                    st.latex(r"Y = a + b X'")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown(t["t4_model_coefs"])
                    st.latex(rf"\Rightarrow a = {a_nl:.4f}")
                    st.latex(rf"\Rightarrow b = {b_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_f2:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown(t["t1_f_slope"])
                    st.latex(r"b = \frac{\sum (\ln(X_i) - \bar{\ln(X)})(Y_i - \bar{Y})}{\sum (\ln(X_i) - \bar{\ln(X)})^2}")
                    st.latex(rf"\Rightarrow b = {b_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown(t["t1_f_intercept"])
                    st.latex(r"a = \bar{Y} - b \bar{\ln(X)}")
                    st.latex(rf"\Rightarrow a = {a_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            elif selected_nl_key == "Power":
                with col_f1:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown(t["t4_eq_orig_lin"])
                    st.latex(r"Y = a \cdot X^b \implies \ln(Y) = \ln(a) + b \ln(X)")
                    if st.session_state.lang == 'es':
                        st.markdown("Definiendo $X' = \\ln(X)$, $Y' = \\ln(Y)$ y $A = \\ln(a)$, ajustamos:")
                    elif st.session_state.lang == 'hi':
                        st.markdown("$X' = \\ln(X)$, $Y' = \\ln(Y)$ और $A = \\ln(a)$ परिभाषित करते हुए, हम फिट करते हैं:")
                    else:
                        st.markdown("Defining $X' = \\ln(X)$, $Y' = \\ln(Y)$ and $A = \\ln(a)$, we fit:")
                    st.latex(r"Y' = A + b X'")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown(t["t4_formulas_params"])
                    st.latex(rf"b = {b_nl:.4f}")
                    st.latex(r"a = e^A")
                    st.latex(rf"\Rightarrow a = e^{{{A_nl:.4f}}} = {a_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_f2:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown(t["t1_f_slope"])
                    st.latex(r"b = \frac{\sum (\ln(X_i) - \bar{\ln(X)})(\ln(Y_i) - \bar{\ln(Y)})}{\sum (\ln(X_i) - \bar{\ln(X)})^2}")
                    st.latex(rf"\Rightarrow b = {b_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown(t["t1_f_intercept"])
                    st.latex(r"A = \bar{\ln(Y)} - b \bar{\ln(X)}")
                    st.latex(rf"\Rightarrow A = {A_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            elif selected_nl_key == "Quadratic":
                with col_f1:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown(t["t4_quad_eq"])
                    st.latex(r"Y = a X^2 + b X + c")
                    st.markdown("A pesar de ser curvilíneo, es **lineal en sus parámetros** ($a, b, c$)." if st.session_state.lang == 'es' else ("वक्राकार होने के बावजूद, यह अपने मापदंडों ($a, b, c$) में **रेखीय** है।" if st.session_state.lang == 'hi' else "Despite being curvilinear, it is **linear in its parameters** ($a, b, c$)."))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown(t["t4_coefs_calc"])
                    st.latex(rf"\Rightarrow a = {a_nl:.4f}")
                    st.latex(rf"\Rightarrow b = {b_nl:.4f}")
                    st.latex(rf"\Rightarrow c = {c_nl:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_f2:
                    st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                    st.markdown(t["t4_normal_eq_sys"])
                    st.latex(r"\begin{pmatrix} \sum X_i^4 & \sum X_i^3 & \sum X_i^2 \\ \sum X_i^3 & \sum X_i^2 & \sum X_i \\ \sum X_i^2 & \sum X_i & n \end{pmatrix} \begin{pmatrix} a \\ b \\ c \end{pmatrix} = \begin{pmatrix} \sum X_i^2 Y_i \\ \sum X_i Y_i \\ \sum Y_i \end{pmatrix}")
                    st.markdown("Se resuelve directamente por álgebra matricial lineal." if st.session_state.lang == 'es' else ("यह सीधे रेखीय आव्यूह बीजगणित द्वारा हल किया जाता है।" if st.session_state.lang == 'hi' else "It is solved directly by linear matrix algebra."))
                    st.markdown("</div>", unsafe_allow_html=True)

            st.success(t["t4_eq_adjusted_generic"].format(nl_equation))
            
            # Comparación específica de métricas
            st.markdown(t["t4_metrics_comp"])
            metrics_nl_data = {
                t["t4_metrics_cols"][0]: t["t4_metrics_rows"],
                t["t4_metrics_cols"][1]: [sse_slr, rmse_slr, r2_slr],
                t["t4_metrics_cols"][2].format(selected_nl_short): [sse_nl, rmse_nl, r2_nl]
            }
            df_metrics_nl = pd.DataFrame(metrics_nl_data)
            st.dataframe(df_metrics_nl.style.format({
                t["t4_metrics_cols"][1]: "{:.4f}",
                t["t4_metrics_cols"][2].format(selected_nl_short): "{:.4f}"
            }), use_container_width=True)
            
            # Calculadora de predicción interactiva
            st.markdown("---")
            st.markdown(t["t4_calc_hdr"])
            x_input = st.number_input(t["t4_calc_input"].format(selected_nl_short), value=float(np.mean(X)), format="%.4f")
            
            y_pred_calc = None
            if selected_nl_key == "Exponential":
                y_pred_calc = a_nl + b_nl * np.exp(c_nl * x_input)
                sign_b = "+" if b_nl >= 0 else "-"
                abs_b = abs(b_nl)
                calc_latex = rf"y = {a_nl:.4f} {sign_b} {abs_b:.4f}e^{{{c_nl:.4f} \cdot {x_input:.4f}}} = {y_pred_calc:.4f}"
            elif selected_nl_key == "Logarithmic":
                if x_input <= 0:
                    st.error(t["t4_calc_err_log"])
                else:
                    y_pred_calc = a_nl + b_nl * np.log(x_input)
                    calc_latex = rf"Y = {a_nl:.4f} + {b_nl:.4f} \cdot \ln({x_input:.4f}) = {y_pred_calc:.4f}"
            elif selected_nl_key == "Power":
                if x_input <= 0:
                    st.error(t["t4_calc_err_pot"])
                else:
                    y_pred_calc = a_nl * (x_input ** b_nl)
                    calc_latex = rf"Y = {a_nl:.4f} \cdot {x_input:.4f}^{{{b_nl:.4f}}} = {y_pred_calc:.4f}"
            elif selected_nl_key == "Quadratic":
                y_pred_calc = a_nl * (x_input**2) + b_nl * x_input + c_nl
                calc_latex = rf"Y = {a_nl:.4f} \cdot ({x_input:.4f})^2 + {b_nl:.4f} \cdot ({x_input:.4f}) + ({c_nl:.4f}) = {y_pred_calc:.4f}"
                
            if y_pred_calc is not None:
                st.latex(calc_latex)
                st.markdown(t["t4_calc_res_desc"].format(x_input, y_pred_calc))

    with tab5:
        st.subheader(t["t5_title"])
        if not power_valid:
            st.warning(power_error_msg)
            st.markdown(t["t5_warn_note"])
        else:
            st.markdown(t["t5_edu_focus"])
            
            st.markdown(t["t4_process_hdr"])
            
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
            st.markdown(t["t4_table_interm"])
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
                st.markdown(t["t4_sums_hdr"])
                st.markdown(f"""
                *   {t['t4_n_points'].format(n)}
                *   {t['t5_sum_x_trans'].format(sum_x_trans_power)}
                *   {t['t5_sum_x2_trans'].format(sum_x2_trans_power)}
                *   {t['t5_sum_y_trans'].format(sum_y_trans_power)}
                *   {t['t5_sum_xy_trans'].format(sum_xy_trans_power)}
                """)
                
                st.markdown(t["t4_formulas_params"])
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
                    x=X, y=Y, mode='markers', name=t["chart_obs"],
                    marker=dict(color='black', size=10, symbol='circle'),
                    hovertemplate='x: %{x}<br>Y: %{y:.2f}<extra></extra>'
                ))
                
                # Curva de potencia
                x_curve = np.linspace(min(X), max(X), 100)
                y_curve_power = a_power * (x_curve ** b_power)
                
                fig_power.add_trace(plgo.Scatter(
                    x=x_curve, y=y_curve_power, mode='lines', name=t["t4_graph_lin_fit"],
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='x: %{x:.2f}<br>Y estima: %{y:.4f}<extra></extra>'
                ))
                
                fig_power.update_layout(
                    title=t["t5_graph_title"].format(a_power, b_power, r2_power),
                    xaxis_title="x",
                    yaxis_title="Y",
                    template='plotly_white',
                    showlegend=False,
                    height=350,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_power, use_container_width=True)
                
            st.markdown(t["t5_res_hdr"])
            st.success(t["t5_eq_adjusted"].format(a_power, b_power))
            
            # Comparativa de métricas
            st.markdown(t["t5_metrics_comp"])
            metrics_power_data = {
                t["t5_metrics_cols"][0]: t["t4_metrics_rows"],
                t["t5_metrics_cols"][1]: [sse_slr, rmse_slr, r2_slr],
                t["t5_metrics_cols"][2]: [sse_power, rmse_power, r2_power]
            }
            df_metrics_power = pd.DataFrame(metrics_power_data)
            st.dataframe(df_metrics_power.style.format({
                t["t5_metrics_cols"][1]: "{:.4f}",
                t["t5_metrics_cols"][2]: "{:.4f}"
            }), use_container_width=True)
            
            # Calculadora de predicción interactiva
            st.markdown("---")
            st.markdown(t["t4_calc_hdr"])
            x_input = st.number_input(t["t5_calc_input"], value=float(np.mean(X)), format="%.4f")
            
            if x_input <= 0:
                st.error(t["t5_calc_err"])
            else:
                y_pred_calc = a_power * (x_input ** b_power)
                calc_latex = rf"y = {a_power:.4f} \cdot {x_input:.4f}^{{{b_power:.4f}}} = {y_pred_calc:.4f}"
                st.latex(calc_latex)
                st.markdown(t["t5_calc_res"].format(x_input, y_pred_calc))

    with tab6:
        st.subheader(t["t6_title"])
        st.markdown(t["t6_desc"])
        
        X_mod = df["X"].values
        Y_mod = df["Y"].values

        if not fun_pot_valid:
            st.error(fun_pot_error_msg)
            st.info(t["t6_warn_note"])
        else:
            r2_lin = r2_lin_pot10
            df_mod_calc = pd.DataFrame({
                "x": X,
                "y": Y,
                "Log Y": Y_log,
                "Log X": X_log
            })

            st.markdown(t["t6_table_hdr"])
            st.dataframe(df_mod_calc.style.format({
                "x": "{:g}",
                "y": "{:g}",
                "Log Y": "{:.5f}",
                "Log X": "{:.5f}"
            }), use_container_width=True)

            col_coef1, col_coef2 = st.columns(2)
            with col_coef1:
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown(t["t6_coefs_lin"])
                st.latex(rf"\log_{{10}}(y) = {m_log:.4f} \cdot \log_{{10}}(x) + {c_log:.4f}")
                st.latex(rf"R^2 = {r2_lin:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_coef2:
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown(t["t6_coefs_adj"])
                st.markdown(t["t6_coefs_log_a"].format(c_log))
                st.markdown(t["t6_coefs_log_b"].format(m_log))
                st.markdown(t["t6_coefs_a"].format(a_correct))
                st.markdown(t["t6_coefs_b_corr"].format(b_correct))
                st.markdown(t["t6_coefs_b_inc"].format(b_err))
                st.markdown("</div>", unsafe_allow_html=True)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_lin_power = plgo.Figure()
                fig_lin_power.add_trace(plgo.Scatter(
                    x=X_log, y=Y_log, mode='markers', name=t["t4_graph_lin_dots"],
                    marker=dict(color='#1f77b4', size=10, symbol='circle'),
                    hovertemplate='Log X: %{x:.5f}<br>Log Y: %{y:.5f}<extra></extra>'
                ))
                x_log_line = np.linspace(min(X_log), max(X_log), 100)
                y_log_line = m_log * x_log_line + c_log
                fig_lin_power.add_trace(plgo.Scatter(
                    x=x_log_line, y=y_log_line, mode='lines', name=t["t4_graph_lin_fit"],
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    hovertemplate='Log X: %{x:.5f}<br>Log Y Pred: %{y:.5f}<extra></extra>'
                ))
                fig_lin_power.update_layout(
                    title=t["t6_graph_lin_title"].format(m_log, c_log, r2_lin),
                    xaxis_title=t["t6_graph_lin_x"],
                    yaxis_title=t["t6_graph_lin_y"],
                    template='plotly_white',
                    showlegend=False,
                    height=380,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_lin_power, use_container_width=True)

            with col_g2:
                fig_orig_power = plgo.Figure()
                fig_orig_power.add_trace(plgo.Scatter(
                    x=X_mod, y=Y_mod, mode='markers', name=t["chart_obs"],
                    marker=dict(color='black', size=10, symbol='circle'),
                    hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>'
                ))

                x_orig_line = np.linspace(min(X_mod), max(X_mod), 200)
                y_orig_correct = a_correct * (x_orig_line ** b_correct)
                
                fig_orig_power.add_trace(plgo.Scatter(
                    x=x_orig_line, y=y_orig_correct, mode='lines', name=t["t6_graph_orig_correct"],
                    line=dict(color='green', width=2.5),
                    hovertemplate='X: %{x:.2f}<br>Y (Correcto): %{y:.4f}<extra></extra>'
                ))

                try:
                    y_orig_err = a_err * (x_orig_line ** b_err)
                    if not np.any(np.isinf(y_orig_err)) and np.max(y_orig_err) < 1000000:
                        fig_orig_power.add_trace(plgo.Scatter(
                            x=x_orig_line, y=y_orig_err, mode='lines', name=t["t6_graph_orig_incorrect"],
                            line=dict(color='red', width=2.5, dash='dot'),
                            hovertemplate='X: %{x:.2f}<br>Y (Despeje Erróneo): %{y:.4f}<extra></extra>'
                        ))
                except Exception:
                    pass

                fig_orig_power.update_layout(
                    title=t["t6_graph_orig_title"],
                    xaxis_title="X",
                    yaxis_title="Y",
                    template='plotly_white',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    height=380,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_orig_power, use_container_width=True)

            st.warning(t["t6_analysis_hdr"])
            st.markdown(t["t6_analysis_body"].format(c_log, c_log, a_correct, m_log, a_correct, b_correct, m_log, b_err, a_err, b_err))

            st.markdown(t["t6_math_formulation"])
            st.markdown(t["t6_math_steps"])

    with tab7:
        st.subheader(t["t7_title"])
        st.markdown(t["t7_desc"])
        
        X_mod = df["X"].values
        Y_mod = df["Y"].values

        if not fun_exp_valid:
            st.error(fun_exp_error_msg)
            st.info(t["t6_warn_note"])
        else:
            r2_lin = r2_lin_exp
            df_mod_calc = pd.DataFrame({
                "x": X,
                "y": Y,
                "Ln Y": Y_log_exp
            })

            st.markdown(t["t7_table_hdr"])
            st.dataframe(df_mod_calc.style.format({
                "x": "{:g}",
                "y": "{:g}",
                "Ln Y": "{:.8f}"
            }), use_container_width=True)

            col_coef1, col_coef2 = st.columns(2)
            with col_coef1:
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown(t["t7_coefs_lin"])
                st.latex(rf"\ln(y) = {b_coeff:.4f} \cdot x + {ln_a:.4f}")
                st.latex(rf"R^2 = {r2_lin:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_coef2:
                st.markdown("<div class='latex-container'>", unsafe_allow_html=True)
                st.markdown(t["t7_coefs_adj"])
                st.markdown(t["t7_coefs_ln_a"].format(ln_a))
                st.markdown(t["t7_coefs_b"].format(b_coeff))
                st.markdown(t["t7_coefs_a"].format(a_coeff))
                st.markdown(t["t7_eq_final"].format(a_coeff, b_coeff))
                st.markdown("</div>", unsafe_allow_html=True)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_lin_exp = plgo.Figure()
                fig_lin_exp.add_trace(plgo.Scatter(
                    x=X_mod, y=Y_log, mode='markers', name=t["t4_graph_lin_dots"],
                    marker=dict(color='#1f77b4', size=10, symbol='circle'),
                    hovertemplate='X: %{x}<br>ln(Y): %{y:.5f}<extra></extra>'
                ))
                x_line = np.linspace(min(X_mod), max(X_mod), 100)
                y_line = b_coeff * x_line + ln_a
                fig_lin_exp.add_trace(plgo.Scatter(
                    x=x_line, y=y_line, mode='lines', name=t["t4_graph_lin_fit"],
                    line=dict(color='#ff7f0e', width=2),
                    hovertemplate='X: %{x:.2f}<br>ln(Y) Pred: %{y:.5f}<extra></extra>'
                ))
                fig_lin_exp.update_layout(
                    title=t["t7_graph_lin_title"].format(b_coeff, ln_a, r2_lin),
                    xaxis_title="x",
                    yaxis_title=t["t7_graph_lin_y"],
                    template='plotly_white',
                    showlegend=False,
                    height=380,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_lin_exp, use_container_width=True)

            with col_g2:
                fig_orig_exp = plgo.Figure()
                fig_orig_exp.add_trace(plgo.Scatter(
                    x=X_mod, y=Y_mod, mode='markers', name=t["chart_obs"],
                    marker=dict(color='black', size=10, symbol='circle'),
                    hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>'
                ))

                x_orig_line = np.linspace(min(X_mod), max(X_mod), 200)
                y_orig_fit = a_coeff * np.exp(b_coeff * x_orig_line)

                fig_orig_exp.add_trace(plgo.Scatter(
                    x=x_orig_line, y=y_orig_fit, mode='lines', name=t["t7_graph_orig_curve"],
                    line=dict(color='green', width=2.5),
                    hovertemplate='X: %{x:.2f}<br>Y estima: %{y:.4f}<extra></extra>'
                ))

                fig_orig_exp.update_layout(
                    title=t["t7_graph_orig_title"],
                    xaxis_title="X",
                    yaxis_title="Y",
                    template='plotly_white',
                    showlegend=False,
                    height=380,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_orig_exp, use_container_width=True)

            st.info(t["t7_analysis_hdr"])
            st.markdown(t["t7_analysis_body"].format(ln_a, ln_a, a_coeff, b_coeff, a_coeff, b_coeff))

            st.markdown(t["t7_math_formulation"])
            st.markdown(t["t7_math_steps"])

    with tab8:
        st.subheader(t["t8_title_data"])
        st.markdown(t["t8_desc_data"])
        # Translate df_results columns for presentation
        rename_cols = {
            "X": "X",
            "Y": "Y",
            "Y_est (SLR)": t["xls_details_y_est"].format(t["methods_names"]["SLR"]),
            "Residuo (SLR)": t["xls_details_residual"].format(t["methods_names"]["SLR"]),
            "Y_est (GOR Conv)": t["xls_details_y_est"].format(t["methods_names"]["GOR Conv"]),
            "Residuo (GOR Conv)": t["xls_details_residual"].format(t["methods_names"]["GOR Conv"]),
            "X_t (GOR)": "X_t (GOR)",
            "Y_t (GOR)": "Y_t (GOR)",
            "Y_est (GOR Prop)": t["xls_details_y_est"].format(t["methods_names"]["GOR Prop"]),
            "Residuo (GOR Prop)": t["xls_details_residual"].format(t["methods_names"]["GOR Prop"]),
        }
        if exp_asymp_valid:
            rename_cols["Y_est (No Lin. Exponencial Asintota)"] = t["xls_details_y_est"].format(t["methods_names"]["No Lin. Exponencial Asintota"])
            rename_cols["Residuo (No Lin. Exponencial Asintota)"] = t["xls_details_residual"].format(t["methods_names"]["No Lin. Exponencial Asintota"])
        if log_valid:
            rename_cols["Y_est (No Lin. Logaritmico)"] = t["xls_details_y_est"].format(t["methods_names"]["No Lin. Logarítmico"])
            rename_cols["Residuo (No Lin. Logaritmico)"] = t["xls_details_residual"].format(t["methods_names"]["No Lin. Logarítmico"])
        if pot_valid:
            rename_cols["Y_est (No Lin. Potencial / Power Law)"] = t["xls_details_y_est"].format(t["methods_names"]["No Lin. Potencial / Power Law"])
            rename_cols["Residuo (No Lin. Potencial / Power Law)"] = t["xls_details_residual"].format(t["methods_names"]["No Lin. Potencial / Power Law"])
        if quad_valid:
            rename_cols["Y_est (No Lin. Cuadratico)"] = t["xls_details_y_est"].format(t["methods_names"]["No Lin. Cuadrático"])
            rename_cols["Residuo (No Lin. Cuadratico)"] = t["xls_details_residual"].format(t["methods_names"]["No Lin. Cuadrático"])
        if fun_pot_valid:
            rename_cols["Y_est (Fun. Potencia log10 - Correcto)"] = t["xls_details_y_est"].format(t["methods_names"]["Fun. Potencia log10 (Correcto)"])
            rename_cols["Residuo (Fun. Potencia log10 - Correcto)"] = t["xls_details_residual"].format(t["methods_names"]["Fun. Potencia log10 (Correcto)"])
            rename_cols["Y_est (Fun. Potencia log10 - Incorrecto)"] = t["xls_details_y_est"].format(t["methods_names"]["Fun. Potencia log10 (Incorrecto)"])
            rename_cols["Residuo (Fun. Potencia log10 - Incorrecto)"] = t["xls_details_residual"].format(t["methods_names"]["Fun. Potencia log10 (Incorrecto)"])
        if fun_exp_valid:
            rename_cols["Y_est (Fun. Exponencial Simple ln)"] = t["xls_details_y_est"].format(t["methods_names"]["Fun. Exponencial Simple ln"])
            rename_cols["Residuo (Fun. Exponencial Simple ln)"] = t["xls_details_residual"].format(t["methods_names"]["Fun. Exponencial Simple ln"])

        df_results_translated = df_results.rename(columns=rename_cols)
        st.dataframe(df_results_translated, use_container_width=True)

        st.markdown("---")
        st.subheader(t["t8_guide_title"])
        st.markdown(t["t8_guide_desc"])
        
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
                "name": "No Lin. Exponencial Asintota",
                "pred": lambda r: f"= {a_val:.6f} + {b_nl_exp:.6f} * EXP({c_nl_exp:.6f} * B{r})",
            })
        if log_valid:
            models_details_guide.append({
                "name": "No Lin. Logarítmico",
                "pred": lambda r: f"= {a_nl_log:.6f} + {b_nl_log:.6f} * LN(B{r})",
            })
        if pot_valid:
            models_details_guide.append({
                "name": "No Lin. Potencial / Power Law",
                "pred": lambda r: f"= {a_nl_pot:.6f} * (B{r} ^ {b_nl_pot:.6f})",
            })
        if quad_valid:
            models_details_guide.append({
                "name": "No Lin. Cuadrático",
                "pred": lambda r: f"= {a_nl_quad:.6f} * (B{r} ^ 2) + {b_nl_quad:.6f} * B{r} + {c_nl_quad:.6f}",
            })
        if fun_pot_valid:
            models_details_guide.append({
                "name": "Fun. Potencia log10 (Correcto)",
                "pred": lambda r: f"= {a_correct:.6f} * (B{r} ^ {b_correct:.6f})",
            })
            models_details_guide.append({
                "name": "Fun. Potencia log10 (Incorrecto)",
                "pred": lambda r: f"= {a_err:.6f} * (B{r} ^ {b_err:.6f})",
            })
        if fun_exp_valid:
            models_details_guide.append({
                "name": "Fun. Exponencial Simple ln",
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

        cols = t["t8_guide_table_cols"]
        col_desc = t["t8_guide_table_col_desc"]
        
        guide_data = {
            cols[0]: [],
            cols[1]: [],
            cols[2]: []
        }

        for m in models_details_guide:
            pred_col, res_col = col_letters[m["name"]]
            translated_name = t["methods_names"].get(m["name"], m["name"])
            guide_data[cols[0]].append(translated_name)
            guide_data[cols[1]].append(f"`{m['pred'](6)}` ({col_desc} {pred_col})")
            guide_data[cols[2]].append(f"`= C6 - {pred_col}6` ({col_desc} {res_col})")

        df_guide = pd.DataFrame(guide_data)
        st.table(df_guide)

        st.markdown("---")
        st.subheader(t["t8_export_title"])
        col_out1, col_out2 = st.columns(2)
        
        with col_out1:
            st.markdown(f"**{t['t8_export_excel_lbl']}**")
            
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
                ws_summary = wb.create_sheet(title=t["xls_sheet_summary"])
                ws_summary.views.sheetView[0].showGridLines = True
                
                ws_summary.cell(row=2, column=2, value=t["xls_summary_title"]).font = title_font
                ws_summary.cell(row=3, column=2, value=t["xls_summary_subtitle"]).font = subtitle_font
                
                headers_summary = t["xls_summary_headers"]
                for col_idx, header in enumerate(headers_summary, start=2):
                    cell = ws_summary.cell(row=5, column=col_idx, value=header)
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = align_center
                    cell.border = thin_border
                
                methods_list = [
                    {"name": t["methods_names"]["SLR"], "math": "y = m*x + c", "formula": "= m * X + c", "valid": True, "rmse": rmse_slr, "r2": r2_slr},
                    {"name": t["methods_names"]["GOR Conv"], "math": "y = m*x + c", "formula": "= m * X + c", "valid": True, "rmse": rmse_gor, "r2": r2_gor},
                    {"name": t["methods_names"]["GOR Prop"], "math": "y = m*x + c", "formula": "= m * X + c", "valid": True, "rmse": rmse_prop, "r2": r2_prop},
                    {"name": t["methods_names"]["No Lin. Exponencial Asintota"], "math": "y = a + b*e^(cx)", "formula": "= a + b * EXP(c * X)", "valid": exp_asymp_valid, "rmse": rmse_nl_exp if exp_asymp_valid else np.nan, "r2": r2_nl_exp if exp_asymp_valid else np.nan},
                    {"name": t["methods_names"]["No Lin. Logarítmico"], "math": "y = a + b*ln(x)", "formula": "= a + b * LN(X)", "valid": log_valid, "rmse": rmse_nl_log if log_valid else np.nan, "r2": r2_nl_log if log_valid else np.nan},
                    {"name": t["methods_names"]["No Lin. Potencial / Power Law"], "math": "y = a*x^b", "formula": "= a * (X ^ b)", "valid": pot_valid, "rmse": rmse_nl_pot if pot_valid else np.nan, "r2": r2_nl_pot if pot_valid else np.nan},
                    {"name": t["methods_names"]["No Lin. Cuadrático"], "math": "y = a*x^2 + b*x + c", "formula": "= a * (X ^ 2) + b * X + c", "valid": quad_valid, "rmse": rmse_nl_quad if quad_valid else np.nan, "r2": r2_nl_quad if quad_valid else np.nan},
                    {"name": t["methods_names"]["Fun. Potencia log10 (Correcto)"], "math": "y = a*x^b", "formula": "= a * (X ^ b)", "valid": fun_pot_valid, "rmse": rmse_pot10_correct if fun_pot_valid else np.nan, "r2": r2_pot10_correct if fun_pot_valid else np.nan},
                    {"name": t["methods_names"]["Fun. Potencia log10 (Incorrecto)"], "math": "y = a_err * x^b_err", "formula": "= a_err * (X ^ b_err)", "valid": fun_pot_valid, "rmse": rmse_pot10_incorrect if fun_pot_valid else np.nan, "r2": r2_pot10_incorrect if fun_pot_valid else np.nan},
                    {"name": t["methods_names"]["Fun. Exponencial Simple ln"], "math": "y = a*e^(bx)", "formula": "= a * EXP(b * X)", "valid": fun_exp_valid, "rmse": rmse_exp_simple if fun_exp_valid else np.nan, "r2": r2_exp_simple if fun_exp_valid else np.nan},
                ]
                
                current_row = 6
                for m in methods_list:
                    ws_summary.cell(row=current_row, column=2, value=m["name"]).font = regular_font
                    ws_summary.cell(row=current_row, column=3, value=m["math"]).font = regular_font
                    ws_summary.cell(row=current_row, column=4, value=m["formula"]).font = regular_font
                    
                    if m["valid"]:
                        ws_summary.cell(row=current_row, column=5, value=m["rmse"]).number_format = "0.0000"
                        ws_summary.cell(row=current_row, column=6, value=m["r2"]).number_format = "0.0000"
                        ws_summary.cell(row=current_row, column=7, value=t["xls_summary_status_fit"]).font = regular_font
                    else:
                        ws_summary.cell(row=current_row, column=5, value="-").alignment = align_center
                        ws_summary.cell(row=current_row, column=6, value="-").alignment = align_center
                        ws_summary.cell(row=current_row, column=7, value=t["xls_summary_status_invalid"]).font = Font(name="Calibri", size=11, color="FF0000")
                        
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
                ws_details = wb.create_sheet(title=t["xls_sheet_details"])
                ws_details.views.sheetView[0].showGridLines = True
                
                ws_details.cell(row=2, column=2, value=t["xls_details_title"]).font = title_font
                ws_details.cell(row=3, column=2, value=t["xls_details_subtitle"]).font = subtitle_font
                
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
                        "name": "No Lin. Exponencial Asintota",
                        "pred": lambda r: f"= {a_val:.12f} + {b_nl_exp:.12f} * EXP({c_nl_exp:.12f} * B{r})",
                    })
                if log_valid:
                    models_details.append({
                        "name": "No Lin. Logarítmico",
                        "pred": lambda r: f"= {a_nl_log:.12f} + {b_nl_log:.12f} * LN(B{r})",
                    })
                if pot_valid:
                    models_details.append({
                        "name": "No Lin. Potencial / Power Law",
                        "pred": lambda r: f"= {a_nl_pot:.12f} * (B{r} ^ {b_nl_pot:.12f})",
                    })
                if quad_valid:
                    models_details.append({
                        "name": "No Lin. Cuadrático",
                        "pred": lambda r: f"= {a_nl_quad:.12f} * (B{r} ^ 2) + {b_nl_quad:.12f} * B{r} + {c_nl_quad:.12f}",
                    })
                if fun_pot_valid:
                    models_details.append({
                        "name": "Fun. Potencia log10 (Correcto)",
                        "pred": lambda r: f"= {a_correct:.12f} * (B{r} ^ {b_correct:.12f})",
                    })
                    models_details.append({
                        "name": "Fun. Potencia log10 (Incorrecto)",
                        "pred": lambda r: f"= {a_err:.12f} * (B{r} ^ {b_err:.12f})",
                    })
                if fun_exp_valid:
                    models_details.append({
                        "name": "Fun. Exponencial Simple ln",
                        "pred": lambda r: f"= {a_coeff:.12f} * EXP({b_coeff:.12f} * B{r})",
                    })
                
                # Write Headers
                ws_details.cell(row=5, column=2, value=t["xls_details_col_x"]).font = header_font
                ws_details.cell(row=5, column=2).fill = header_fill
                ws_details.cell(row=5, column=2).border = thin_border
                ws_details.cell(row=5, column=2).alignment = align_center
                
                ws_details.cell(row=5, column=3, value=t["xls_details_col_y"]).font = header_font
                ws_details.cell(row=5, column=3).fill = header_fill
                ws_details.cell(row=5, column=3).border = thin_border
                ws_details.cell(row=5, column=3).alignment = align_center
                
                col_idx = 4
                for m in models_details:
                    cell_est = ws_details.cell(row=5, column=col_idx, value=t["xls_details_y_est"].format(t["methods_names"][m["name"]]))
                    cell_est.font = header_font
                    cell_est.fill = header_fill
                    cell_est.border = thin_border
                    cell_est.alignment = align_center
                    col_idx += 1
                    
                    cell_res = ws_details.cell(row=5, column=col_idx, value=t["xls_details_residual"].format(t["methods_names"][m["name"]]))
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
                ws_coefs = wb.create_sheet(title=t["xls_sheet_coefs"])
                ws_coefs.views.sheetView[0].showGridLines = True
                
                ws_coefs.cell(row=2, column=2, value=t["xls_coefs_title"]).font = title_font
                ws_coefs.cell(row=3, column=2, value=t["xls_coefs_subtitle"]).font = subtitle_font
                
                headers_coefs = t["xls_coefs_headers"]
                for col_idx, header in enumerate(headers_coefs, start=2):
                    cell = ws_coefs.cell(row=5, column=col_idx, value=header)
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = align_center
                    cell.border = thin_border
                    
                coefs_data = []
                coefs_data.append([t["methods_names"]["SLR"], t["xls_param_slope"], m_slr, t["xls_param_intercept"], c_slr, "", ""])
                coefs_data.append([t["methods_names"]["GOR Conv"], t["xls_param_slope"], m_gor, t["xls_param_intercept"], b_gor, "", ""])
                coefs_data.append([t["methods_names"]["GOR Prop"], t["xls_param_slope"], m_prop, t["xls_param_intercept"], b_prop, "", ""])
                if exp_asymp_valid:
                    coefs_data.append([t["methods_names"]["No Lin. Exponencial Asintota"], t["xls_param_asymptote"], a_val, t["xls_param_scale_b"], b_nl_exp, t["xls_param_rate_c"], c_nl_exp])
                if log_valid:
                    coefs_data.append([t["methods_names"]["No Lin. Logarítmico"], t["xls_param_intercept_a"], a_nl_log, t["xls_param_scale_b"], b_nl_log, "", ""])
                if pot_valid:
                    coefs_data.append([t["methods_names"]["No Lin. Potencial / Power Law"], t["xls_param_scale_a"], a_nl_pot, t["xls_param_exponent_b"], b_nl_pot, "", ""])
                if quad_valid:
                    coefs_data.append([t["methods_names"]["No Lin. Cuadrático"], t["xls_param_quad_a"], a_nl_quad, t["xls_param_quad_b"], b_nl_quad, t["xls_param_quad_c"], c_nl_quad])
                if fun_pot_valid:
                    coefs_data.append([t["methods_names"]["Fun. Potencia log10 (Correcto)"], t["xls_param_scale_a"], a_correct, t["xls_param_exponent_b"], b_correct, "", ""])
                    coefs_data.append([t["methods_names"]["Fun. Potencia log10 (Incorrecto)"], f"{t['xls_param_scale_a']} (err)", a_err, f"{t['xls_param_exponent_b']} (err)", b_err, "", ""])
                if fun_exp_valid:
                    coefs_data.append([t["methods_names"]["Fun. Exponencial Simple ln"], t["xls_param_scale_a"], a_coeff, t["xls_param_exponent_b"], b_coeff, "", ""])

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
                    label=t["t8_export_excel_btn"],
                    data=excel_data,
                    file_name="Reporte_Comparacion_Regresiones.xlsx" if st.session_state.lang == 'es' else ("प्रतिगमन_तुलना_रिपोर्ट.xlsx" if st.session_state.lang == 'hi' else "Regression_Comparison_Report.xlsx"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error generando Excel: {e}")
            
        with col_out2:
            st.markdown(f"**{t['t8_export_pdf_lbl']}**")
            
            def create_pdf():
                # Use English translations for PDF if the current language is Hindi to avoid Unicode encoding errors in fpdf2
                t_pdf = TRANSLATIONS["en"] if st.session_state.lang == "hi" else t
                
                # Rebuild comparison dataframe for PDF using t_pdf to avoid language conflicts
                comp_data_pdf = {
                    t_pdf["table_col_method"]: [t_pdf["methods_names"]["SLR"], t_pdf["methods_names"]["GOR Conv"], t_pdf["methods_names"]["GOR Prop"]],
                    t_pdf["table_col_slope"]: [m_slr, m_gor, m_prop],
                    t_pdf["table_col_intercept"]: [c_slr, b_gor, b_prop],
                    t_pdf["table_col_se_m"]: [se_m_slr, se_m_gor, se_m_prop],
                    t_pdf["table_col_se_c"]: [se_c_slr, se_b_gor, se_b_prop],
                    t_pdf["table_col_rmse"]: [rmse_slr, rmse_gor, rmse_prop],
                    t_pdf["table_col_r2"]: [r2_slr, r2_gor, r2_prop]
                }
                if exp_asymp_valid:
                    comp_data_pdf[t_pdf["table_col_method"]].append(t_pdf["methods_names"]["No Lin. Exponencial Asintota"])
                    comp_data_pdf[t_pdf["table_col_slope"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_intercept"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_m"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_c"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_rmse"]].append(rmse_nl_exp)
                    comp_data_pdf[t_pdf["table_col_r2"]].append(r2_nl_exp)

                if log_valid:
                    comp_data_pdf[t_pdf["table_col_method"]].append(t_pdf["methods_names"]["No Lin. Logarítmico"])
                    comp_data_pdf[t_pdf["table_col_slope"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_intercept"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_m"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_c"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_rmse"]].append(rmse_nl_log)
                    comp_data_pdf[t_pdf["table_col_r2"]].append(r2_nl_log)

                if pot_valid:
                    comp_data_pdf[t_pdf["table_col_method"]].append(t_pdf["methods_names"]["No Lin. Potencial / Power Law"])
                    comp_data_pdf[t_pdf["table_col_slope"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_intercept"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_m"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_c"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_rmse"]].append(rmse_nl_pot)
                    comp_data_pdf[t_pdf["table_col_r2"]].append(r2_nl_pot)

                if quad_valid:
                    comp_data_pdf[t_pdf["table_col_method"]].append(t_pdf["methods_names"]["No Lin. Cuadrático"])
                    comp_data_pdf[t_pdf["table_col_slope"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_intercept"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_m"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_c"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_rmse"]].append(rmse_nl_quad)
                    comp_data_pdf[t_pdf["table_col_r2"]].append(r2_nl_quad)

                if fun_pot_valid:
                    comp_data_pdf[t_pdf["table_col_method"]].append(t_pdf["methods_names"]["Fun. Potencia log10 (Correcto)"])
                    comp_data_pdf[t_pdf["table_col_slope"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_intercept"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_m"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_c"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_rmse"]].append(rmse_pot10_correct)
                    comp_data_pdf[t_pdf["table_col_r2"]].append(r2_pot10_correct)

                    comp_data_pdf[t_pdf["table_col_method"]].append(t_pdf["methods_names"]["Fun. Potencia log10 (Incorrecto)"])
                    comp_data_pdf[t_pdf["table_col_slope"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_intercept"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_m"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_c"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_rmse"]].append(rmse_pot10_incorrect)
                    comp_data_pdf[t_pdf["table_col_r2"]].append(r2_pot10_incorrect)

                if fun_exp_valid:
                    comp_data_pdf[t_pdf["table_col_method"]].append(t_pdf["methods_names"]["Fun. Exponencial Simple ln"])
                    comp_data_pdf[t_pdf["table_col_slope"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_intercept"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_m"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_se_c"]].append(np.nan)
                    comp_data_pdf[t_pdf["table_col_rmse"]].append(rmse_exp_simple)
                    comp_data_pdf[t_pdf["table_col_r2"]].append(r2_exp_simple)

                df_comp_pdf = pd.DataFrame(comp_data_pdf)

                # Crear imagen de Matplotlib estática en background para el PDF
                fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5))
                ax_pdf.scatter(X, Y, color='black', label=t_pdf["chart_obs"])
                x_vals = np.linspace(min(X), max(X), 100)
                ax_pdf.plot(x_vals, m_slr * x_vals + c_slr, color='blue', label=t_pdf["chart_slr"])
                ax_pdf.plot(x_vals, m_gor * x_vals + b_gor, color='orange', label=t_pdf["chart_gor_conv"])
                ax_pdf.plot(x_vals, m_prop * x_vals + b_prop, color='green', label=t_pdf["chart_gor_prop"])
                
                if nl_valid:
                    if selected_nl_key == "Exponential":
                        y_vals_nl = a_nl + b_nl * np.exp(c_nl * x_vals)
                    elif selected_nl_key == "Logarithmic":
                        x_vals_safe = np.maximum(x_vals, 1e-9)
                        y_vals_nl = a_nl + b_nl * np.log(x_vals_safe)
                    elif selected_nl_key == "Power":
                        x_vals_safe = np.maximum(x_vals, 1e-9)
                        y_vals_nl = a_nl * (x_vals_safe ** b_nl)
                    elif selected_nl_key == "Quadratic":
                        y_vals_nl = a_nl * (x_vals ** 2) + b_nl * x_vals + c_nl
                    
                    nl_short_names_pdf = {
                        "Exponential": t_pdf["nl_models"]["exp"].split(' ')[0],
                        "Logarithmic": t_pdf["nl_models"]["log"].split(' ')[0],
                        "Power": t_pdf["nl_models"]["pot"].split(' ')[0],
                        "Quadratic": t_pdf["nl_models"]["quad"].split(' ')[0]
                    }
                    selected_nl_short_pdf = nl_short_names_pdf[selected_nl_key]
                    ax_pdf.plot(x_vals, y_vals_nl, color='purple', linestyle='-.', label=t_pdf["chart_nlin"].format(selected_nl_short_pdf))
                
                if power_valid:
                    x_vals_safe = np.maximum(x_vals, 1e-9)
                    y_vals_power = a_power * (x_vals_safe ** b_power)
                    ax_pdf.plot(x_vals, y_vals_power, color='coral', linestyle='--', label=t_pdf["chart_power"])

                ax_pdf.legend()
                ax_pdf.set_title(t_pdf["pdf_chart_title"])
                
                img_buffer = io.BytesIO()
                fig_pdf.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                plt.close(fig_pdf)
                
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, t_pdf["pdf_title"], 0, 1, 'C')
                pdf.ln(5)
                
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(img_buffer.getvalue())
                    tmp_path = tmp_file.name
                pdf.image(tmp_path, x=15, w=180)
                os.unlink(tmp_path)
                
                pdf.ln(5)
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, t_pdf["pdf_sec_summary"], 0, 1)
                pdf.set_font('Arial', '', 10)
                
                for i, row in df_comp_pdf.iterrows():
                    method_name = row[t_pdf["table_col_method"]]
                    if method_name == t_pdf["methods_names"]["SLR"]:
                        eq = f"y = {m_slr:.4f}x + {c_slr:.4f}"
                    elif method_name == t_pdf["methods_names"]["GOR Conv"]:
                        eq = f"y = {m_gor:.4f}x + {b_gor:.4f}"
                    elif method_name == t_pdf["methods_names"]["GOR Prop"]:
                        eq = f"y = {m_prop:.4f}x + {b_prop:.4f}"
                    elif method_name == t_pdf["methods_names"]["No Lin. Exponencial Asintota"]:
                        eq = nl_equation_exp
                    elif method_name == t_pdf["methods_names"]["No Lin. Logarítmico"]:
                        eq = nl_equation_log
                    elif method_name == t_pdf["methods_names"]["No Lin. Potencial / Power Law"]:
                        eq = nl_equation_pot
                    elif method_name == t_pdf["methods_names"]["No Lin. Cuadrático"]:
                        eq = nl_equation_quad
                    elif method_name == t_pdf["methods_names"]["Fun. Potencia log10 (Correcto)"]:
                        eq = f"y = {a_correct:.4f} * x^{{{b_correct:.4f}}}"
                    elif method_name == t_pdf["methods_names"]["Fun. Potencia log10 (Incorrecto)"]:
                        eq = f"y = {a_err:.4f} * x^{{{b_err:.4f}}}"
                    elif method_name == t_pdf["methods_names"]["Fun. Exponencial Simple ln"]:
                        eq = f"y = {a_coeff:.4f} * e^{{{b_coeff:.4f}x}}"
                    else:
                        eq = ""
                    clean_eq = eq.replace('\\cdot', '*').replace('\\ln', 'ln').replace('\\bar', '').replace('\\hat', '').replace('{', '').replace('}', '')
                    pdf.cell(0, 6, f"{method_name}: {clean_eq} | RMSE: {row[t_pdf['table_col_rmse']]:.4f} | R2: {row[t_pdf['table_col_r2']]:.4f}", 0, 1)
                
                return bytes(pdf.output())

            try:
                pdf_data = create_pdf()
                st.download_button(
                    label=t["t8_export_pdf_btn"],
                    data=pdf_data,
                    file_name="Reporte_Comparacion_Regresiones.pdf" if st.session_state.lang == 'es' else ("प्रतिगमन_तुलना_रिपोर्ट.pdf" if st.session_state.lang == 'hi' else "Regression_Comparison_Report.pdf"),
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(t["pdf_err_msg"].format(e))



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
