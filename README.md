# 📈 Análisis de Regresión Lineal y No Lineal: Comparación de Métodos / Linear and Non-Linear Regression Analysis: Method Comparison

Selecciona tu idioma / Select your language:
*   [🇪🇸 Español](#-español)
*   [🇺🇸 English](#-english)

---

## 🇪🇸 Español

¡Bienvenido a la plataforma interactiva de **Comparación de Métodos de Regresión Lineal y No Lineal**! 

Esta es una herramienta educativa y profesional construida con Python y [Streamlit](https://streamlit.io/) que te permite automatizar, proyectar y comparar simultáneamente múltiples métodos de regresión matemática, analizando críticamente el comportamiento de los modelos frente a hojas de cálculo tradicionales.

---

### 🚀 Características Principales

*   🌐 **Selector de Idioma Interactivo (Bilingüe)**:
    *   Cambio de inmediato entre **Español** e **Inglés** desde la barra lateral.
    *   Traducción completa en tiempo real de toda la interfaz (explicaciones LaTeX, sumatorias, calculadoras, simulaciones y gráficos interactivos).
    *   Reportes exportados (PDF y Excel) adaptados 100% al idioma activo.
*   📈 **Módulos de Análisis Disponibles**:
    1.  **Regresión Simple (SLR / GOR / No Lineal)**:
        *   **LSR (Mínimos Cuadrados Ordinarios):** Minimizando errores verticales en el eje Y (asume variable independiente sin error).
        *   **GOR Convencional:** Regresión Ortogonal Generalizada utilizando el procedimiento de *Das et al. (2018)*, minimizando las distancias perpendiculares ponderadas por una relación de varianzas $\eta$.
        *   **GOR Propuesto (Insesgado):** Implementación del modelo linealizado insesgado propuesto por *Ranjit Das et al.* que optimiza el ajuste final a partir de las proyecciones ortogonales verdaderas.
        *   **MoM (Método de Momentos):** Ajuste linealizado que mitiga el sesgo de atenuación.
        *   **Regresión No Lineal:** Ajustes por linealización que incluyen modelos **Exponencial** ($Y = a + b \cdot e^{cx}$), **Logarítmico** ($Y = a + b \cdot \ln(x)$), **Potencial** ($Y = a \cdot x^b$) y **Cuadrático** ($Y = ax^2 + bx + c$).
        *   **Power Law Model (Modelo de Ley de Potencia):** Ajuste del tipo $Y = a \cdot X^b$ mediante transformación logarítmica dual con análisis de aplicabilidad en fenómenos de escala (como la Ley de Omori o Gutenberg-Richter en sismología).
        *   **Función Potencia (Educativa):** Módulo especializado que ajusta $y = a \cdot x^b$ usando $\log_{10}$ y detalla de forma didáctica la diferencia entre el **despeje matemático correcto** ($b = m$) frente al **despeje incorrecto común** ($b = 10^m$).
        *   **Función Exponencial (Educativa):** Módulo enfocado en modelar $y = a \cdot e^{bx}$ aplicando el logaritmo natural ($\ln$) sobre la variable dependiente.
    2.  **Módulo de Regresión Lineal Múltiple (MLR)**:
        *   Ajusta y analiza la relación entre una variable dependiente ($Y$) y múltiples variables independientes ($X_1, X_2, \dots, X_k$).
        *   Preferencia predeterminada por un conjunto sismológico de predicción de movimiento del suelo (GMPE: `logPOA`, `M`, `M^2`, `logR`, `R`).
        *   Estadísticas avanzadas idénticas a las de Excel: Coeficiente de correlación múltiple, $R^2$, $R^2$ ajustado, Error típico, tabla ANOVA completa y coeficientes con errores estándar, estadísticos t, P-valores e intervalos de confianza del 95%.
        *   Visualizaciones dinámicas en pestañas separadas para estadísticas y ANOVA, fórmulas matemáticas en LaTeX, análisis de los residuales, resultados de datos de probabilidad, gráfico de probabilidad normal y gráficos de residuales y ajuste.
    3.  **Distribuciones de Probabilidad en el Modelado Matemático**:
        *   Visualiza, explora y calcula probabilidades interactivamente para distribuciones discretas (Binomial, Poisson) y continuas (Normal, Lognormal, Exponencial) utilizadas frecuentemente en la modelación científica y sismológica.
    4.  **Simulación de Monte Carlo (Regresión mb vs Mw)**:
        *   Evalúa el efecto de los errores de medición agregando ruido gaussiano configurable a magnitudes sísmicas reales, comparando visualmente los ajustes OLS y GOR sobre los datos sintéticos contaminados.
    5.  **Comparación Carroll & Ruppert (GOR con Error de Ecuación)**:
        *   Implementación de las ecuaciones de Carroll & Ruppert (1996) y Ranjit Das para analizar el comportamiento de las pendientes de regresión bajo heterocedasticidad y diversas incertidumbres asociadas a variables con error de medición y error de ecuación.
    6.  **Cadenas de Markov (Modelos Estocásticos) 🎲**:
        *   Modelado y simulación de procesos estocásticos en tiempo discreto.
        *   Configuración dinámica de los estados y la Matriz de Probabilidades de Transición (TPM) con opción de normalización rápida de filas.
        *   Pestañas para visualización de la **Evolución Temporal**, cálculo exacto del **Estado Estacionario** ($\pi P = \pi$) y **Simulación Monte Carlo de Caminata Aleatoria** (Random Walk) comparando la distribución empírica vs. la teórica.
    7.  **Regresión Sísmica y el Rol del 'Error de Ecuación' (BSSA 2025/2026) 📊**:
        *   Réplica exacta de los análisis y figuras del artículo científico *'The Role of "Equation Error" in Empirical Regressions for Seismic Magnitude Conversions'* (Gasperini et al., 2025/2026).
        *   Compara la pendiente ajustada ($\beta_1$) en función del error supuesto de la variable independiente ($\sigma_x$) usando métodos OLS, MM y EIV.
        *   Catálogos precargados: Nueva Zelanda 2004-2011, Nueva Zelanda 2012-2020, Italia 2005-2023, Chile 2010-2024, y opción de entrada personalizada.
*   📊 **Entrada Única de Datos**:
    *   **Subida de Archivos**: Sube tus datos directamente en formatos `.csv` o `.xlsx`.
    *   **Ingreso Manual**: Usa un editor interactivo para ingresar o corregir tus puntos (X, Y) dinámicamente. Todos los modelos procesan esta misma base simultáneamente.
*   📉 **Dashboard Interactivo y Visual (Plotly)**:
    *   Gráficos dinámicos interactivos con tooltips y zoom en el espacio original de datos.
    *   Gráficos adicionales en espacios linealizados para observar la calidad del ajuste de las transformaciones matemáticas.
*   🗂️ **Interfaz de Regresión Simple Organizada en 9 Pestañas**:
    *   *1. SLR (Mínimos Cuadrados):* Concepto, fórmulas detalladas en LaTeX y cálculo de errores estándar.
    *   *2. GOR Convencional:* Parámetro de relación de varianzas ($\eta$), proyecciones ortogonales ($X_t, Y_t$) y desviación típica residual ortogonal.
    *   *3. GOR Propuesto:* Innovación matemática de Ranjit Das para corrección de sesgos en el ajuste.
    *   *4. MoM (Atenuación):* Método de Momentos para mitigar el sesgo de atenuación en la pendiente de regresión.
    *   *5. Regresión No Lineal:* Análisis de transformaciones, tabla de sumatorias, calculadora de predicciones interactiva y comparación métrica contra SLR.
    *   *6. Power Law Model:* Leyes de potencia aplicadas a la sismología y ciencias naturales con su calculadora interactiva.
    *   *7. Función Potencia:* Comparación gráfica en el espacio original de la curva correcta vs. la curva con error de despeje (curva explosiva).
    *   *8. Función Exponencial:* Ecuaciones de decaimiento y crecimiento linealizadas bajo base natural.
    *   *9. Datos y Exportación:* Tabla de residuos unificada y controles de descarga.
*   📄 **Exportación de Reportes**:
    *   **Reportes PDF**: Genera un reporte formal ejecutivo que incluye un gráfico estático consolidado (generado con Matplotlib) y un resumen analítico con las métricas y ecuaciones de todos los modelos ajustados.
    *   **Reportes Excel**: Descarga tus predicciones, parámetros, residuos y datos proyectados para todos los modelos en un documento estructurado `.xlsx`.

---

### 🛠️ Tecnologías y Librerías

*   **Python:** Lenguaje base de desarrollo.
*   **Streamlit:** Framework para el desarrollo de la interfaz de usuario web interactiva.
*   **Plotly:** Renderizado de gráficos web interactivos con soporte tooltip dinámico.
*   **Pandas & NumPy:** Manipulación de estructuras de datos y cálculo numérico matricial.
*   **Scikit-Learn:** Ajustes matemáticos de regresiones lineales base.
*   **Matplotlib & Seaborn:** Generación de gráficos estáticos optimizados para exportación.
*   **fpdf2:** Generación estructurada de documentos PDF de reporte.
*   **openpyxl:** Motor de escritura de hojas de cálculo Excel.

---

### 💻 Instalación y Uso Local

Para correr este proyecto de manera local, asegúrate de tener [Python 3](https://www.python.org/downloads/) y [Git](https://git-scm.com/) instalados. Sigue estos pasos dependiendo de tu sistema operativo:

> [!TIP]
> **¿No tienes Git instalado?** 
> No te preocupes, puedes descargar el código fuente directamente en formato ZIP haciendo clic en el botón verde **Code** (en la esquina superior derecha de esta página en GitHub) y seleccionando **Download ZIP**, o bien a través de [este enlace de descarga directa](https://github.com/j-alexander-acosta/Regresiones_Lineales/archive/refs/heads/main.zip). Una vez descargado, descomprime el archivo `.zip` en tu computadora, abre la terminal/consola directamente dentro de esa carpeta y continúa con las instrucciones a partir del **Paso 2** (creación del entorno virtual).

#### 🍎 En macOS y 🐧 Linux

1. **Clona el Repositorio**:
    ```bash
    git clone https://github.com/j-alexander-acosta/Regresiones_Lineales.git
    cd Regresiones_Lineales
    ```

2. **Crea y activa un entorno virtual (Recomendado)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Instala las dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Ejecuta el Servidor de Streamlit**:
    ```bash
    streamlit run app.py
    ```

#### 🪟 En Windows

1. **Clona el Repositorio**:
    Abre tu terminal (Símbolo del sistema o PowerShell) y ejecuta:
    ```cmd
    git clone https://github.com/j-alexander-acosta/Regresiones_Lineales.git
    cd Regresiones_Lineales
    ```

2. **Crea y activa un entorno virtual (Recomendado)**:
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```
    *(Nota: Si recibes un error de ejecución de scripts en PowerShell, ejecuta primero `Set-ExecutionPolicy Unrestricted -Scope CurrentUser`)*

3. **Instala las dependencias**:
    ```cmd
    pip install -r requirements.txt
    ```

4. **Ejecuta el Servidor de Streamlit**:
    ```cmd
    streamlit run app.py
    ```

---

#### 🚀 Comienza a trabajar
Automáticamente se abrirá tu navegador web en la dirección local `http://localhost:8501`. Si no se abre por sí solo, puedes acceder manualmente a ese enlace.

---

### 👨‍💻 Acerca del Autor

Desarrollado y mantenido por **Alexander Acosta** ([@j-alexander-acosta](https://github.com/j-alexander-acosta)).

---
---

## 🇺🇸 English

Welcome to the interactive **Linear and Non-Linear Regression Method Comparison** platform!

This is an educational and professional tool built with Python and [Streamlit](https://streamlit.io/) that allows you to automate, project, and simultaneously compare multiple mathematical regression methods, critically analyzing model behaviors against traditional spreadsheets.

---

### 🚀 Key Features

*   🌐 **Interactive Language Selector (Bilingual)**:
    *   Switch instantly between **Spanish** and **English** using buttons in the sidebar.
    *   Real-time translation of the entire interface (LaTeX math explanations, intermediate sums, calculators, simulations, and interactive plots).
    *   Exported reports (PDF and Excel) fully adapted to the selected language.
*   📈 **Available Analysis Modules**:
    1.  **Simple Regression (SLR / GOR / Non-Linear)**:
        *   **LSR (Ordinary Least Squares):** Minimizing vertical errors on the Y-axis (assumes independent variable has no measurement error).
        *   **Conventional GOR:** Generalized Orthogonal Regression using the *Das et al. (2018)* procedure, minimizing perpendicular distances weighted by a variance ratio $\eta$.
        *   **Proposed GOR (Unbiased):** Implementation of the unbiased linearized model proposed by *Ranjit Das et al.* which optimizes the final fit based on true orthogonal projections.
        *   **MoM (Method of Moments):** Linearized fit that mitigates attenuation bias.
        *   **Non-Linear Regression:** Fitted using linearization techniques, including **Exponential** ($Y = a + b \cdot e^{cx}$), **Logarithmic** ($Y = a + b \cdot \ln(x)$), **Power** ($Y = a \cdot x^b$), and **Quadratic** ($Y = ax^2 + bx + c$) models.
        *   **Power Law Model:** Fitted as $Y = a \cdot X^b$ via dual logarithmic transformation with analysis of scaling applicability in natural sciences and seismology (such as Omori's Law or Gutenberg-Richter Law).
        *   **Power Function (Educational):** Specialized module that fits $y = a \cdot x^b$ using $\log_{10}$ and teaches the difference between the **correct mathematical parameters solving** ($b = m$) vs. the **common incorrect spreadsheet solving** ($b = 10^m$).
        *   **Exponential Function (Educational):** Dedicated module focused on modeling $y = a \cdot e^{bx}$ applying the natural logarithm ($\ln$) on the dependent variable.
    2.  **Multiple Linear Regression (MLR) Module**:
        *   Fit and analyze the relationship between a dependent variable ($Y$) and multiple independent variables ($X_1, X_2, \dots, X_k$).
        *   Preloaded with a seismological Ground Motion Prediction Equation (GMPE) dataset (`logPOA`, `M`, `M^2`, `logR`, `R`).
        *   Calculates Excel-matching regression statistics (Multiple R, $R^2$, Adjusted $R^2$, Standard Error, complete ANOVA table, coefficient estimates, standard errors, t stats, P-values, and 95% Confidence Intervals).
        *   Interactive separate tabs for statistics & ANOVA, mathematical formulas in LaTeX, residual analysis, probability output, normal probability plot, and residual & fit plots.
    3.  **Probability Distributions in Mathematical Modeling**:
        *   Visualize, explore, and calculate probabilities interactively for discrete (Binomial, Poisson) and continuous (Normal, Lognormal, Exponential) distributions commonly used in scientific and seismological modeling.
    4.  **Monte Carlo Simulation (mb vs Mw Regression)**:
        *   Evaluate the effect of measurement errors by adding configurable Gaussian noise to true seismic magnitudes, visually comparing OLS and GOR fits on the contaminated synthetic data.
    5.  **Carroll & Ruppert Comparison (GOR with Equation Error)**:
        *   Implementation of the equations from Carroll & Ruppert (1996) and Ranjit Das to analyze regression slope behavior under heteroscedasticity and various uncertainties associated with variables with measurement errors and equation errors.
    6.  **Markov Chains (Stochastic Models) 🎲**:
        *   Modeling and simulation of discrete-time stochastic processes.
        *   Dynamic configuration of state names and the Transition Probability Matrix (TPM) with a quick row normalization feature.
        *   Tabs for **Temporal Evolution** visualization, exact **Steady State** calculation ($\pi P = \pi$), and **Monte Carlo Random Walk Simulation** comparing empirical vs. theoretical distributions.
    7.  **Seismic Regression and the Role of 'Equation Error' (BSSA 2025/2026) 📊**:
        *   Exact replication of the analyses and figures from the scientific paper *'The Role of "Equation Error" in Empirical Regressions for Seismic Magnitude Conversions'* (Gasperini et al., 2025/2026).
        *   Compares the variation of the fitted slope ($\beta_1$) as a function of the assumed independent variable error ($\sigma_x$) using OLS, MM, and EIV methods.
        *   Preloaded catalogs: New Zealand 2004-2011, New Zealand 2012-2020, Italy 2005-2023, Chile 2010-2024, and custom manual coordinate entry.
*   📊 **Single Data Input**:
    *   **File Upload**: Upload your data directly in `.csv` or `.xlsx` formats.
    *   **Manual Entry**: Use an interactive editor to dynamically input or correct your (X, Y) points. All models process this same base simultaneously.
*   📉 **Interactive Dashboard (Plotly)**:
    *   Dynamic interactive charts with tooltips and zoom capabilities in the original data space.
    *   Additional charts in linearized spaces to observe the fit quality of mathematical transformations.
*   🗂️ **Tab-Organized Simple Regression Interface (9 Tabs)**:
    *   *1. SLR (Least Squares):* Concept, detailed LaTeX formulas, and coefficient standard error calculations.
    *   *2. GOR Conventional:* Variance ratio parameter ($\eta$), orthogonal projections ($X_t, Y_t$), and typical residual orthogonal deviation.
    *   *3. GOR Proposed:* Ranjit Das's mathematical innovation for bias correction.
    *   *4. MoM (Attenuation):* Method of Moments to mitigate attenuation bias in the regression slope.
    *   *5. Non-Linear Regression:* Transformation analysis, sum tables, interactive prediction calculator, and comparison metrics against SLR.
    *   *6. Power Law Model:* Power scaling laws applied to seismology and physics with an interactive prediction calculator.
    *   *7. Power Function:* Visual comparison in the original space of the correct curve vs. the curve with parameter solving errors (exponentially explosive curve).
    *   *8. Exponential Function:* Decay and growth equations linearized under natural base.
    *   *9. Data & Export:* Unified residuals table and download controls.
*   📄 **Report Exporting**:
    *   **PDF Reports**: Generates a formal executive report including a consolidated static chart (built using Matplotlib) and an analytical summary of metrics and equations for all fitted models.
    *   **Excel Reports**: Download predictions, parameters, residuals, and projected data for all models in a structured `.xlsx` document.

---

### 🛠️ Technologies and Libraries

*   **Python:** Core development language.
*   **Streamlit:** Web framework for interactive user interface development.
*   **Plotly:** Interactive web charts with dynamic tooltip support.
*   **Pandas & NumPy:** Data structures and matrix mathematical calculations.
*   **Scikit-Learn:** Core simple linear regression fittings.
*   **Matplotlib & Seaborn:** Optimized static chart generation for reporting.
*   **fpdf2:** Structural generation of PDF report documents.
*   **openpyxl:** Excel spreadsheet writing engine.

---

### 💻 Installation and Local Usage

To run this project locally, make sure you have [Python 3](https://www.python.org/downloads/) and [Git](https://git-scm.com/) installed. Follow these steps depending on your operating system:

> [!TIP]
> **Don't have Git installed?** 
> Don't worry, you can download the source code directly in ZIP format by clicking the green **Code** button (in the upper-right corner of this page on GitHub) and selecting **Download ZIP**, or via [this direct download link](https://github.com/j-alexander-acosta/Regresiones_Lineales/archive/refs/heads/main.zip). Once downloaded, extract the `.zip` file on your computer, open the terminal/console directly inside that folder, and continue with the instructions from **Step 2** (virtual environment creation).

#### 🍎 On macOS and 🐧 Linux

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/j-alexander-acosta/Regresiones_Lineales.git
    cd Regresiones_Lineales
    ```

2. **Create and activate a virtual environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit Server**:
    ```bash
    streamlit run app.py
    ```

#### 🪟 On Windows

1. **Clone the Repository**:
    Open your terminal (Command Prompt or PowerShell) and run:
    ```cmd
    git clone https://github.com/j-alexander-acosta/Regresiones_Lineales.git
    cd Regresiones_Lineales
    ```

2. **Create and activate a virtual environment (Recommended)**:
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```
    *(Note: If you receive a script execution error in PowerShell, first run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser`)*

3. **Install dependencies**:
    ```cmd
    pip install -r requirements.txt
    ```

4. **Run the Streamlit Server**:
    ```cmd
    streamlit run app.py
    ```

---

#### 🚀 Start working
Your web browser will automatically open at the local address `http://localhost:8501`. If it doesn't open on its own, you can manually access that link.

---

### 👨‍💻 About the Author

Developed and maintained by **Alexander Acosta** ([@j-alexander-acosta](https://github.com/j-alexander-acosta)).

---
