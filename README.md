# 📈 Análisis de Regresión Lineal: Comparación de Métodos / Linear Regression Analysis: Method Comparison

Selecciona tu idioma / Select your language:
*   [🇪🇸 Español](#-español)
*   [🇺🇸 English](#-english)

---

## 🇪🇸 Español

¡Bienvenido a la aplicación de **Comparación de Métodos de Regresión Lineal**! 

Esta es una herramienta interactiva construida con Python y [Streamlit](https://streamlit.io/) que te permite automatizar el análisis, proyectar y comparar simultáneamente múltiples métodos de regresión matemática. 

---

### 🚀 Características Principales

*   📊 **Entrada Única de Datos**:
    *   **Subida de Archivos**: Sube tus datos directamente en formatos `.csv` o `.xlsx`.
    *   **Ingreso Manual**: Usa un editor interactivo para ingresar o corregir tus puntos (X, Y) manualmente. Los tres modelos procesan esta misma base para sus cálculos simultáneamente.
*   🧮 **Comparación Simultánea de 3 Modelos**: 
    1.  **LSR (Mínimos Cuadrados Ordinarios):** Minimizando errores verticales.
    2.  **GOR Convencional:** Regresión Ortogonal Generalizada utilizando el procedimiento de *Das et al. (2018)*.
    3.  **GOR Propuesto (Insesgado):** Implementando una corrección final mediante proyecciones ortogonales.
*   📉 **Dashboard Interactivo (Plotly)**: Gráficos dinámicos donde puedes visualizar las tres líneas de tendencia y explorar individualmente cada predicción o residuo simplemente pasando el ratón por encima de los puntos.
*   🗂️ **Interfaz Organizada por Pestañas**:
    *   *Análisis Comparativo:* Gráficas dinámicas y una tabla de métricas clave (Pendiente, Intercepción, Error Estándar y $R^2$).
    *   *Datos y Proyecciones:* Tabla que detalla los residuos calculados y las proyecciones ortogonales verdaderas ($X_t, Y_t$).
    *   *Explicación de Fórmulas:* Visualización de todas las matemáticas y fórmulas subyacentes desarrolladas en LaTeX.
*   📄 **Exportación de Reportes**:
    *   **Reportes PDF**: Genera un informe ejecutivo que incluye una renderización estática en alta calidad de la gráfica y un resumen detallado de las métricas.
    *   **Reportes Excel**: Descarga tus predicciones, parámetros de los tres métodos y datos proyectados en un documento estructurado `.xlsx`.

---

### 🛠️ Tecnologías y Librerías

*   **Python:** Lenguaje base de desarrollo.
*   **Streamlit:** Framework para el desarrollo ágil de la interfaz web (UI).
*   **Plotly:** Renderizado de gráficos interactivos, tooltips dinámicos y métricas visuales.
*   **Pandas & NumPy:** Manipulación de estructuras de datos y cálculo numérico complejo.
*   **Scikit-Learn:** Algoritmos óptimos subyacentes para el cálculo de la Regresión Lineal Simple.
*   **Matplotlib:** Renderizado en segundo plano para la exportación optimizada de gráficos en PDF.
*   **fpdf:** Generación estructural de los documentos PDF de reporte.

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

## 🇺🇸 English

Welcome to the **Linear Regression Method Comparison** application! 

This is an interactive tool built with Python and [Streamlit](https://streamlit.io/) that allows you to automate analysis, project, and simultaneously compare multiple mathematical regression methods. 

---

### 🚀 Key Features

*   📊 **Single Data Input**:
    *   **File Upload**: Upload your data directly in `.csv` or `.xlsx` formats.
    *   **Manual Entry**: Use an interactive editor to enter or correct your (X, Y) points manually. All three models process this same base for their calculations simultaneously.
*   🧮 **Simultaneous 3-Model Comparison**: 
    1.  **LSR (Ordinary Least Squares):** Minimizing vertical errors.
    2.  **Conventional GOR:** Generalized Orthogonal Regression using the *Das et al. (2018)* procedure.
    3.  **Proposed GOR (Unbiased):** Implementing a final correction using orthogonal projections.
*   📉 **Interactive Dashboard (Plotly)**: Dynamic charts where you can visualize the three trend lines and individually explore each prediction or residue simply by hovering over the points.
*   🗂️ **Tab-Organized Interface**:
    *   *Comparative Analysis:* Dynamic charts and a table of key metrics (Slope, Intercept, Standard Error, and $R^2$).
    *   *Data and Projections:* Table detailing calculated residuals and true orthogonal projections ($X_t, Y_t$).
    *   *Formula Explanation:* Visualization of all underlying mathematics and formulas developed in LaTeX.
*   📄 **Report Exporting**:
    *   **PDF Reports**: Generates an executive report that includes a high-quality static rendering of the chart and a detailed summary of metrics.
    *   **Excel Reports**: Download your predictions, parameters of the three methods, and projected data in a structured `.xlsx` document.

---

### 🛠️ Technologies and Libraries

*   **Python:** Core development language.
*   **Streamlit:** Framework for rapid web interface (UI) development.
*   **Plotly:** Rendering of interactive charts, dynamic tooltips, and visual metrics.
*   **Pandas & NumPy:** Data structure manipulation and complex numerical calculation.
*   **Scikit-Learn:** Underlying optimal algorithms for Simple Linear Regression calculation.
*   **Matplotlib:** Background rendering for optimized PDF chart export.
*   **fpdf:** Structural generation of PDF report documents.

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
