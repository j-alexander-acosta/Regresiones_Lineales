# 📈 Análisis de Regresión Lineal y No Lineal: Comparación de Métodos / Linear and Non-Linear Regression Analysis: Method Comparison

Selecciona tu idioma / Select your language:
*   [🇪🇸 Español](#-español)
*   [🇺🇸 English](#-english)
*   [🇮🇳 हिन्दी](#-हिन्दी)

---

## 🇪🇸 Español

¡Bienvenido a la plataforma interactiva de **Comparación de Métodos de Regresión Lineal y No Lineal**! 

Esta es una herramienta educativa y profesional construida con Python y [Streamlit](https://streamlit.io/) que te permite automatizar, proyectar y comparar simultáneamente múltiples métodos de regresión matemática, analizando críticamente el comportamiento de los modelos frente a hojas de cálculo tradicionales.

---

### 🚀 Características Principales

*   🌐 **Selector de Idioma Interactivo (Trilingüe)**:
    *   Cambio de inmediato entre **Español**, **Inglés** y **Hindi** desde la barra lateral.
    *   Traducción completa en tiempo real de toda la interfaz (explicaciones LaTeX, sumatorias, calculadoras y gráficos interactivos).
    *   Reportes exportados (PDF y Excel) adaptados 100% al idioma activo.
*   📊 **Módulo de Regresión Lineal Múltiple (MLR)**:
    *   Ajusta y analiza la relación entre una variable dependiente ($Y$) y múltiples variables independientes ($X_1, X_2, \dots, X_k$).
    *   Preferencia predeterminada por un conjunto sismológico de predicción de movimiento del suelo (GMPE: `logPOA`, `M`, `M^2`, `logR`, `R`).
    *   Estadísticas avanzadas idénticas a las de Excel: Coeficiente de correlación múltiple, $R^2$, $R^2$ ajustado, Error típico, tabla ANOVA completa y coeficientes con errores estándar, estadísticos t, P-valores e intervalos de confianza del 95%.
    *   Visualizaciones dinámicas en pestañas separadas para gráficos de residuales individuales, gráficos de curvas de ajuste y el gráfico de probabilidad normal.
*   📊 **Entrada Única de Datos**:
    *   **Subida de Archivos**: Sube tus datos directamente en formatos `.csv` o `.xlsx`.
    *   **Ingreso Manual**: Usa un editor interactivo para ingresar o corregir tus puntos (X, Y) dinámicamente. Todos los modelos procesan esta misma base simultáneamente.
*   🧮 **Comparación Simultánea de Múltiples Modelos**: 
    1.  **LSR (Mínimos Cuadrados Ordinarios):** Minimizando errores verticales en el eje Y.
    2.  **GOR Convencional:** Regresión Ortogonal Generalizada utilizando el procedimiento de *Das et al. (2018)*, minimizando las distancias perpendiculares ponderadas por una relación de varianzas $\eta$.
    3.  **GOR Propuesto (Insesgado):** Implementación del modelo linealizado insesgado propuesto por *Ranjit Das et al.* que optimiza el ajuste final a partir de las proyecciones ortogonales verdaderas.
    4.  **Regresión No Lineal:** Ajustes por linealización que incluyen modelos **Exponencial** ($Y = a + b \cdot e^{cx}$), **Logarítmico** ($Y = a + b \cdot \ln(x)$), **Potencial** ($Y = a \cdot x^b$) y **Cuadrático** ($Y = ax^2 + bx + c$).
    5.  **Power Law Model (Modelo de Ley de Potencia):** Ajuste del tipo $Y = a \cdot X^b$ mediante transformación logarítmica dual con análisis de aplicabilidad en fenómenos de escala (como la Ley de Omori o Gutenberg-Richter en sismología).
    6.  **Función Potencia (Educativa):** Módulo especializado que ajusta $y = a \cdot x^b$ usando $\log_{10}$ y detalla de forma didáctica la diferencia entre el **despeje matemático correcto** ($b = m$) frente al **despeje incorrecto común** ($b = 10^m$).
    7.  **Función Exponencial (Educativa):** Módulo enfocado en modelar $y = a \cdot e^{bx}$ aplicando el logaritmo natural ($\ln$) sobre la variable dependiente.
*   📉 **Dashboard Interactivo y Visual (Plotly)**:
    *   Gráficos dinámicos interactivos con tooltips y zoom en el espacio original de datos.
    *   Gráficos adicionales en espacios linealizados para observar la calidad del ajuste de las transformaciones matemáticas.
*   🗂️ **Interfaz Organizada por 8 Pestañas**:
    *   *1. SLR (Mínimos Cuadrados):* Concepto, fórmulas detalladas en LaTeX y cálculo de errores estándar.
    *   *2. GOR Convencional:* Parámetro de relación de varianzas ($\eta$), proyecciones ortogonales ($X_t, Y_t$) y desviación típica residual ortogonal.
    *   *3. GOR Propuesto:* Innovación matemática de Ranjit Das para corrección de sesgos en el ajuste.
    *   *4. Regresión No Lineal:* Análisis de transformaciones, tabla de sumatorias, calculadora de predicciones interactiva y comparación métrica contra SLR.
    *   *5. Power Law Model:* Leyes de potencia aplicadas a la sismología y ciencias naturales con su calculadora interactiva.
    *   *6. Función Potencia:* Comparación gráfica en el espacio original de la curva correcta vs. la curva con error de despeje (curva explosiva).
    *   *7. Función Exponencial:* Ecuaciones de decaimiento y crecimiento linealizadas bajo base natural.
    *   *8. Datos y Exportación:* Tabla de residuos unificada y controles de descarga.
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

*   🌐 **Interactive Language Selector (Trilingual)**:
    *   Switch instantly between **Spanish**, **English**, and **Hindi** using buttons in the sidebar.
    *   Real-time translation of the entire interface (LaTeX math explanations, intermediate sums, calculators, and interactive plots).
    *   Exported reports (PDF and Excel) fully adapted to the selected language.
*   📊 **Multiple Linear Regression (MLR) Module**:
    *   Fit and analyze the relationship between a dependent variable ($Y$) and multiple independent variables ($X_1, X_2, \dots, X_k$).
    *   Preloaded with a seismological Ground Motion Prediction Equation (GMPE) dataset (`logPOA`, `M`, `M^2`, `logR`, `R`).
    *   Calculates Excel-matching regression statistics (Multiple R, $R^2$, Adjusted $R^2$, Standard Error, complete ANOVA table, coefficient estimates, standard errors, t stats, P-values, and 95% Confidence Intervals).
    *   Interactive separate tabs for individual residual plots, line fit plots, and the normal probability plot.
*   📊 **Single Data Input**:
    *   **File Upload**: Upload your data directly in `.csv` or `.xlsx` formats.
    *   **Manual Entry**: Use an interactive editor to dynamically input or correct your (X, Y) points. All models process this same base simultaneously.
*   🧮 **Simultaneous Multiple Model Comparison**: 
    1.  **LSR (Ordinary Least Squares):** Minimizing vertical errors on the Y-axis.
    2.  **Conventional GOR:** Generalized Orthogonal Regression using the *Das et al. (2018)* procedure, minimizing perpendicular distances weighted by a variance ratio $\eta$.
    3.  **Proposed GOR (Unbiased):** Implementation of the unbiased linearized model proposed by *Ranjit Das et al.* which optimizes the final fit based on true orthogonal projections.
    4.  **Non-Linear Regression:** Fitted using linearization techniques, including **Exponential** ($Y = a + b \cdot e^{cx}$), **Logarithmic** ($Y = a + b \cdot \ln(x)$), **Power** ($Y = a \cdot x^b$), and **Quadratic** ($Y = ax^2 + bx + c$) models.
    5.  **Power Law Model:** Fitted as $Y = a \cdot X^b$ via dual logarithmic transformation with analysis of scaling applicability in natural sciences and seismology (such as Omori's Law or Gutenberg-Richter Law).
    6.  **Power Function (Educational):** Specialized module that fits $y = a \cdot x^b$ using $\log_{10}$ and teaches the difference between the **correct mathematical parameters solving** ($b = m$) vs. the **common incorrect spreadsheet solving** ($b = 10^m$).
    7.  **Exponential Function (Educational):** Dedicated module focused on modeling $y = a \cdot e^{bx}$ applying the natural logarithm ($\ln$) on the dependent variable.
*   📉 **Interactive Dashboard (Plotly)**:
    *   Dynamic interactive charts with tooltips and zoom capabilities in the original data space.
    *   Additional charts in linearized spaces to observe the fit quality of mathematical transformations.
*   🗂️ **Tab-Organized Interface (8 Tabs)**:
    *   *1. SLR (Least Squares):* Concept, detailed LaTeX formulas, and coefficient standard error calculations.
    *   *2. GOR Conventional:* Variance ratio parameter ($\eta$), orthogonal projections ($X_t, Y_t$), and typical residual orthogonal deviation.
    *   *3. GOR Proposed:* Ranjit Das's mathematical innovation for bias correction.
    *   *4. Non-Linear Regression:* Transformation analysis, sum tables, interactive prediction calculator, and comparison metrics against SLR.
    *   *5. Power Law Model:* Power scaling laws applied to seismology and physics with an interactive prediction calculator.
    *   *6. Power Function:* Visual comparison in the original space of the correct curve vs. the curve with parameter solving errors (exponentially explosive curve).
    *   *7. Exponential Function:* Decay and growth equations linearized under natural base.
    *   *8. Data & Export:* Unified residuals table and download controls.
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
---

## 🇮🇳 हिन्दी

**रैखिक और गैर-रैखिक प्रतिगमन विधि तुलना** मंच पर आपका स्वागत है!

यह पायथन और [Streamlit](https://streamlit.io/) के साथ बनाया गया एक शैक्षिक और पेशेवर उपकरण है जो आपको पारंपरिक स्प्रेडशीट के विपरीत मॉडल व्यवहारों का आलोचनात्मक विश्लेषण करते हुए, एक साथ कई गणितीय प्रतिगमन विधियों को स्वचालित, प्रोजेक्ट और तुलना करने की अनुमति देता है।

---

### 🚀 मुख्य विशेषताएं

*   🌐 **इंटरएक्टिव भाषा चयनकर्ता (त्रिभाषी)**:
    *   साइडबार में बटन का उपयोग करके तुरंत **स्पैनिश**, **अंग्रेजी** और **हिन्दी** के बीच स्विच करें।
    *   सभी शैक्षिक टैब (LaTeX गणितीय स्पष्टीकरण, मध्यवर्ती योग, कैलकुलेटर और इंटरैक्टिव आलेख) का रीयल-टाइम अनुवाद।
    *   निर्यातित रिपोर्ट (PDF और Excel) पूरी तरह से चुनी गई भाषा के अनुकूल।
*   📊 **रैखिक बहु-प्रतिगमन (MLR) मॉड्यूल**:
    *   एक आश्रित चर ($Y$) और कई स्वतंत्र चरों ($X_1, X_2, \dots, X_k$) के बीच संबंध का विश्लेषण करें।
    *   भूकंपीय ग्राउंड मोशन प्रेडिक्शन इक्वेशन (GMPE) डेटासेट (`logPOA`, `M`, `M^2`, `logR`, `R`) के साथ डिफ़ॉल्ट रूप से प्रीलोड किया गया।
    *   Microsoft Excel से मेल खाने वाले प्रतिगमन आँकड़े (Multiple R, $R^2$, समायोजित $R^2$, मानक त्रुटि, ANOVA तालिका, गुणांक, मानक त्रुटि, टी सांख्यिकी, पी-मान और 95% विश्वास अंतराल) की गणना करता है।
    *   इंटरएक्टिव अवशिष्ट आलेख, लाइन फिट आलेख और सामान्य संभाव्यता आलेख शामिल हैं।
*   📈 **एकल डेटा प्रविष्टि**:
    *   **फ़ाइल अपलोड**: `.csv` या `.xlsx` प्रारूपों में सीधे डेटा अपलोड करें।
    *   **मैनुअल प्रविष्टि**: अपने बिंदुओं को गतिशील रूप से दर्ज या सही करने के लिए इंटरैक्टिव डेटा संपादक का उपयोग करें।
*   🧮 **एक साथ कई मॉडलों की तुलना**:
    1. **LSR (सामान्य न्यूनतम वर्ग):** Y-अक्ष पर ऊर्ध्वाधर त्रुटियों को न्यूनतम करना।
    2. **पारंपरिक GOR:** प्रसरण अनुपात $\eta$ द्वारा भारित लंबवत दूरियों को न्यूनतम करना (Das et al., 2018)।
    3. **प्रस्तावित GOR (निष्पक्ष):** वास्तविक लंबवत अनुमानों के आधार पर अंतिम फिट को अनुकूलित करता है।
    4. **गैर-रेखीय प्रतिगमन:** एक्सपोनेंशियल, लॉगरिदमिक, पोटेंशियल और क्वाड्रेटिक मॉडल का फिट।
*   📉 **इंटरएक्टिव डैशबोर्ड (Plotly)**:
    *   मूल डेटा स्पेस में टूलटिप्स और ज़ूम क्षमताओं के साथ गतिशील आलेख।
*   📄 **रिपोर्ट निर्यात करना**:
    *   **PDF रिपोर्ट**: सभी फिट किए गए मॉडलों के समीकरणों और मैट्रिक्स का सारांश।
    *   **Excel रिपोर्ट**: संरचित `.xlsx` दस्तावेज़ में सभी मॉडलों के लिए भविष्यवाणियां, पैरामीटर, अवशिष्ट और अनुमानित डेटा डाउनलोड करें।

---

### 🛠️ प्रौद्योगिकियां और पुस्तकालय

*   **Python:** कोर विकास भाषा।
*   **Streamlit:** इंटरैक्टिव यूजर इंटरफेस के लिए वेब फ्रेमवर्क।
*   **Plotly:** गतिशील चार्ट रेंडरिंग।
*   **Pandas & NumPy:** डेटा हेरफेर और मैट्रिक्स गणना।
*   **Scikit-Learn:** कोर रैखिक प्रतिगमन फिटिंग।
*   **Matplotlib & Seaborn:** स्थिर चार्ट जनरेशन।
*   **fpdf2:** पीडीएफ रिपोर्ट जनरेशन।
*   **openpyxl:** एक्सेल राइटिंग इंजन।

---

### 💻 स्थापना और स्थानीय उपयोग

अपने स्थानीय कंप्यूटर पर इस परियोजना को चलाने के लिए, सुनिश्चित करें कि आपके पास [Python 3](https://www.python.org/downloads/) और [Git](https://git-scm.com/) स्थापित हैं।

#### 🍎 macOS और 🐧 Linux पर
1. **रिपॉजिटरी क्लोन करें**:
    ```bash
    git clone https://github.com/j-alexander-acosta/Regresiones_Lineales.git
    cd Regresiones_Lineales
    ```
2. **वर्चुअल एनवायरनमेंट बनाएं और सक्रिय करें**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **निर्भरताएं स्थापित करें**:
    ```bash
    pip install -r requirements.txt
    ```
4. **सर्वर चलाएं**:
    ```bash
    streamlit run app.py
    ```

#### 🪟 Windows पर
1. **रिपॉजिटरी क्लोन करें**:
    ```cmd
    git clone https://github.com/j-alexander-acosta/Regresiones_Lineales.git
    cd Regresiones_Lineales
    ```
2. **वर्चुअल एनवायरनमेंट बनाएं और सक्रिय करें**:
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```
3. **निर्भरताएं स्थापित करें**:
    ```cmd
    pip install -r requirements.txt
    ```
4. **सर्वर चलाएं**:
    ```cmd
    streamlit run app.py
    ```

---

### 👨‍💻 लेखक के बारे में

**Alexander Acosta** ([@j-alexander-acosta](https://github.com/j-alexander-acosta)) द्वारा विकसित और प्रबंधित।
