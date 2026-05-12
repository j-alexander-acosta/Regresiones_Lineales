# 📈 Análisis de Regresión Lineal: Comparación de Métodos

¡Bienvenido a la aplicación de **Comparación de Métodos de Regresión Lineal**! 

Esta es una herramienta interactiva construida con Python y [Streamlit](https://streamlit.io/) que te permite automatizar el análisis, proyectar y comparar simultáneamente múltiples métodos de regresión matemática. 

---

## 🚀 Características Principales

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

## 🛠️ Tecnologías y Librerías

*   **Python:** Lenguaje base de desarrollo.
*   **Streamlit:** Framework para el desarrollo ágil de la interfaz web (UI).
*   **Plotly:** Renderizado de gráficos interactivos, tooltips dinámicos y métricas visuales.
*   **Pandas & NumPy:** Manipulación de estructuras de datos y cálculo numérico complejo.
*   **Scikit-Learn:** Algoritmos óptimos subyacentes para el cálculo de la Regresión Lineal Simple.
*   **Matplotlib:** Renderizado en segundo plano para la exportación optimizada de gráficos en PDF.
*   **fpdf:** Generación estructural de los documentos PDF de reporte.

---

## 💻 Instalación y Uso Local

Para correr este proyecto de manera local, asegúrate de tener Python 3 instalado y sigue estos pasos desde tu terminal:

1. **Clona el Repositorio**:
    ```bash
    git clone https://github.com/j-alexander-acosta/Regresiones_Lineales.git
    cd Regresiones_Lineales
    ```

2. **Instala las dependencias**:
    Asegúrate de instalar los módulos y utilidades especificadas utilizando `pip`.
    ```bash
    pip install -r requirements.txt
    ```

3. **Ejecuta el Servidor de Streamlit**:
    ```bash
    streamlit run app.py
    ```

4. **Comienza a trabajar:** Automáticamente se abrirá tu navegador web en la dirección local `http://localhost:8501`.

---

## 👨‍💻 Acerca del Autor

Desarrollado y mantenido por **Alexander Acosta** ([@j-alexander-acosta](https://github.com/j-alexander-acosta)).
