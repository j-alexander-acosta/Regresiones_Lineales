# 📈 Análisis de Regresión Lineal Simple

¡Bienvenido a la aplicación de **Análisis de Regresión Lineal Simple**! 

Esta es una herramienta interactiva construida con Python y [Streamlit](https://streamlit.io/) que te permite automatizar el análisis de regresiones calculando métricas, visualizando tendencias y exportando informes ejecutivos de manera rápida y eficiente.

---

## 🚀 Características Principales

*   📊 **Tipos de Entrada de Datos**:
    *   **Subida de Archivos**: Sube tus datos directamente en formatos `.csv` o `.xlsx`.
    *   **Ingreso Manual**: Usa un editor interactivo para ingresar o corregir tus puntos (X, Y) manualmente con suma facilidad.
*   🧮 **Cálculo Automático (OLS)**: Encuentra automáticamente el modelo de regresión óptimo utilizando `scikit-learn`. Se calcula al momento la pendiente ($m$), intersección ($b$), coeficiente de determinación ($R^2$) y residuos.
*   🛠️ **Modelado Personalizado**: Alterna entre el modelo óptimo calculado matemáticamente o ingresa de forma manual tus propios valores de pendiente e intersección para observar cómo cambian tus métricas.
*   📉 **Visualizaciones Modernas**: Gráficos claros, dibujando la línea de tendencia exacta comparada con tus datos y proyectando residuales utilizando `matplotlib` y `seaborn`.
*   📄 **Exportación  de Reportes**:
    *   **Reportes PDF**: Genera y descarga un documento listo para tu equipo o profesores con métricas, gráficos en alta resolución y la tabla de datos completa directamente a formato PDF.
    *   **Reportes Excel**: Descarga tus predicciones, métricas y datos calculados en un documento estándar `.xlsx`.

---

## 🛠️ Tecnologías y Librerías

*   **Python:** Lenguaje del desarrollo.
*   **Streamlit:** Para el desarrollo de la interfaz de usuario web (UI Frontend).
*   **Pandas & NumPy:** Manejo, manipulación y cálculo numérico sobre la tabla de datos.
*   **Scikit-Learn:** Módulo base para el cálculo algorítmico de la Regresión Lineal.
*   **Matplotlib & Seaborn:** Manejo y renderizado personalizado de la gráfica estadística.
*   **fpdf:** Generación estructural en código del documento PDF.

---

## 💻 Instalación y Uso Local

Para correr este proyecto dentro de tu entorno, asegúrate de tener Python 3 instalado y ejecutar estos pasos desde tu terminal:

1. **Clona el Repositorio**:
    ```bash
    git clone https://github.com/j-alexander-acosta/Regresiones_Lineales.git
    cd Regresiones_Lineales
    ```

2. **Instala las dependencias**:
    Asegúrate de instalar los módulos especificados utilizando `pip`.
    ```bash
    pip install -r requirements.txt
    ```

3. **Ejecuta el Servidor de Streamlit**:
    ```bash
    streamlit run app.py
    ```

4. **Comienza a trabajar:** Automáticamente se abrirá tu navegador predeterminado bajo la dirección: `http://localhost:8501`.

---

## 👨‍💻 Acerca del Autor

Desarrollado y mantenido por **Alexander Acosta** ([@j-alexander-acosta](https://github.com/j-alexander-acosta)).
