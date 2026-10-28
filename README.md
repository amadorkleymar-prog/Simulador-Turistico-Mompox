#  Simulador Turístico de Mompox

Aplicación web interactiva desarrollada con **Python y Streamlit** para explorar y analizar la oferta turística registrada en Santa Cruz de Mompox, Bolívar, Colombia, mediante técnicas de ciencia de datos y estadística inferencial.

##  Descripción del proyecto

El Simulador Turístico de Mompox es un proyecto académico que busca facilitar la exploración de los datos del Registro Nacional de Turismo (RNT), permitiendo identificar características de los prestadores turísticos, analizar sus segmentos y comparar indicadores relacionados con habitaciones, camas y empleados.

La aplicación integra herramientas de análisis de datos, visualización interactiva y métodos estadísticos para presentar información que contribuya a comprender mejor la composición de la oferta turística del municipio.

##  Objetivos

- Explorar y visualizar los registros de prestadores turísticos de Mompox.
- Identificar y analizar segmentos mediante técnicas de clustering.
- Calcular intervalos de confianza y realizar pruebas de hipótesis.
- Comparar las características de los diferentes segmentos turísticos.
- Simular escenarios hipotéticos de crecimiento de la oferta turística.
- Facilitar la interpretación de los datos mediante tablas y gráficos interactivos.

##  Funcionalidades

La aplicación cuenta con las siguientes secciones:

- **Resumen:** visualización de la composición de la oferta turística.
- **Segmentación:** exploración de los segmentos identificados y sus características.
- **Estadística:** cálculo de intervalos de confianza y análisis de asociación.
- **Datos:** consulta y descarga de registros filtrados.
- **Simulador:** evaluación de escenarios hipotéticos de crecimiento de prestadores.
- **Comparador:** comparación de habitaciones, camas y empleados entre segmentos.
- **Inferencia avanzada:** intervalos de confianza, ANOVA de Welch, prueba de Kruskal-Wallis y comparaciones estadísticas por pares con corrección de Holm.

##  Tecnologías utilizadas

- Python
- Pandas y NumPy
- Streamlit
- Plotly
- SciPy
- Scikit-learn
- Matplotlib y Seaborn
- Jupyter Notebook

##  Estructura del proyecto

```text
Simulador-Turistico-Mompox/
├── app/
│   └── streamlit_app.py
├── datos/
│   ├── RNT_Mompox_preparado.csv
│   └── RNT_Mompox_segmentado.csv
├── notebooks/
│   └── analisis_turistico_mompox.ipynb
├── resultados/
├── .streamlit/
│   └── config.toml
├── .gitignore
├── requirements.txt
└── README.md
```

*La estructura corresponde a la organización prevista del proyecto. Algunos archivos pueden variar según la versión publicada en el repositorio.*

## ⚙️ Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/amadorkleymar-prog/Simulador-Turistico-Mompox.git
cd Simulador-Turistico-Mompox
```

### 2. Crear un entorno virtual

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
python -m streamlit run app/streamlit_app.py
```

Una vez iniciada, abre en el navegador la dirección local que indique Streamlit, normalmente `http://localhost:8501`.

##  Metodología

El proyecto contempla la preparación y exploración de los datos, la identificación de segmentos de prestadores turísticos, la visualización de indicadores descriptivos y la aplicación de técnicas de estadística inferencial.

Los resultados deben interpretarse considerando los supuestos de cada método, las características de los registros y las limitaciones de representatividad de los datos disponibles.

##  Consideraciones

- Las simulaciones representan escenarios hipotéticos, no predicciones garantizadas.
- Los resultados estadísticos dependen de la calidad, estructura y selección de los datos.
- La existencia de una asociación estadística no demuestra causalidad.
- Los registros del RNT no necesariamente equivalen a establecimientos únicos ni representan directamente la demanda turística.

##  Contexto académico

Proyecto desarrollado en el contexto de la formación en Ingeniería de Software, integrando conocimientos de Ciencia de Datos y Estadística Inferencial mediante una aplicación web interactiva.

## Autora

Kleymar Yorgelis Amador Cova

Ingeniería de Software

Repositorio: [Simulador Turístico de Mompox](https://github.com/amadorkleymar-prog/Simulador-Turistico-Mompox)

---

*Proyecto académico orientado al análisis exploratorio de datos turísticos y a la aplicación de métodos estadísticos.*
