from calendar import monthcalendar, month_name
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu, t, ttest_ind

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

st.set_page_config(
    page_title="Simulador Turístico de Mompox",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parents[1]
RUTA_DATOS = BASE_DIR / "datos" / "RNT_Mompox_preparado.csv"
RUTA_SEGMENTOS = BASE_DIR / "datos" / "RNT_Mompox_segmentado.csv"
RUTA_CSS = Path(__file__).with_name("styles.css")

CATEGORIAS_ALOJAMIENTO = [
    "ESTABLECIMIENTOS DE ALOJAMIENTO TURÍSTICO",
    "VIVIENDAS TURÍSTICAS",
    "OTROS TIPOS DE HOSPEDAJE TURÍSTICOS NO PERMANENTES",
]

PAGINAS = [
    ("Dashboard", "⌂", "Dashboard"),
    ("Segmentación", "◎", "Segmentos"),
    ("Prestadores", "♟", "Prestadores"),
    ("Estadística", "▣", "Estadística"),
    ("Simulador", "▤", "Simulador"),
    ("Comparador", "⇄", "Comparador"),
    ("Inferencia", "∑", "Inferencia"),
]

COLORES = ["#1c1c1c", "#4a4a4a", "#7a7a7a", "#b0b0b0", "#d6d6d6"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#222", size=12),
    margin=dict(l=10, r=10, t=90, b=10),
    title=dict(x=0, xanchor="left", y=0.96, yanchor="top"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
)

def inyectar_css():
    css = RUTA_CSS.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def aplicar_tema(figura):
    figura.update_layout(**PLOTLY_LAYOUT)
    figura.update_xaxes(showgrid=False, zeroline=False)
    figura.update_yaxes(showgrid=True, gridcolor="#efefef", zeroline=False)
    return figura


def kpi_html(label, value, delta, dark=False, direction="up"):
    tone = "kpi-dark" if dark else "kpi-light"
    return (
        f'<div class="kpi-card {tone}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-delta {direction}">{delta}</div>'
        "</div>"
    )


def calendario_html(hoy: date) -> str:
    semanas = monthcalendar(hoy.year, hoy.month)
    dias = "".join(
        f'<div class="cal-dow">{nombre}</div>'
        for nombre in ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    )
    celdas = []
    for semana in semanas:
        for dia in semana:
            if dia == 0:
                celdas.append('<div class="cal-day muted"></div>')
            elif dia == hoy.day:
                celdas.append(f'<div class="cal-day today">{dia}</div>')
            else:
                celdas.append(f'<div class="cal-day">{dia}</div>')
    titulo = f"{month_name[hoy.month]} {hoy.year}"
    return (
        '<div class="calendar-wrap">'
        f'<div class="cal-head"><span>‹</span><span>{titulo}</span><span>›</span></div>'
        f'<div class="cal-grid">{dias}{"".join(celdas)}</div>'
        "</div>"
    )


def delta_porcentaje(actual, anterior):
    if anterior in (0, None) or pd.isna(anterior):
        return "Sin comparativo", "flat"
    cambio = (actual - anterior) / anterior * 100
    if cambio > 0:
        return f"↗ {cambio:.1f}% frente al año anterior", "up"
    if cambio < 0:
        return f"↘ {abs(cambio):.1f}% frente al año anterior", "down"
    return "→ 0.0% frente al año anterior", "flat"


def holm(p_valores):
    p_valores = np.asarray(p_valores, dtype=float)
    cantidad = len(p_valores)
    orden = np.argsort(p_valores)
    ajustados = np.empty(cantidad)
    maximo = 0.0
    for rango, indice in enumerate(orden):
        valor = min(1.0, (cantidad - rango) * p_valores[indice])
        maximo = max(maximo, valor)
        ajustados[indice] = maximo
    return ajustados


def promedio_seguro(tabla, columna):
    if columna not in tabla.columns:
        return 0.0
    valores = pd.to_numeric(tabla[columna], errors="coerce").dropna()
    return float(valores.mean()) if not valores.empty else 0.0


def anova_welch(*grupos):
    grupos = [np.asarray(g, dtype=float) for g in grupos]
    k = len(grupos)
    ni = np.array([len(g) for g in grupos], dtype=float)
    mi = np.array([np.mean(g) for g in grupos])
    vi = np.array([np.var(g, ddof=1) for g in grupos])
    if np.any(vi == 0):
        resultado = f_oneway(*grupos)
        return float(resultado.statistic), float(resultado.pvalue)
    wi = ni / vi
    w = wi.sum()
    media = np.sum(wi * mi) / w
    numerador = np.sum(wi * (mi - media) ** 2) / (k - 1)
    lambda_ = np.sum(((1 - wi / w) ** 2) / (ni - 1))
    denominador = 1 + (2 * (k - 2) / (k ** 2 - 1)) * lambda_
    estadistico = numerador / denominador
    gl1 = k - 1
    gl2 = (k ** 2 - 1) / (3 * lambda_) if lambda_ else np.inf
    p = float(stats.f.sf(estadistico, gl1, gl2))
    return float(estadistico), p


@st.cache_data
def cargar_datos():
    datos = pd.read_csv(RUTA_DATOS, encoding="utf-8-sig")
    segmentos = pd.read_csv(RUTA_SEGMENTOS, encoding="utf-8-sig")
    return datos, segmentos


inyectar_css()

try:
    df, segmentos = cargar_datos()
except FileNotFoundError as error:
    st.error(
        f"No se encontró uno de los archivos de datos: {error}. "
        "Verifica que ambos CSV estén dentro de la carpeta datos."
    )
    st.stop()
except Exception as error:
    st.error(f"No se pudieron cargar los datos: {error}")
    st.stop()

columnas_requeridas = {
    "categoria",
    "numero_de_habitaciones",
    "numero_de_camas",
    "numero_de_empleados",
    "segmento",
}
faltantes = columnas_requeridas - set(segmentos.columns)
if faltantes:
    st.error(
        "Al archivo segmentado le faltan estas columnas: "
        + ", ".join(sorted(faltantes))
    )
    st.stop()

for columna in [
    "numero_de_habitaciones",
    "numero_de_camas",
    "numero_de_empleados",
    "año",
]:
    if columna in df.columns:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    if columna in segmentos.columns:
        segmentos[columna] = pd.to_numeric(segmentos[columna], errors="coerce")

lista_segmentos = sorted(segmentos["segmento"].dropna().unique().tolist())
anios = sorted(df["año"].dropna().astype(int).unique().tolist()) if "año" in df.columns else []

if "pagina" not in st.session_state:
    st.session_state.pagina = "Dashboard"

# --------------------------------------------------
# BARRA LATERAL
# --------------------------------------------------

st.sidebar.markdown(
    '<div class="brand"><div class="brand-mark">M</div>Mompox</div>',
    unsafe_allow_html=True,
)

for clave, icono, etiqueta in PAGINAS:
    activo = st.session_state.pagina == clave
    if st.sidebar.button(
        f"{icono}  {etiqueta}",
        key=f"nav_{clave}",
        use_container_width=True,
        type="primary" if activo else "secondary",
    ):
        st.session_state.pagina = clave
        st.rerun()

st.sidebar.markdown(
    """
    <div class="upgrade-card">
        <h4>Proyecto académico</h4>
        <p>Exploración, segmentación e inferencia sobre el RNT de Mompox. Los resultados dependen de los archivos cargados.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="nav-caption">Filtros</div>', unsafe_allow_html=True)
seleccion = st.sidebar.multiselect(
    "Segmentos",
    options=lista_segmentos,
    default=lista_segmentos,
)
if anios:
    rango_anios = st.sidebar.select_slider(
        "Rango de años",
        options=anios,
        value=(min(anios), max(anios)),
    )
else:
    rango_anios = None

st.sidebar.markdown('<div class="nav-caption">Sistema</div>', unsafe_allow_html=True)
st.sidebar.caption("Ajustes · Centro de ayuda · Cerrar sesión")

vista = segmentos[segmentos["segmento"].isin(seleccion)].copy()
if rango_anios and "año" in vista.columns:
    vista = vista[vista["año"].between(rango_anios[0], rango_anios[1])]
df_filtrado = df.copy()
if rango_anios and "año" in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado["año"].between(rango_anios[0], rango_anios[1])]

if vista.empty and st.session_state.pagina != "Dashboard":
    st.warning("Selecciona al menos un segmento para ver los resultados.")
    st.stop()

conteo_anio = (
    df_filtrado["año"].value_counts().sort_index()
    if "año" in df_filtrado.columns
    else pd.Series(dtype=float)
)
ultimo_anio = int(conteo_anio.index.max()) if not conteo_anio.empty else None
anio_previo = int(conteo_anio.index[-2]) if len(conteo_anio) >= 2 else None
registros_ultimo = int(conteo_anio.loc[ultimo_anio]) if ultimo_anio else 0
registros_previo = int(conteo_anio.loc[anio_previo]) if anio_previo else 0
texto_delta, sentido_delta = delta_porcentaje(registros_ultimo, registros_previo)

# --------------------------------------------------
# ENCABEZADO
# --------------------------------------------------

col_titulo, col_busqueda, col_periodo = st.columns([1.4, 1.2, 1.1])
titulos = {
    "Dashboard": "Dashboard",
    "Segmentación": "Segmentos",
    "Prestadores": "Prestadores",
    "Estadística": "Estadística",
    "Simulador": "Simulador",
    "Comparador": "Comparador",
    "Inferencia": "Inferencia",
}
with col_titulo:
    st.markdown(f"<div class='topbar'><h1>{titulos[st.session_state.pagina]}</h1></div>", unsafe_allow_html=True)
consulta = ""
with col_busqueda:
    if st.session_state.pagina in ("Dashboard", "Prestadores"):
        consulta = st.text_input(
            "Buscar",
            placeholder="Buscar prestador, categoría o segmento...",
            label_visibility="collapsed",
        )
with col_periodo:
    agrupacion = st.radio(
        "Periodo",
        ["Año", "Categoría", "Segmento", "Todo"],
        horizontal=True,
        index=0,
        label_visibility="collapsed",
    )

# =====================================================
# DASHBOARD
# =====================================================

if st.session_state.pagina == "Dashboard":
    st.markdown(
        '<div class="kpi-grid">'
        + kpi_html("Registros RNT", f"{len(df_filtrado):,}", texto_delta, dark=True, direction=sentido_delta)
        + kpi_html("Prestadores segmentados", f"{len(vista):,}", f"{vista['segmento'].nunique()} segmentos activos", direction="up")
        + kpi_html("Registros del último año", f"{registros_ultimo:,}", f"Año {ultimo_anio or '—'}", direction=sentido_delta)
        + kpi_html("Categorías", f"{df_filtrado['categoria'].nunique():,}", "Oferta registrada en el RNT", direction="flat")
        + "</div>",
        unsafe_allow_html=True,
    )

    izq, der = st.columns([1.55, 1])
    with izq:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        if agrupacion == "Categoría":
            serie = df_filtrado["categoria"].value_counts().head(6)
            eje_x, titulo = "Categoría", "Registros por categoría"
            datos_graf = serie.rename_axis("eje").reset_index(name="Registros")
        elif agrupacion == "Segmento":
            serie = vista["segmento"].value_counts()
            eje_x, titulo = "Segmento", "Registros por segmento"
            datos_graf = serie.rename_axis("eje").reset_index(name="Registros")
        else:
            serie = conteo_anio
            eje_x, titulo = "Año", "Registros por año"
            datos_graf = serie.rename_axis("eje").reset_index(name="Registros")

        figura = px.bar(
            datos_graf,
            x="eje",
            y="Registros",
            title=titulo,
            text="Registros",
        )
        figura.update_traces(
            marker_color="#1c1c1c",
            marker_line_width=0,
            width=0.45,
            textposition="outside",
        )
        figura.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(aplicar_tema(figura), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with der:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(calendario_html(date.today()), unsafe_allow_html=True)
        alojamiento = vista[vista["categoria"].isin(CATEGORIAS_ALOJAMIENTO)]
        porcentaje = (len(alojamiento) / len(vista) * 100) if len(vista) else 0
        dona = go.Figure(
            data=[
                go.Pie(
                    values=[porcentaje, max(0, 100 - porcentaje)],
                    hole=0.78,
                    marker_colors=["#1c1c1c", "#ececec"],
                    textinfo="none",
                    sort=False,
                )
            ]
        )
        dona.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(text=f"<b>{porcentaje:.0f}%</b>", x=0.5, y=0.5, font_size=18, showarrow=False)
            ],
            height=160,
        )
        g1, g2 = st.columns([1.4, 0.8])
        with g1:
            st.markdown(
                f'<div class="growth"><div><div class="growth-title">Peso del alojamiento</div>'
                f'<div class="growth-delta">↗ {porcentaje:.1f}% de los segmentos filtrados</div></div></div>',
                unsafe_allow_html=True,
            )
        with g2:
            st.plotly_chart(dona, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Registros de la oferta")
    tabla_dash = df_filtrado.copy()
    if consulta:
        mascara = tabla_dash.apply(lambda fila: consulta.lower() in " ".join(map(str, fila.values)).lower(), axis=1)
        tabla_dash = tabla_dash[mascara]
    columnas_tabla = [
        col for col in [
            "razon_social_establecimiento",
            "categoria",
            "sub_categoria",
            "año",
            "numero_de_habitaciones",
            "numero_de_camas",
            "estado_rnt",
        ]
        if col in tabla_dash.columns
    ]
    mostrar = tabla_dash[columnas_tabla].head(12).rename(
        columns={
            "razon_social_establecimiento": "Prestador",
            "categoria": "Categoría",
            "sub_categoria": "Tipo",
            "año": "Año",
            "numero_de_habitaciones": "Habitaciones",
            "numero_de_camas": "Camas",
            "estado_rnt": "Estado",
        }
    )
    st.dataframe(mostrar, use_container_width=True, hide_index=True)
    st.caption(
        "Las cantidades corresponden a registros del archivo preparado. "
        "No deben interpretarse automáticamente como establecimientos únicos ni como demanda turística."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# SEGMENTACIÓN
# =====================================================

elif st.session_state.pagina == "Segmentación":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Distribución de los segmentos")
    conteo = (
        vista["segmento"].value_counts().rename_axis("Segmento").reset_index(name="Registros")
    )
    conteo["Porcentaje"] = (conteo["Registros"] / conteo["Registros"].sum() * 100).round(2)
    graf_segmentos = px.bar(
        conteo,
        x="Segmento",
        y="Registros",
        color="Segmento",
        text="Registros",
        color_discrete_sequence=COLORES,
        title="Cantidad de registros por segmento",
    )
    graf_segmentos.update_traces(marker_line_width=0, width=0.5)
    st.plotly_chart(aplicar_tema(graf_segmentos), use_container_width=True, config={"displayModeBar": False})
    st.dataframe(conteo, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Características promedio")
        variables = [
            col
            for col in ["numero_de_habitaciones", "numero_de_camas", "numero_de_empleados"]
            if col in vista.columns
        ]
        if variables:
            resumen = vista.groupby("segmento")[variables].mean().round(2)
            st.dataframe(resumen, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Categorías dentro de cada segmento")
        tabla_categorias = pd.crosstab(vista["segmento"], vista["categoria"])
        st.dataframe(tabla_categorias, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ESTADÍSTICA
# =====================================================

elif st.session_state.pagina == "Estadística":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Intervalo de confianza para habitaciones")
    alojamiento = vista[vista["categoria"].isin(CATEGORIAS_ALOJAMIENTO)]
    valores = alojamiento["numero_de_habitaciones"].dropna()
    if len(valores) >= 2:
        n = len(valores)
        media = valores.mean()
        error = stats.sem(valores)
        t_critico = stats.t.ppf(0.975, df=n - 1)
        margen = t_critico * error
        m1, m2, m3 = st.columns(3)
        m1.metric("Registros analizados", f"{n:,}")
        m2.metric("Media de habitaciones", f"{media:.2f}")
        m3.metric("IC del 95 %", f"{media - margen:.2f} – {media + margen:.2f}")
        st.caption(
            "Interpretación condicionada a los supuestos del intervalo t y a cómo se seleccionaron los registros. "
            "Si los datos no son una muestra aleatoria, la generalización poblacional es limitada."
        )
    else:
        st.warning("No hay suficientes registros para calcular el intervalo.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Asociación entre segmento y tipo de prestador")
    prueba = segmentos.copy()
    prueba["tipo_prestador"] = np.where(
        prueba["categoria"].isin(CATEGORIAS_ALOJAMIENTO),
        "Alojamiento",
        "Otros servicios",
    )
    prueba = prueba[prueba["segmento"].isin(seleccion)]
    if rango_anios and "año" in prueba.columns:
        prueba = prueba[prueba["año"].between(rango_anios[0], rango_anios[1])]
    tabla = pd.crosstab(prueba["segmento"], prueba["tipo_prestador"])
    st.write("Tabla de contingencia")
    st.dataframe(tabla, use_container_width=True)
    if tabla.shape[0] >= 2 and tabla.shape[1] >= 2:
        chi2, p, gl, esperados = chi2_contingency(tabla)
        s1, s2, s3 = st.columns(3)
        s1.metric("Chi-cuadrado", f"{chi2:.4f}")
        s2.metric("Grados de libertad", str(gl))
        s3.metric("Valor p", f"{p:.6g}")
        if (esperados < 5).any():
            st.warning("Hay frecuencias esperadas menores que 5. La aproximación chi-cuadrado puede no ser adecuada.")
        elif p < 0.05:
            st.success("Se rechaza la hipótesis nula de independencia al nivel de significancia del 5 %.")
        else:
            st.info("No se rechaza la hipótesis nula al nivel del 5 %.")
        st.caption(
            "La asociación no implica causalidad. La composición de los segmentos puede estar relacionada "
            "con las variables usadas para construirlos. Los filtros también cambian el resultado."
        )
    else:
        st.warning("Se necesitan al menos dos segmentos y dos tipos de prestador en los datos seleccionados.")
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PRESTADORES / DATOS
# =====================================================

elif st.session_state.pagina == "Prestadores":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Consultar registros segmentados")
    tabla_vista = vista.copy()
    if consulta:
        mascara = tabla_vista.apply(lambda fila: consulta.lower() in " ".join(map(str, fila.values)).lower(), axis=1)
        tabla_vista = tabla_vista[mascara]
    st.write(f"Registros con los filtros seleccionados: {len(tabla_vista)}")
    st.dataframe(tabla_vista, use_container_width=True, hide_index=True)
    csv = tabla_vista.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Descargar datos filtrados en CSV",
        data=csv,
        file_name="segmentos_turisticos_mompox.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# SIMULADOR
# =====================================================

elif st.session_state.pagina == "Simulador":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write(
        "Explora cómo podría cambiar la capacidad turística de Mompox "
        "si aumentara hipotéticamente el número de prestadores de un segmento."
    )
    st.info(
        "Esta simulación es hipotética. Los datos representan prestadores registrados "
        "y no permiten predecir directamente la llegada de turistas."
    )
    segmentos_disponibles = sorted(segmentos["segmento"].dropna().unique().tolist())
    c1, c2 = st.columns([1.2, 1.5])
    with c1:
        segmento_elegido = st.selectbox("Segmento a simular", segmentos_disponibles)
    with c2:
        crecimiento = st.slider(
            "Crecimiento hipotético de prestadores (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
        )

    datos_segmento = segmentos[segmentos["segmento"] == segmento_elegido].copy()
    cantidad_actual = len(datos_segmento)
    cantidad_adicional = round(cantidad_actual * crecimiento / 100)
    cantidad_proyectada = cantidad_actual + cantidad_adicional
    promedio_habitaciones = promedio_seguro(datos_segmento, "numero_de_habitaciones")
    promedio_camas = promedio_seguro(datos_segmento, "numero_de_camas")
    promedio_empleados = promedio_seguro(datos_segmento, "numero_de_empleados")

    k1, k2, k3 = st.columns(3)
    k1.metric("Prestadores actuales", f"{cantidad_actual:,}")
    k2.metric("Prestadores adicionales", f"{cantidad_adicional:,}", delta=f"{crecimiento}% hipotético")
    k3.metric("Prestadores proyectados", f"{cantidad_proyectada:,}")
    k4, k5, k6 = st.columns(3)
    k4.metric("Habitaciones adicionales", f"{cantidad_adicional * promedio_habitaciones:,.1f}")
    k5.metric("Camas adicionales", f"{cantidad_adicional * promedio_camas:,.1f}")
    k6.metric("Empleados adicionales", f"{cantidad_adicional * promedio_empleados:,.1f}")

    comparacion = pd.DataFrame(
        {
            "Indicador": ["Prestadores", "Habitaciones", "Camas", "Empleados"],
            "Situación actual": [
                cantidad_actual,
                cantidad_actual * promedio_habitaciones,
                cantidad_actual * promedio_camas,
                cantidad_actual * promedio_empleados,
            ],
            "Escenario simulado": [
                cantidad_proyectada,
                cantidad_proyectada * promedio_habitaciones,
                cantidad_proyectada * promedio_camas,
                cantidad_proyectada * promedio_empleados,
            ],
        }
    )
    comparacion_grafico = comparacion.melt(id_vars="Indicador", var_name="Escenario", value_name="Valor estimado")
    figura = px.bar(
        comparacion_grafico,
        x="Indicador",
        y="Valor estimado",
        color="Escenario",
        barmode="group",
        title="Situación actual frente al escenario simulado",
        text_auto=".1f",
        color_discrete_sequence=["#1c1c1c", "#9a9a9a"],
    )
    st.plotly_chart(aplicar_tema(figura), use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "Los resultados suponen que los nuevos prestadores tendrían la capacidad promedio del segmento seleccionado. "
        "Son estimaciones ilustrativas, no predicciones ni datos observados."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# COMPARADOR
# =====================================================

elif st.session_state.pagina == "Comparador":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write("Compara las características de los segmentos identificados mediante clustering.")
    columnas_comparar = ["numero_de_habitaciones", "numero_de_camas", "numero_de_empleados"]
    nombres_indicadores = {
        "numero_de_habitaciones": "Habitaciones",
        "numero_de_camas": "Camas",
        "numero_de_empleados": "Empleados",
    }
    resumen_comparacion = (
        segmentos.groupby("segmento")[columnas_comparar]
        .mean()
        .rename(columns=nombres_indicadores)
        .reset_index()
    )
    st.subheader("Promedios por segmento")
    st.dataframe(resumen_comparacion.round(2), use_container_width=True, hide_index=True)
    indicador = st.selectbox("Indicador a comparar", ["Habitaciones", "Camas", "Empleados"])
    grafico = px.bar(
        resumen_comparacion,
        x="segmento",
        y=indicador,
        color="segmento",
        text_auto=".2f",
        title=f"Promedio de {indicador.lower()} por segmento",
        color_discrete_sequence=COLORES,
        labels={"segmento": "Segmento turístico", indicador: f"Promedio de {indicador.lower()}"},
    )
    st.plotly_chart(aplicar_tema(grafico), use_container_width=True, config={"displayModeBar": False})
    mayor = resumen_comparacion.loc[resumen_comparacion[indicador].idxmax()]
    menor = resumen_comparacion.loc[resumen_comparacion[indicador].idxmin()]
    st.write(f"**Mayor promedio:** {mayor['segmento']}, con {mayor[indicador]:.2f} en {indicador.lower()}.")
    st.write(f"**Menor promedio:** {menor['segmento']}, con {menor[indicador]:.2f} en {indicador.lower()}.")
    st.info(
        "Estos resultados describen los registros analizados. "
        "No demuestran por sí solos diferencias estadísticamente significativas ni explican sus causas."
    )
    st.download_button(
        "Descargar comparación en CSV",
        data=resumen_comparacion.to_csv(index=False).encode("utf-8-sig"),
        file_name="comparacion_segmentos_mompox.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# INFERENCIA
# =====================================================

elif st.session_state.pagina == "Inferencia":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write("Intervalos de confianza y pruebas de hipótesis para comparar prestadores por segmento.")
    st.warning(
        "Los registros del RNT pueden no constituir una muestra aleatoria. "
        "Interpreta los resultados considerando los supuestos estadísticos y las características de los datos."
    )
    indicadores = {
        "Habitaciones": "numero_de_habitaciones",
        "Camas": "numero_de_camas",
        "Empleados": "numero_de_empleados",
    }
    indicador_nombre = st.selectbox("Indicador", list(indicadores.keys()), key="infer_indicador")
    columna = indicadores[indicador_nombre]
    datos_inferencia = segmentos[["segmento", columna]].copy()
    datos_inferencia[columna] = pd.to_numeric(datos_inferencia[columna], errors="coerce")
    datos_inferencia = datos_inferencia.dropna()
    grupos, nombres_grupos = [], []
    for nombre, grupo in datos_inferencia.groupby("segmento"):
        valores = grupo[columna].to_numpy(dtype=float)
        if len(valores) >= 2:
            grupos.append(valores)
            nombres_grupos.append(nombre)

    st.subheader("Intervalos de confianza del 95 %")
    filas_ic = []
    for nombre, valores in zip(nombres_grupos, grupos):
        n = len(valores)
        media = np.mean(valores)
        desviacion = np.std(valores, ddof=1)
        error = desviacion / np.sqrt(n)
        margen = t.ppf(0.975, df=n - 1) * error
        filas_ic.append(
            {
                "Segmento": nombre,
                "n": n,
                "Media": media,
                "Límite inferior 95 %": media - margen,
                "Límite superior 95 %": media + margen,
            }
        )
    tabla_ic = pd.DataFrame(filas_ic)
    if tabla_ic.empty:
        st.info("No hay suficientes datos para estimar intervalos de confianza.")
    else:
        st.dataframe(tabla_ic.round(3), use_container_width=True, hide_index=True)
        st.caption(
            "Los intervalos t estiman la incertidumbre de la media bajo sus supuestos. "
            "No garantizan representatividad de todos los prestadores turísticos de Mompox."
        )

    st.subheader("Comparación global de segmentos")
    if len(grupos) >= 2:
        estadistico_welch, p_welch = anova_welch(*grupos)
        resultado_kruskal = kruskal(*grupos)
        col1, col2 = st.columns(2)
        col1.metric("ANOVA de Welch: valor p", f"{p_welch:.3e}")
        col2.metric("Kruskal-Wallis: valor p", f"{resultado_kruskal.pvalue:.3e}")
        st.markdown("**H₀ de Welch:** las medias de los segmentos son iguales.")
        st.markdown("**H₁ de Welch:** al menos una media es diferente.")
        if p_welch < 0.05:
            st.success("Se rechaza H₀ para el ANOVA de Welch. Existe evidencia de diferencias entre al menos dos medias.")
        else:
            st.info("No se rechaza H₀ para el ANOVA de Welch. No hay evidencia suficiente de diferencias entre las medias.")
        st.caption("Kruskal-Wallis evalúa diferencias en las distribuciones por rangos. No equivale automáticamente a una prueba de medianas.")

        st.subheader("Comparaciones por pares")
        resultados_pares = []
        for (nombre_a, datos_a), (nombre_b, datos_b) in combinations(zip(nombres_grupos, grupos), 2):
            resultado_t = ttest_ind(datos_a, datos_b, equal_var=False)
            resultado_u = mannwhitneyu(datos_a, datos_b, alternative="two-sided")
            resultados_pares.append(
                {
                    "Segmento A": nombre_a,
                    "Segmento B": nombre_b,
                    "Media A": np.mean(datos_a),
                    "Media B": np.mean(datos_b),
                    "Diferencia de medias (A - B)": np.mean(datos_a) - np.mean(datos_b),
                    "p Welch": resultado_t.pvalue,
                    "p Mann-Whitney": resultado_u.pvalue,
                }
            )
        tabla_pares = pd.DataFrame(resultados_pares)
        if not tabla_pares.empty:
            tabla_pares["p Welch (Holm)"] = holm(tabla_pares["p Welch"].to_numpy())
            tabla_pares["p Mann-Whitney (Holm)"] = holm(tabla_pares["p Mann-Whitney"].to_numpy())
            tabla_pares_visual = tabla_pares.copy()
            for columna_p in ["p Welch", "p Mann-Whitney", "p Welch (Holm)", "p Mann-Whitney (Holm)"]:
                tabla_pares_visual[columna_p] = tabla_pares_visual[columna_p].map(lambda p: f"{p:.3e}")
            st.dataframe(tabla_pares_visual.round(3), use_container_width=True, hide_index=True)
            st.caption(
                "Los valores p ajustados por Holm controlan el error familiar. "
                "Un valor mostrado como 0.000e+00 indica un valor numéricamente muy pequeño, no probabilidad cero."
            )
            st.download_button(
                "Descargar comparaciones por pares",
                data=tabla_pares.to_csv(index=False).encode("utf-8-sig"),
                file_name="comparaciones_pares_mompox.csv",
                mime="text/csv",
                key="descargar_pares",
            )
    else:
        st.info("Se necesitan al menos dos segmentos con dos o más registros válidos para realizar las pruebas.")

    st.subheader("Distribución del indicador")
    figura_inferencia = px.box(
        datos_inferencia,
        x="segmento",
        y=columna,
        color="segmento",
        points="outliers",
        title=f"Distribución de {indicador_nombre.lower()} por segmento",
        color_discrete_sequence=COLORES,
        labels={"segmento": "Segmento", columna: indicador_nombre},
    )
    st.plotly_chart(aplicar_tema(figura_inferencia), use_container_width=True, config={"displayModeBar": False})
    if not tabla_ic.empty:
        st.download_button(
            "Descargar intervalos de confianza",
            data=tabla_ic.to_csv(index=False).encode("utf-8-sig"),
            file_name="intervalos_confianza_mompox.csv",
            mime="text/csv",
            key="descargar_intervalos",
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<p class="footer-note">Proyecto académico de análisis de datos turísticos de Mompox. '
    "Los resultados dependen de los archivos de datos utilizados.</p>",
    unsafe_allow_html=True,
)
