import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import gdown
from pathlib import Path
import tempfile
import re

st.set_page_config(
    page_title="3DPaws | Estación meteorológica El Salvador",
    page_icon="🌦️",
    layout="wide",
)

MISSING = -999.99  # gdown se usa para leer la carpeta pública de Drive

# ---------------------------------------------------------------------
# Carga y limpieza
# ---------------------------------------------------------------------
@st.cache_data
def cargar_dat(file_bytes):
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        sep=r"\s+",
        engine="python",
    )

    # Normalizamos nombres de columnas: algunos .dat pueden traer
    # espacios, BOM o mayúsculas. Así bt1/mt1/st1 se reconocen siempre.
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    # Convertimos las variables a numérico cuando corresponde.
    # Esto evita que una columna de temperatura leída como texto
    # termine sin dibujarse.
    columnas_fecha = {"year", "mon", "day", "hour", "min"}
    for col in df.columns:
        if col not in columnas_fecha:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    columnas_obligatorias = {"year", "mon", "day", "hour", "min"}
    faltan_fecha = columnas_obligatorias - set(df.columns)
    if faltan_fecha:
        raise ValueError(
            "El archivo no tiene el formato 3DPaws esperado. "
            f"Faltan columnas: {', '.join(sorted(faltan_fecha))}. "
            "Solo deben procesarse archivos recordings_YYYY_MM_DD.dat."
        )

    # Fecha/hora ajustada a El Salvador (UTC-6 sin horario de verano)
    df["fecha_hora"] = (
        pd.to_datetime(
            dict(year=df["year"], month=df["mon"], day=df["day"], hour=df["hour"], minute=df["min"]),
            errors="coerce",
        ) - pd.Timedelta(hours=6)
    )

    # Limpieza de valores nulos (-999.99 o similares)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] <= -999, col] = np.nan

    df = df.sort_values("fecha_hora").reset_index(drop=True)

    # Diferencia entre sensor de precisión MCP9808 y SHT31D
    if "hum_temp" in df and "mcp9808" in df:
        df["dif_temp_sensores"] = df["hum_temp"] - df["mcp9808"]

    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def resumen_variable(df, col):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return {
        "n": len(s),
        "min": s.min(),
        "max": s.max(),
        "media": s.mean(),
        "mediana": s.median(),
        "std": s.std(),
        "p05": s.quantile(.05),
        "p95": s.quantile(.95),
    }

def rango_texto(df):
    t = df["fecha_hora"].dropna()
    if t.empty:
        return "Sin fechas válidas"
    return f"{t.min():%d/%m/%Y %H:%M} → {t.max():%d/%m/%Y %H:%M}"

# ---------------------------------------------------------------------
# Funciones de Gráficos
# ---------------------------------------------------------------------
def _normalizar_nombre_columna(nombre):
    """Normaliza nombres para poder reconocer variantes del archivo .dat."""
    return (
        str(nombre)
        .replace("\ufeff", "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _buscar_columna(df, aliases):
    """Devuelve la columna real que corresponda a alguno de los alias."""
    mapa = {_normalizar_nombre_columna(c): c for c in df.columns}
    for alias in aliases:
        a = _normalizar_nombre_columna(alias)
        if a in mapa:
            return mapa[a]
    return None


def detectar_temperaturas(df):
    """Reconoce las 3 temperaturas del formato original 3DPaws.

    bmp_temp: temperatura interna del sensor de presión BMP
    hum_temp: temperatura del sensor de humedad
    mcp9808: temperatura del sensor MCP9808
    """
    sensores = {
        "bmp_temp": "Temperatura BMP (sensor de presión)",
        "hum_temp": "Temperatura del sensor de humedad",
        "mcp9808": "MCP9808",
    }

    resultado = {}
    for col, descripcion in sensores.items():
        if col in df.columns:
            serie = pd.to_numeric(df[col], errors="coerce")
            serie = serie.replace([np.inf, -np.inf], np.nan)
            n_validos = int(serie.notna().sum())
            if n_validos > 0:
                resultado[col] = (col, n_validos, descripcion)
    return resultado

def grafico_temperaturas(df):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sensores = detectar_temperaturas(df)

    for col, (_, _, descripcion) in sensores.items():
        serie = pd.to_numeric(df[col], errors="coerce")
        mask = df["fecha_hora"].notna() & serie.notna()
        if mask.any():
            ax.plot(
                df.loc[mask, "fecha_hora"],
                serie.loc[mask],
                linewidth=1.2,
                label=f"{col} — {descripcion}",
            )

    ax.set_title("Comparativa de los 3 Sensores de Temperatura (°C)")
    ax.set_xlabel("Fecha y hora (UTC-6)")
    ax.set_ylabel("Temperatura (°C)")
    ax.grid(True, alpha=.25)

    if sensores:
        ax.legend(loc="upper left")
    else:
        ax.text(
            0.5, 0.5,
            "No se encontraron datos numéricos de temperatura",
            transform=ax.transAxes, ha="center", va="center",
        )

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def grafico_linea(df, columnas, titulo, ylabel):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for col in columnas:
        if col in df.columns:
            ax.plot(df["fecha_hora"], df[col], linewidth=1.3, label=col)
    ax.set_title(titulo)
    ax.set_xlabel("Fecha y hora (UTC-6)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def dibujar_pluviometro(valor_mm, capacidad=100):
    fig, ax = plt.subplots(figsize=(4, 6))
    v = max(0, float(valor_mm))
    n = min(v / capacidad, 1)
    x0, x1, y0, y1 = .3, .7, .08, .88

    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2))
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, (y1 - y0) * n, color="tab:blue", alpha=.55))
    ax.plot([x0 - .04, x1 + .04], [y1, y1], linewidth=3, color="black")

    for mm in range(0, capacidad + 1, 10):
        y = y0 + (y1 - y0) * mm / capacidad
        ln = .07 if mm % 20 == 0 else .045
        ax.plot([x1, x1 + ln], [y, y], color="black")
        if mm % 20 == 0:
            ax.text(x1 + ln + .025, y, str(mm), va="center", fontsize=9)

    ax.text(.5, .96, "PLUVIÓGRAFO 3DPaws", ha="center", fontsize=12, fontweight="bold")
    ax.text(.5, .02, f"Precipitación: {v:.1f} mm", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlim(.18, .95)
    ax.set_ylim(0, 1.03)
    ax.axis("off")
    fig.tight_layout()
    return fig

def grafico_viento(df):
    """Rosa de viento con frecuencia (%) y flecha del último dato."""
    col_dir = "wind_dir" if "wind_dir" in df.columns else ("wd" if "wd" in df.columns else None)
    col_spd = "wind_speed" if "wind_speed" in df.columns else ("ws" if "ws" in df.columns else None)

    if not col_dir or not col_spd:
        return None, None

    # Convertimos explícitamente a numérico para evitar problemas si Drive
    # entrega alguna columna como texto.
    w = pd.DataFrame({
        "direccion": pd.to_numeric(df[col_dir], errors="coerce"),
        "velocidad": pd.to_numeric(df[col_spd], errors="coerce"),
        "fecha": df["fecha_hora"],
    })

    # Dirección válida 0–360. Se conserva la convención meteorológica:
    # 0°/360° = N, 90° = E, 180° = S, 270° = O.
    w = w[w["direccion"].between(0, 360, inclusive="both")]
    w = w[np.isfinite(w["direccion"])]

    # Para la rosa se mantienen las observaciones con velocidad positiva,
    # igual que en la versión anterior.
    w_rosa = w[w["velocidad"] > 0].copy()

    if w_rosa.empty:
        return None, None

    sectores = np.arange(0, 360, 22.5)
    nombres = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"
    ]

    # Cada observación se asigna al sector de 22.5° más cercano.
    idx = ((w_rosa["direccion"] + 11.25) // 22.5).astype(int) % 16
    frecuencias = idx.value_counts().reindex(range(16), fill_value=0).sort_index()
    porcentajes = frecuencias / frecuencias.sum() * 100

    # Sector predominante según frecuencia.
    idx_predominante = int(porcentajes.values.argmax())
    direccion_predominante = float(sectores[idx_predominante])
    nombre_predominante = nombres[idx_predominante]
    porcentaje_predominante = float(porcentajes.iloc[idx_predominante])

    # Último dato válido de dirección. No necesariamente tiene que ser el
    # último registro del archivo si ese registro viene sin dirección.
    w_ultimo = w.dropna(subset=["direccion", "fecha"]).sort_values("fecha")
    if w_ultimo.empty:
        ultimo_dir = None
        ultima_fecha = None
        ultimo_nombre = None
    else:
        ultimo_dir = float(w_ultimo.iloc[-1]["direccion"]) % 360
        ultima_fecha = w_ultimo.iloc[-1]["fecha"]
        idx_ultimo = int(((ultimo_dir + 11.25) // 22.5)) % 16
        ultimo_nombre = nombres[idx_ultimo]

    theta = np.deg2rad(sectores)
    width = np.deg2rad(22.5)

    fig = plt.figure(figsize=(6.8, 6.4))
    ax = fig.add_subplot(111, polar=True)

    # Rosa de viento: porcentaje de las mediciones disponibles hasta el momento.
    ax.bar(
        theta,
        porcentajes.values,
        width=width * 0.92,
        align="center",
        alpha=.75,
        color="tab:blue",
        edgecolor="black",
        linewidth=.5,
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels(nombres)
    ax.set_ylabel("Frecuencia (%)")
    ax.set_title("Rosa de Viento — Frecuencia de las mediciones (%)", pad=22)

    # Flecha de alto contraste: dirección del ÚLTIMO dato registrado.
    if ultimo_dir is not None:
        ang = np.deg2rad(ultimo_dir)
        r_flecha = max(float(porcentajes.max()) * 0.92, 10.0)
        ax.annotate(
            "",
            xy=(ang, r_flecha),
            xytext=(ang, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=3.0,
                mutation_scale=18,
            ),
            zorder=10,
        )
        ax.scatter([ang], [r_flecha], s=55, color="black", zorder=11)

        ax.text(
            0.5,
            -0.10,
            f"Último dato: {ultimo_dir:.1f}° ({ultimo_nombre})",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.text(
        0.5,
        -0.16,
        f"Predominante: {nombre_predominante} ({direccion_predominante:.1f}°) — {porcentaje_predominante:.1f}%",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.tight_layout()

    info = {
        "n_mediciones": len(w_rosa),
        "predominante": nombre_predominante,
        "predominante_grados": direccion_predominante,
        "predominante_porcentaje": porcentaje_predominante,
        "ultimo_grados": ultimo_dir,
        "ultimo_sector": ultimo_nombre,
        "ultima_fecha": ultima_fecha,
    }
    return fig, info


# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------
st.title("🌦️ 3DPaws — Analizador de estación meteorológica")
st.caption("Visualización y control exploratorio de datos 3DPaws (Zona Horaria El Salvador UTC-6)")

# Intervalo de actualización automática (5 minutos)
AUTO_REFRESH_SECONDS = 300

# ---------------------------------------------------------------------
# Actualización automática
# ---------------------------------------------------------------------
# La estación publica nuevos datos aproximadamente cada 5 minutos.
# st_autorefresh fuerza un rerun COMPLETO de la aplicación, de modo que
# también se vuelve a consultar Google Drive y se descarga la versión
# actualizada del archivo seleccionado.
st_autorefresh(interval=AUTO_REFRESH_SECONDS * 1000, key="3dpaws_auto_refresh")

st.caption("Actualización automática cada 5 minutos desde Google Drive · no es necesario recargar ni pulsar botones.")

# ---------------------------------------------------------------------
# Fuente de datos: carpeta pública de Google Drive
# ---------------------------------------------------------------------
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1ECyuzx0Ec_6g7eoJvBkvHXWH89kgeLpJ"
DRIVE_CACHE_TTL = 240  # 4 minutos; la estación actualiza aprox. cada 5 min

@st.cache_data(ttl=DRIVE_CACHE_TTL, show_spinner=False)
def listar_archivos_drive():
    cache_dir = Path(tempfile.gettempdir()) / "3dpaws_drive"
    cache_dir.mkdir(parents=True, exist_ok=True)

    archivos = gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=str(cache_dir),
        quiet=True,
        use_cookies=False,
        skip_download=True,
    )

    # SOLO se aceptan archivos con el formato oficial de la estación:
    # recordings_YYYY_MM_DD.dat
    patron = re.compile(r"^recordings_(20\d{2})_(\d{2})_(\d{2})\.dat$", re.IGNORECASE)

    encontrados = []
    for item in archivos:
        nombre = Path(getattr(item, "path", str(item))).name
        m = patron.match(nombre)
        if not m:
            continue

        try:
            fecha = pd.Timestamp(
                int(m.group(1)), int(m.group(2)), int(m.group(3))
            )
        except ValueError:
            continue

        encontrados.append({
            "nombre": nombre,
            "id": getattr(item, "id", None),
            "fecha": fecha,
        })

    return encontrados


def fecha_nombre_archivo(nombre):
    patrones = [
        r"(20\d{2})[_-](\d{1,2})[_-](\d{1,2})",
        r"(20\d{2})(\d{2})(\d{2})",
    ]
    for patron in patrones:
        m = re.search(patron, nombre)
        if m:
            try:
                return pd.Timestamp(
                    int(m.group(1)), int(m.group(2)), int(m.group(3))
                )
            except ValueError:
                pass
    return pd.Timestamp.min


def descargar_archivo_drive(file_id, nombre):
    """Descarga únicamente el archivo seleccionado de Google Drive."""
    cache_dir = Path(tempfile.gettempdir()) / "3dpaws_drive"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destino = cache_dir / Path(nombre).name

    # El archivo de Drive puede cambiar aunque conserve el mismo nombre.
    # Por eso lo volvemos a descargar en cada actualización de la página.
    if destino.exists():
        try:
            destino.unlink()
        except OSError:
            pass

    if not file_id:
        raise RuntimeError(
            f"Google Drive no devolvió el ID del archivo '{nombre}'. "
            "Instala gdown>=6.0."
        )

    gdown.download(
        id=file_id,
        output=str(destino),
        quiet=True,
        use_cookies=False,
    )

    if not destino.exists():
        raise FileNotFoundError(f"No se pudo descargar: {nombre}")

    return destino.read_bytes()


try:
    archivos_drive = listar_archivos_drive()
except Exception as e:
    st.error(f"No se pudo leer la carpeta pública de Google Drive: {e}")
    st.info(
        "Verifica que la carpeta tenga acceso: "
        "'Cualquiera con el enlace → Lector'."
    )
    st.stop()

if not archivos_drive:
    st.warning("No se encontraron archivos con el formato recordings_YYYY_MM_DD.dat en Google Drive.")
    st.stop()

# El archivo con la fecha más reciente queda primero.
archivos_drive = sorted(
    archivos_drive,
    key=lambda x: (x["fecha"], x["nombre"]),
    reverse=True,
)

nombres = [x["nombre"] for x in archivos_drive]

st.subheader("📂 Datos desde Google Drive")

seleccion = st.selectbox(
    "Archivo de la estación",
    nombres,
    index=0,
    help="Por defecto se selecciona el archivo con la fecha más reciente.",
)

archivo_info = next(
    x for x in archivos_drive if x["nombre"] == seleccion
)

st.caption(
    f"Archivo seleccionado: **{seleccion}** · "
    f"{len(archivos_drive)} archivos disponibles"
)

with st.spinner(f"Descargando {seleccion} desde Google Drive..."):
    datos_archivo = descargar_archivo_drive(archivo_info["id"], archivo_info["nombre"])

df = cargar_dat(datos_archivo)

# Información general
st.subheader("Resumen del registro")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observaciones", f"{len(df):,}")
c2.metric("Inicio", df["fecha_hora"].min().strftime("%d/%m %H:%M"))
c3.metric("Fin", df["fecha_hora"].max().strftime("%d/%m %H:%M"))
duracion = df["fecha_hora"].max() - df["fecha_hora"].min()
c4.metric("Duración", str(duracion).split(".")[0])

st.write(f"**Periodo:** {rango_texto(df)}")

# Control de calidad
with st.expander("🔍 Control de calidad de datos"):
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in ["year", "mon", "day", "hour", "min"]
    ]
    qc = []
    for col in numeric_cols:
        total = len(df)
        faltantes = int(df[col].isna().sum())
        qc.append({
            "Variable": col,
            "Válidos": total - faltantes,
            "Faltantes": faltantes,
            "% Faltantes": round(faltantes / total * 100, 2),
        })
    qc_df = pd.DataFrame(qc).sort_values("% Faltantes", ascending=False)
    st.dataframe(qc_df, use_container_width=True, hide_index=True)

# Pestañas de análisis
tabs = st.tabs([
    "Temperatura",
    "Humedad Relativa",
    "Presión",
    "Radiación / Luz",
    "Viento",
    "Precipitación",
    "Datos",
])

with tabs[0]:
    st.pyplot(grafico_temperaturas(df), use_container_width=True)

    sensores_temp = detectar_temperaturas(df)
    esperados = ["bmp_temp", "hum_temp", "mcp9808"]
    detalle = " · ".join(
        f"{col} ({sensores_temp[col][1]:,} válidos)"
        for col in esperados if col in sensores_temp
    )
    if detalle:
        st.caption("Temperaturas detectadas: " + detalle)

    faltantes = [x for x in esperados if x not in sensores_temp]
    if faltantes:
        st.warning(
            "No se encontraron datos válidos para: " + ", ".join(faltantes) +
            ". El formato esperado contiene: bmp_temp, hum_temp y mcp9808."
        )

    if "dif_temp_sensores" in df:
        st.caption("Nota: La diferencia entre sensores permite identificar desfases por radiación o sesgo de calibración.")

with tabs[1]:
    col_hum = [c for c in ["hum_hum"] if c in df.columns]
    if col_hum:
        st.pyplot(grafico_linea(df, col_hum, "Humedad Relativa", "Humedad (%)"), use_container_width=True)
        h_ser = df[col_hum[0]].dropna()
        if not h_ser.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Humedad Media", f"{h_ser.mean():.1f} %")
            c2.metric("Humedad Mínima", f"{h_ser.min():.1f} %")
            c3.metric("Humedad Máxima", f"{h_ser.max():.1f} %")
    else:
        st.warning("No se encontró la columna hum_hum (Humedad Relativa) en el archivo.")

with tabs[2]:
    cols_pres = [c for c in ["bmp_pres", "bmp_slp", "bp1"] if c in df.columns]
    if cols_pres:
        st.pyplot(grafico_linea(df, cols_pres, "Presión Atmosférica Registrada", "Presión (hPa)"), use_container_width=True)

with tabs[3]:
    cols_luz = [c for c in ["vis_light", "ir_light", "uv_light", "sv1", "si1", "su1"] if c in df.columns]
    if cols_luz:
        st.pyplot(grafico_linea(df, cols_luz, "Sensores de Luz / Irradiancia", "Cuentas / Nivel"), use_container_width=True)

with tabs[4]:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        fig_viento, info_viento = grafico_viento(df)
        if fig_viento:
            st.pyplot(fig_viento, use_container_width=True)
            if info_viento:
                if info_viento["ultima_fecha"] is not None:
                    fecha_ult = info_viento["ultima_fecha"].strftime("%d/%m/%Y %H:%M")
                else:
                    fecha_ult = "sin fecha"
                st.caption(
                    f"{info_viento['n_mediciones']:,} mediciones de viento válidas. "
                    f"Última dirección: {info_viento['ultimo_grados']:.1f}° "
                    f"({info_viento['ultimo_sector']}) · {fecha_ult}. "
                    f"Dirección predominante: {info_viento['predominante']} "
                    f"({info_viento['predominante_porcentaje']:.1f}%)."
                )
        else:
            st.info("No hay datos suficientes de dirección/velocidad de viento.")
    with c2:
        col_viento = [c for c in ["wind_speed", "ws", "wg"] if c in df.columns]
        if col_viento:
            st.pyplot(grafico_linea(df, col_viento, "Velocidad y Ráfagas de Viento", "Velocidad (m/s)"), use_container_width=True)

with tabs[5]:
    st.markdown("### 🌧️ Precipitación 3DPaws")

    if "rg" in df.columns:
        st.info(
            "`rg` está en mililitros (ml). `rgt` es el total del día actual "
            "y `rgp` el total del día anterior. La conversión ml → mm depende "
            "del área de captación."
        )

        rg = pd.to_numeric(df["rg"], errors="coerce")

        ml_por_mm = st.number_input(
            "Mililitros equivalentes a 1 mm",
            min_value=0.001,
            value=10.0,
            step=0.1,
            format="%.3f",
            help="Ajusta este valor con las dimensiones reales del colector.",
        )

        ultimo_rg = rg.dropna().iloc[-1] if not rg.dropna().empty else 0
        lluvia_mm = max(0.0, float(ultimo_rg) / ml_por_mm)

        c1, c2 = st.columns([1, 2])

        with c1:
            st.pyplot(
                dibujar_pluviometro(lluvia_mm, 100),
                use_container_width=True,
            )

        with c2:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(
                df["fecha_hora"],
                rg / ml_por_mm,
                linewidth=1.5,
            )
            ax.set_title("Precipitación registrada — `rg`")
            ax.set_xlabel("Fecha y hora (UTC-6)")
            ax.set_ylabel("Precipitación equivalente (mm)")
            ax.grid(True, alpha=.25)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        if "rgt" in df.columns:
            rgt = pd.to_numeric(df["rgt"], errors="coerce").dropna()
            if not rgt.empty:
                c1.metric("Rain total current day", f"{rgt.iloc[-1]:.2f} ml")

        if "rgp" in df.columns:
            rgp = pd.to_numeric(df["rgp"], errors="coerce").dropna()
            if not rgp.empty:
                c2.metric("Rain total prior day", f"{rgp.iloc[-1]:.2f} ml")

    elif "tipping" in df.columns:
        st.markdown("### Pluviógrafo")
        factor = st.number_input(
            "Factor de conversión (mm/vuelco)",
            min_value=0.001,
            value=0.254,
            step=0.001,
            format="%.3f",
        )
        serie = pd.to_numeric(df["tipping"], errors="coerce")
        precip = float(serie.max() or 0) * factor

        c1, c2 = st.columns([1, 2])
        with c1:
            st.pyplot(
                dibujar_pluviometro(precip, 100),
                use_container_width=True,
            )
        with c2:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(df["fecha_hora"], serie * factor, linewidth=1.5)
            ax.set_title("Precipitación acumulada")
            ax.set_xlabel("Fecha y hora (UTC-6)")
            ax.set_ylabel("mm")
            ax.grid(True, alpha=.25)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
    else:
        st.warning("El archivo no contiene `rg` ni `tipping`.")

with tabs[6]:
    mostrar = st.multiselect("Variables a mostrar", options=list(df.columns), default=list(df.columns)[:10])
    st.dataframe(df[mostrar], use_container_width=True, height=450)

# Estadísticas
st.subheader("Estadísticas resumidas")
variables_est = st.multiselect("Selecciona variables para la tabla", options=numeric_cols, default=numeric_cols[:5] if numeric_cols else [])

stats = []
for col in variables_est:
    r = resumen_variable(df, col)
    if r:
        stats.append({
            "Variable": col,
            "N": r["n"],
            "Mínimo": r["min"],
            "Máximo": r["max"],
            "Media": r["media"],
            "Mediana": r["mediana"],
            "Desv. Est.": r["std"],
        })

if stats:
    st.dataframe(pd.DataFrame(stats).round(3), use_container_width=True, hide_index=True)

# Exportación
st.subheader("Exportación de datos")
st.download_button(
    "⬇️ Descargar CSV limpio",
    data=convert_df_to_csv(df),
    file_name="3DPaws_ElSalvador_limpio.csv",
    mime="text/csv",
)
