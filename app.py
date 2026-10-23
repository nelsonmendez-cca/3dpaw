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

MISSING = -999.99

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

    # Normalizamos nombres de columnas
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

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
def grafico_temperatura(df):
    """Grafica únicamente el sensor principal de temperatura: mcp9808."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    if "mcp9808" in df.columns:
        serie = pd.to_numeric(df["mcp9808"], errors="coerce")
        mask = df["fecha_hora"].notna() & serie.notna()
        if mask.any():
            ax.plot(
                df.loc[mask, "fecha_hora"],
                serie.loc[mask],
                linewidth=1.3,
                color="tab:red",
                label="mcp9808 — Temperatura del Aire",
            )
            ax.legend(loc="upper left")
        else:
            ax.text(
                0.5, 0.5,
                "No existen registros numéricos válidos en 'mcp9808'",
                transform=ax.transAxes, ha="center", va="center",
            )
    else:
        ax.text(
            0.5, 0.5,
            "No se encontró la columna 'mcp9808' en el archivo",
            transform=ax.transAxes, ha="center", va="center",
        )

    ax.set_title("Temperatura del Aire (°C) — Sensor MCP9808")
    ax.set_xlabel("Fecha y hora (UTC-6)")
    ax.set_ylabel("Temperatura (°C)")
    ax.grid(True, alpha=.25)
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
    """Rosa de viento con frecuencia (%) y flecha del último dato registrado."""
    if "wind_dir" not in df.columns or "wind_speed" not in df.columns:
        return None, None

    w = pd.DataFrame({
        "direccion": pd.to_numeric(df["wind_dir"], errors="coerce"),
        "velocidad": pd.to_numeric(df["wind_speed"], errors="coerce"),
        "fecha": df["fecha_hora"],
    })

    w = w[w["direccion"].between(0, 360, inclusive="both")]
    w = w[np.isfinite(w["direccion"])]
    w_rosa = w[w["velocidad"] > 0].copy()

    if w_rosa.empty:
        return None, None

    sectores = np.arange(0, 360, 22.5)
    nombres = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"
    ]

    idx = ((w_rosa["direccion"] + 11.25) // 22.5).astype(int) % 16
    frecuencias = idx.value_counts().reindex(range(16), fill_value=0).sort_index()
    porcentajes = frecuencias / frecuencias.sum() * 100

    idx_predominante = int(porcentajes.values.argmax())
    direccion_predominante = float(sectores[idx_predominante])
    nombre_predominante = nombres[idx_predominante]
    porcentaje_predominante = float(porcentajes.iloc[idx_predominante])

    w_ultimo = w.dropna(subset=["direccion", "fecha"]).sort_values("fecha")
    if w_ultimo.empty:
        ultimo_dir, ultima_fecha, ultimo_nombre = None, None, None
    else:
        ultimo_dir = float(w_ultimo.iloc[-1]["direccion"]) % 360
        ultima_fecha = w_ultimo.iloc[-1]["fecha"]
        idx_ultimo = int(((ultimo_dir + 11.25) // 22.5)) % 16
        ultimo_nombre = nombres[idx_ultimo]

    theta = np.deg2rad(sectores)
    width = np.deg2rad(22.5)

    fig = plt.figure(figsize=(6.8, 6.4))
    ax = fig.add_subplot(111, polar=True)

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

    if ultimo_dir is not None:
        ang = np.deg2rad(ultimo_dir)
        r_flecha = max(float(porcentajes.max()) * 0.92, 10.0)
        ax.annotate(
            "",
            xy=(ang, r_flecha),
            xytext=(ang, 0),
            arrowprops=dict(
                arrowstyle="-|>", color="black", lw=3.0, mutation_scale=18
            ),
            zorder=10,
        )
        ax.scatter([ang], [r_flecha], s=55, color="black", zorder=11)
        ax.text(
            0.5, -0.10,
            f"Último dato: {ultimo_dir:.1f}° ({ultimo_nombre})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, fontweight="bold",
        )

    ax.text(
        0.5, -0.16,
        f"Predominante: {nombre_predominante} ({direccion_predominante:.1f}°) — {porcentaje_predominante:.1f}%",
        transform=ax.transAxes, ha="center", va="center", fontsize=10,
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
# Interfaz principal
# ---------------------------------------------------------------------
st.title("🌦️ 3DPaws — Analizador de estación meteorológica")
st.caption("Visualización y control exploratorio de datos 3DPaws (Zona Horaria El Salvador UTC-6)")

AUTO_REFRESH_SECONDS = 300
st_autorefresh(interval=AUTO_REFRESH_SECONDS * 1000, key="3dpaws_auto_refresh")

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1ECyuzx0Ec_6g7eoJvBkvHXWH89kgeLpJ"
DRIVE_CACHE_TTL = 240

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

    patron = re.compile(r"^recordings_(20\d{2})_(\d{2})_(\d{2})\.dat$", re.IGNORECASE)
    encontrados = []
    for item in archivos:
        nombre = Path(getattr(item, "path", str(item))).name
        m = patron.match(nombre)
        if not m:
            continue

        try:
            fecha = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue

        encontrados.append({
            "nombre": nombre,
            "id": getattr(item, "id", None),
            "fecha": fecha,
        })
    return encontrados

def descargar_archivo_drive(file_id, nombre):
    cache_dir = Path(tempfile.gettempdir()) / "3dpaws_drive"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destino = cache_dir / Path(nombre).name

    if destino.exists():
        try:
            destino.unlink()
        except OSError:
            pass

    if not file_id:
        raise RuntimeError(f"Google Drive no devolvió el ID del archivo '{nombre}'.")

    gdown.download(id=file_id, output=str(destino), quiet=True, use_cookies=False)

    if not destino.exists():
        raise FileNotFoundError(f"No se pudo descargar: {nombre}")

    return destino.read_bytes()

try:
    archivos_drive = listar_archivos_drive()
except Exception as e:
    st.error(f"No se pudo leer la carpeta pública de Google Drive: {e}")
    st.stop()

if not archivos_drive:
    st.warning("No se encontraron archivos con el formato recordings_YYYY_MM_DD.dat en Google Drive.")
    st.stop()

archivos_drive = sorted(archivos_drive, key=lambda x: (x["fecha"], x["nombre"]), reverse=True)
nombres = [x["nombre"] for x in archivos_drive]

st.subheader("📂 Datos desde Google Drive")
seleccion = st.selectbox("Archivo de la estación", nombres, index=0)

archivo_info = next(x for x in archivos_drive if x["nombre"] == seleccion)

with st.spinner(f"Descargando {seleccion}..."):
    datos_archivo = descargar_archivo_drive(archivo_info["id"], archivo_info["nombre"])

df = cargar_dat(datos_archivo)

# Resumen general
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
    "Presión Atmosférica",
    "Radiación / Luz",
    "Viento",
    "Precipitación",
    "Datos Raw",
])

# 1. Temperatura
with tabs[0]:
    st.pyplot(grafico_temperatura(df), use_container_width=True)
    if "mcp9808" in df.columns:
        t_ser = df["mcp9808"].dropna()
        if not t_ser.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Temperatura Media", f"{t_ser.mean():.1f} °C")
            c2.metric("Temperatura Mínima", f"{t_ser.min():.1f} °C")
            c3.metric("Temperatura Máxima", f"{t_ser.max():.1f} °C")

# 2. Humedad
with tabs[1]:
    if "hum_hum" in df.columns:
        st.pyplot(grafico_linea(df, ["hum_hum"], "Humedad Relativa (%) — Sensor SHT31-D", "Humedad (%)"), use_container_width=True)
        h_ser = df["hum_hum"].dropna()
        if not h_ser.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Humedad Media", f"{h_ser.mean():.1f} %")
            c2.metric("Humedad Mínima", f"{h_ser.min():.1f} %")
            c3.metric("Humedad Máxima", f"{h_ser.max():.1f} %")
    else:
        st.warning("No se encontró la columna 'hum_hum' en el archivo.")

# 3. Presión
with tabs[2]:
    if "bmp_pres" in df.columns:
        st.pyplot(grafico_linea(df, ["bmp_pres"], "Presión Atmosférica — Sensor BMP280", "Presión (hPa)"), use_container_width=True)
    else:
        st.warning("No se encontró la columna 'bmp_pres' en el archivo.")

# 4. Radiación / Luz
with tabs[3]:
    cols_luz = [c for c in ["vis_light", "uv_light", "ir_light"] if c in df.columns]
    if cols_luz:
        st.pyplot(grafico_linea(df, cols_luz, "Sensores de Luz e Irradiancia", "Nivel / Cuentas"), use_container_width=True)
    else:
        st.warning("No se encontraron las columnas 'vis_light', 'uv_light' o 'ir_light' en el archivo.")

# 5. Viento
with tabs[4]:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        fig_viento, info_viento = grafico_viento(df)
        if fig_viento:
            st.pyplot(fig_viento, use_container_width=True)
        else:
            st.info("No hay datos suficientes en 'wind_dir' y 'wind_speed'.")
    with c2:
        if "wind_speed" in df.columns:
            st.pyplot(grafico_linea(df, ["wind_speed"], "Velocidad del Viento", "Velocidad (m/s)"), use_container_width=True)

# 6. Precipitación
with tabs[5]:
    st.markdown("### 🌧️ Precipitación (Pluviógrafo)")
    if "tipping" in df.columns:
        factor = st.number_input(
            "Factor de conversión (mm por vuelco)",
            min_value=0.001,
            value=0.2,
            step=0.001,
            format="%.3f",
            help="Milímetros de lluvia equivalentes a 1 vuelco del balancín.",
        )
        serie = pd.to_numeric(df["tipping"], errors="coerce").fillna(0)
        precip_total_mm = float(serie.max()) * factor

        c1, c2 = st.columns([1, 2])
        with c1:
            st.pyplot(dibujar_pluviometro(precip_total_mm, 100), use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(df["fecha_hora"], serie * factor, linewidth=1.5, color="tab:blue")
            ax.set_title("Precipitación Acumulada (mm)")
            ax.set_xlabel("Fecha y hora (UTC-6)")
            ax.set_ylabel("Precipitación (mm)")
            ax.grid(True, alpha=.25)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
    else:
        st.warning("No se encontró la columna 'tipping' para precipitación.")

# 7. Datos
with tabs[6]:
    mostrar = st.multiselect("Variables a mostrar", options=list(df.columns), default=list(df.columns)[:10])
    st.dataframe(df[mostrar], use_container_width=True, height=450)

# Exportación
st.subheader("Exportación de datos")
st.download_button(
    "⬇️ Descargar CSV limpio",
    data=convert_df_to_csv(df),
    file_name="3DPaws_ElSalvador_limpio.csv",
    mime="text/csv",
)
