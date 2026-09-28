import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
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
def grafico_temperaturas(df):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    cols_temp = [c for c in ["bt1", "mt1", "st1"] if c in df.columns]
    
    for col in cols_temp:
        ax.plot(df["fecha_hora"], df[col], linewidth=1.2, label=col)
        
    ax.set_title("Comparativa de Sensores de Temperatura (°C)")
    ax.set_xlabel("Fecha y hora (UTC-6)")
    ax.set_ylabel("Temperatura (°C)")
    ax.grid(True, alpha=.25)
    ax.legend(loc="upper left")
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
    col_dir = "wind_dir" if "wind_dir" in df.columns else ("wd" if "wd" in df.columns else None)
    col_spd = "wind_speed" if "wind_speed" in df.columns else ("ws" if "ws" in df.columns else None)

    if not col_dir or not col_spd:
        return None
    
    w = df[[col_dir, col_spd]].dropna().copy()
    w = w[w[col_spd] > 0]
    
    if w.empty:
        return None

    sectores = np.arange(0, 360, 22.5)
    nombres = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
               "S","SSO","SO","OSO","O","ONO","NO","NNO"]

    idx = ((w[col_dir] + 11.25) // 22.5).astype(int) % 16
    frecuencias = idx.value_counts().reindex(range(16), fill_value=0).sort_index()
    porcentajes = frecuencias / frecuencias.sum() * 100

    theta = np.deg2rad(sectores)
    width = np.deg2rad(22.5)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.bar(theta, porcentajes.values, width=width, align="center", alpha=.75, color="tab:blue")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels(nombres)
    ax.set_title("Rosa de Viento — Frecuencia (%)", pad=20)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------
st.title("🌦️ 3DPaws — Analizador de estación meteorológica")
st.caption("Visualización y control exploratorio de datos 3DPaws (Zona Horaria El Salvador UTC-6)")

# ---------------------------------------------------------------------
# Fuente de datos: carpeta pública de Google Drive
# ---------------------------------------------------------------------
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1ECyuzx0Ec_6g7eoJvBkvHXWH89kgeLpJ"
DRIVE_CACHE_TTL = 600  # 10 minutos


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

    encontrados = []
    for item in archivos:
        nombre = Path(getattr(item, "path", str(item))).name
        local_path = Path(
            getattr(item, "local_path", cache_dir / nombre)
        )
        if local_path.suffix.lower() in (".dat", ".txt"):
            encontrados.append({
                "nombre": nombre,
                "path": str(local_path),
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


@st.cache_data(ttl=DRIVE_CACHE_TTL, show_spinner=False)
def descargar_archivo_drive(path):
    p = Path(path)

    if not p.exists():
        cache_dir = p.parent
        gdown.download_folder(
            url=DRIVE_FOLDER_URL,
            output=str(cache_dir),
            quiet=True,
            use_cookies=False,
        )

    if not p.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {p.name}")

    return p.read_bytes()


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
    st.warning("No se encontraron archivos .dat o .txt en Google Drive.")
    st.stop()

# El archivo con la fecha más reciente queda primero.
archivos_drive = sorted(
    archivos_drive,
    key=lambda x: (fecha_nombre_archivo(x["nombre"]), x["nombre"]),
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

df = cargar_dat(descargar_archivo_drive(archivo_info["path"]))

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
    "Humedad Directa",
    "Presión",
    "Radiación / Luz",
    "Viento",
    "Precipitación",
    "Datos",
])

with tabs[0]:
    st.pyplot(grafico_temperaturas(df), use_container_width=True)
    if "dif_temp_sensores" in df:
        st.caption("Nota: La diferencia entre sensores permite identificar desfases por radiación o sesgo de calibración.")

with tabs[1]:
    col_hum = [c for c in ["sh1"] if c in df.columns]
    if col_hum:
        st.pyplot(grafico_linea(df, col_hum, "Humedad Relativa Medida por Sensor", "Humedad (%)"), use_container_width=True)
        h_ser = df[col_hum[0]].dropna()
        if not h_ser.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Humedad Media", f"{h_ser.mean():.1f} %")
            c2.metric("Humedad Mínima", f"{h_ser.min():.1f} %")
            c3.metric("Humedad Máxima", f"{h_ser.max():.1f} %")
    else:
        st.warning("No se encontró columna de humedad en el archivo.")

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
        fig_viento = grafico_viento(df)
        if fig_viento:
            st.pyplot(fig_viento, use_container_width=True)
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
