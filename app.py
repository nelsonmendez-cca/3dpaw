import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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
    cols_temp = [c for c in ["mcp9808", "hum_temp", "bmp_temp", "st1", "mt1", "bt1"] if c in df.columns]
    
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

archivo = st.file_uploader("Carga un archivo .dat o .txt de 3DPaws", type=["dat", "txt"])

if archivo is None:
    st.info("Carga el archivo recordings_YYYY_MM_DD.dat para comenzar.")
    st.stop()

df = cargar_dat(archivo.getvalue())

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
    col_hum = [c for c in ["bme2_hum", "hum_hum", "sh1"] if c in df.columns]
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
    col_precip = "tipping" if "tipping" in df.columns else ("rg" if "rg" in df.columns else None)
    if col_precip:
        st.markdown("### Pluviógrafo")
        mm_por_vuelco = st.number_input("Factor de conversión a mm (mm/vuelco o mm/ml)", min_value=0.001, value=0.254, step=0.001, format="%.3f")
        precip = float(pd.to_numeric(df[col_precip], errors="coerce").max() or 0) * mm_por_vuelco
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.pyplot(dibujar_pluviometro(precip, 100), use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(df["fecha_hora"], pd.to_numeric(df[col_precip], errors="coerce") * mm_por_vuelco, color="tab:blue")
            ax.set_title("Precipitación acumulada")
            ax.set_xlabel("Fecha y hora (UTC-6)")
            ax.set_ylabel("mm")
            ax.grid(True, alpha=.25)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
    else:
        st.warning("El archivo no contiene columnas de precipitación (tipping / rg).")

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
