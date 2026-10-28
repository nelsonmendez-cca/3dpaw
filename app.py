import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

def calcular_punto_rocio_magnus(temp, rh):
    if pd.isna(temp) or pd.isna(rh) or rh <= 0:
        return np.nan
    a, b = 17.27, 237.7
    alpha = ((a * temp) / (b + temp)) + math.log(rh / 100.0)
    return (b * alpha) / (a - alpha)

def evaluar_control_calidad_lluvia(row):
    pp = row.get("PP_Calculada", 0.0)
    if pd.isna(pp) or pp <= 0:
        return False

    rh = row.get("hum_hum", np.nan)
    temp = row.get("mcp9808", np.nan)
    dp = row.get("punto_rocio", np.nan)
    bp = row.get("bmp_pres", 1013.25)
    bp_ref = 1013.25

    if pd.isna(rh) or pd.isna(temp) or pd.isna(dp):
        return True

    depresion_rocio = temp - dp

    if rh >= 93.0 and depresion_rocio <= 1.5:
        return True
    if rh >= 85.0 and depresion_rocio <= 2.5 and bp <= (bp_ref + 3.0):
        return True

    return False

@st.cache_data
def cargar_dat(file_bytes):
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        sep=r"\s+",
        engine="python",
    )

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

    faltan_fecha = columnas_fecha - set(df.columns)
    if faltan_fecha:
        raise ValueError(
            "El archivo no tiene el formato 3DPaws esperado. "
            f"Faltan columnas: {', '.join(sorted(faltan_fecha))}. "
            "Solo deben procesarse archivos recordings_YYYY_MM_DD.dat."
        )

    # Reconstruimos fecha_hora sin restar desfases manuales
    df["fecha_hora"] = pd.to_datetime(
        dict(year=df["year"], month=df["mon"], day=df["day"], hour=df["hour"], minute=df["min"]),
        errors="coerce",
    )

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] <= -999, col] = np.nan

    df = df.sort_values("fecha_hora").reset_index(drop=True)

    if "mcp9808" in df.columns and "hum_hum" in df.columns:
        df["punto_rocio"] = df.apply(
            lambda r: calcular_punto_rocio_magnus(r["mcp9808"], r["hum_hum"]), axis=1
        )

    if "tipping" in df.columns:
        df["tipping_diff"] = pd.to_numeric(df["tipping"], errors="coerce").diff().fillna(0)
        df.loc[df["tipping_diff"] < 0, "tipping_diff"] = 0
        df["PP_Calculada"] = df["tipping_diff"] * 0.2
        df["Es_Lluvia_Real"] = df.apply(evaluar_control_calidad_lluvia, axis=1)
        df["PP_Plot"] = df.apply(lambda r: r["PP_Calculada"] if r["Es_Lluvia_Real"] else 0.0, axis=1)

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

def dar_formato_eje_hora(ax, df):
    """
    Formatea el eje X para mostrar la escala completa de 00:00 AM a 11:59 PM (24 horas).
    """
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # Marcas cada 3 horas (12 AM, 3 AM, 6 AM...)
    
    t_min = df["fecha_hora"].min()
    if pd.notna(t_min):
        dia_inicio = t_min.normalize()
        dia_fin = dia_inicio + pd.Timedelta(hours=23, minutes=59, seconds=59)
        ax.set_xlim(dia_inicio, dia_fin)
        
    ax.grid(True, alpha=.25)

def grafico_temperatura(df):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if "mcp9808" in df.columns:
        serie = pd.to_numeric(df["mcp9808"], errors="coerce")
        mask = df["fecha_hora"].notna() & serie.notna() & serie.between(-40, 85)
        if mask.any():
            ax.plot(df.loc[mask, "fecha_hora"], serie.loc[mask], linewidth=1.4,
                    color="tab:red", label="Temperatura")
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "Sin registros válidos del MCP9808", transform=ax.transAxes, ha="center")
    else:
        ax.text(0.5, 0.5, "No se encontró el sensor MCP9808", transform=ax.transAxes, ha="center")
    ax.set_title("Temperatura del Aire (°C)")
    ax.set_xlabel("Hora local (El Salvador)")
    ax.set_ylabel("Grados (°C)")
    dar_formato_eje_hora(ax, df)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def grafico_linea(df, columnas, etiquetas, titulo, ylabel):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for col, lbl in zip(columnas, etiquetas):
        if col in df.columns:
            ax.plot(df["fecha_hora"], df[col], linewidth=1.3, label=lbl)
    ax.set_title(titulo)
    ax.set_xlabel("Hora local")
    ax.set_ylabel(ylabel)
    ax.legend()
    dar_formato_eje_hora(ax, df)
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

    ax.text(.5, .96, "PLUVIÓMETRO", ha="center", fontsize=12, fontweight="bold")
    ax.text(.5, .02, f"Precipitación: {v:.1f} mm", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlim(.18, .95)
    ax.set_ylim(0, 1.03)
    ax.axis("off")
    fig.tight_layout()
    return fig

def grafico_viento(df):
    if "wind_dir" not in df.columns or "wind_speed" not in df.columns:
        return None, {"estado": "faltan_columnas"}

    w = pd.DataFrame({
        "direccion": pd.to_numeric(df["wind_dir"], errors="coerce"),
        "velocidad": pd.to_numeric(df["wind_speed"], errors="coerce"),
        "fecha": df["fecha_hora"],
    })
    w.loc[~w["direccion"].between(0, 360), "direccion"] = np.nan
    w.loc[~w["velocidad"].between(0, 75), "velocidad"] = np.nan
    w = w.sort_values("fecha")
    valid_speed = w.dropna(subset=["velocidad"])
    speed_now = float(valid_speed.iloc[-1]["velocidad"]) if not valid_speed.empty else np.nan
    positive = w.dropna(subset=["direccion", "velocidad"])
    w_rosa = positive[positive["velocidad"] > 0].copy()

    sectores = np.arange(0, 360, 22.5)
    nombres = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
    fig = plt.figure(figsize=(6.8, 6.4))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(sectores))
    ax.set_xticklabels(nombres)

    if not w_rosa.empty:
        idx = ((w_rosa["direccion"] % 360 + 11.25) // 22.5).astype(int) % 16
        frecuencias = idx.value_counts().reindex(range(16), fill_value=0).sort_index()
        porcentajes = frecuencias / frecuencias.sum() * 100
        ax.bar(np.deg2rad(sectores), porcentajes.values, width=np.deg2rad(22.5) * .92,
               align="center", alpha=.75, color="tab:blue", edgecolor="black", linewidth=.5)
        idx_pred = int(porcentajes.values.argmax())
        pred_nombre = nombres[idx_pred]
        pred_grados = float(sectores[idx_pred])
        pred_pct = float(porcentajes.iloc[idx_pred])

        latest = w.dropna(subset=["direccion", "velocidad", "fecha"])
        latest = latest[latest["velocidad"] > 0]
        if not latest.empty:
            last = latest.iloc[-1]
            last_dir = float(last["direccion"] % 360)
            last_idx = int(((last_dir + 11.25) // 22.5)) % 16
            last_name = nombres[last_idx]
            r_arrow = max(float(porcentajes.max()) * .92, 5.0)
            ax.annotate("", xy=(np.deg2rad(last_dir), r_arrow), xytext=(np.deg2rad(last_dir), 0),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=3, mutation_scale=18), zorder=10)
            last_text = f"Última dirección con viento: {last_dir:.1f}° ({last_name}) — {last['fecha']:%d/%m %H:%M}"
        else:
            last_text = "Sin dirección reciente asociada a viento > 0 m/s"
        ax.text(.5, -.10, last_text, transform=ax.transAxes, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(.5, -.16, f"Predominante: {pred_nombre} ({pred_grados:.1f}°) — {pred_pct:.1f}%",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        estado = "viento" if pd.notna(speed_now) and speed_now > 0 else "calma"
        n_rosa = len(w_rosa)
    else:
        pred_nombre, pred_grados, pred_pct = "Sin datos", np.nan, 0.0
        last_text = "No se dibuja dirección: no hay observaciones con velocidad > 0 m/s"
        ax.text(.5, .50, "CALMA / SIN VIENTO MEDIDO\nNo hay velocidades positivas para calcular la rosa", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=.5", facecolor="white", alpha=.85))
        ax.text(.5, -.10, last_text, transform=ax.transAxes, ha="center", va="center", fontsize=9)
        estado, n_rosa = "calma", 0

    ax.set_title("Rosa de viento — frecuencia de direcciones con viento (%)", pad=22)
    fig.tight_layout()
    return fig, {"estado": estado, "velocidad_actual": speed_now, "n_mediciones": n_rosa,
                 "predominante": pred_nombre, "predominante_grados": pred_grados,
                 "predominante_porcentaje": pred_pct}

def radiacion_extraterrestre_mj_dia(fecha, latitud_grados):
    j = pd.Timestamp(fecha).dayofyear
    phi = np.deg2rad(float(latitud_grados))
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    delta = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    x = -np.tan(phi) * np.tan(delta)
    ws = np.arccos(np.clip(x, -1, 1))
    gsc = 0.0820
    ra = (24 * 60 / np.pi) * gsc * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    return max(0.0, float(ra))

def calcular_et_balance_diario(df, latitud, capacidad_suelo, almacenamiento_inicial):
    if "mcp9808" not in df.columns:
        return pd.DataFrame()

    datos = df.dropna(subset=["fecha_hora"]).copy()
    datos["fecha"] = datos["fecha_hora"].dt.floor("D")
    datos["temp_ref"] = pd.to_numeric(datos["mcp9808"], errors="coerce")

    datos.loc[~datos["temp_ref"].between(-5, 60), "temp_ref"] = np.nan

    diario = datos.groupby("fecha").agg(
        Tmin_C=("temp_ref", "min"),
        Tmax_C=("temp_ref", "max"),
        n_temp=("temp_ref", "count")
    )

    if "PP_Plot" in datos.columns:
        diario["P_mm"] = datos.groupby("fecha")["PP_Plot"].sum(min_count=1).fillna(0)
    elif "PP_Calculada" in datos.columns:
        diario["P_mm"] = datos.groupby("fecha")["PP_Calculada"].sum(min_count=1).fillna(0)
    else:
        diario["P_mm"] = 0.0

    mediana_registros = diario["n_temp"].median()
    umbral_minimo = max(12, mediana_registros * 0.70)

    diario["Tmedia_C"] = (diario["Tmax_C"] + diario["Tmin_C"]) / 2
    diario["Ra_MJ_m2_dia"] = [radiacion_extraterrestre_mj_dia(f, latitud) for f in diario.index]
    amplitud = (diario["Tmax_C"] - diario["Tmin_C"]).clip(lower=0)

    diario["ET0_mm_dia"] = 0.0023 * (diario["Tmedia_C"] + 17.8).clip(lower=0) * np.sqrt(amplitud) * diario["Ra_MJ_m2_dia"]
    diario.loc[diario["n_temp"] < umbral_minimo, "ET0_mm_dia"] = np.nan

    almacen = float(almacenamiento_inicial)
    almacen_ini, eta_real, deficit, excedente, almacen_fin = [], [], [], [], []

    for _, row in diario.iterrows():
        almacen_ini.append(almacen)
        p = row["P_mm"]
        et0 = row["ET0_mm_dia"]

        if pd.isna(p) or pd.isna(et0):
            eta_real.append(np.nan)
            deficit.append(np.nan)
            excedente.append(np.nan)
            almacen_fin.append(almacen)
            continue

        agua = almacen + max(0.0, float(p))
        eta = min(max(0.0, float(et0)), agua)
        sobrante = max(0.0, agua - eta - capacidad_suelo)
        almacen = min(capacidad_suelo, max(0.0, agua - eta))

        eta_real.append(eta)
        deficit.append(max(0.0, float(et0) - eta))
        excedente.append(sobrante)
        almacen_fin.append(almacen)

    diario["Almacenamiento_inicial_mm"] = almacen_ini
    diario["ET_real_estimada_mm"] = eta_real
    diario["Deficit_hidrico_mm"] = deficit
    diario["Excedente_mm"] = excedente
    diario["Almacenamiento_final_mm"] = almacen_fin
    diario.index.name = "Fecha"

    return diario.reset_index()

st.title("🌦️ Estación Meteorológica — El Salvador")
st.caption("Visualización y Control de Calidad Meteorológico (Zona Horaria El Salvador UTC-6)")

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
    st.warning("No se encontraron archivos en Google Drive.")
    st.stop()

archivos_drive = sorted(archivos_drive, key=lambda x: (x["fecha"], x["nombre"]), reverse=True)
nombres = [x["nombre"] for x in archivos_drive]

st.subheader("📂 Selección de Registro")
seleccion = st.selectbox("Archivo de la estación", nombres, index=0)

archivo_info = next(x for x in archivos_drive if x["nombre"] == seleccion)

with st.spinner(f"Descargando {seleccion}..."):
    datos_archivo = descargar_archivo_drive(archivo_info["id"], archivo_info["nombre"])

df = cargar_dat(datos_archivo)

st.subheader("Resumen del registro")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observaciones", f"{len(df):,}")
c2.metric("Inicio", df["fecha_hora"].min().strftime("%d/%m %I:%M %p"))
c3.metric("Fin", df["fecha_hora"].max().strftime("%d/%m %I:%M %p"))
duracion = df["fecha_hora"].max() - df["fecha_hora"].min()
c4.metric("Duración", str(duracion).split(".")[0])

st.write(f"**Periodo:** {rango_texto(df)}")

with st.expander("🔍 Control de Calidad — Completitud de Variables"):
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

tabs = st.tabs([
    "Temperatura",
    "Humedad Relativa",
    "Presión Atmosférica",
    "Radiación / Luz",
    "Viento",
    "Precipitación / QC",
    "ET0 y Balance Hídrico",
    "Datos Raw",
])

with tabs[0]:
    st.pyplot(grafico_temperatura(df))
    if "mcp9808" in df.columns:
        t_ser = pd.to_numeric(df["mcp9808"], errors="coerce").dropna()
        if not t_ser.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Media ", f"{t_ser.mean():.1f} °C")
            c2.metric("Mínima ", f"{t_ser.min():.1f} °C")
            c3.metric("Máxima ", f"{t_ser.max():.1f} °C")

with tabs[1]:
    if "hum_hum" in df.columns:
        st.pyplot(grafico_linea(df, ["hum_hum"], ["Humedad Relativa"], "Humedad Relativa (%)", "Humedad (%)"))
        h_ser = df["hum_hum"].dropna()
        if not h_ser.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Humedad Media", f"{h_ser.mean():.1f} %")
            c2.metric("Humedad Mínima", f"{h_ser.min():.1f} %")
            c3.metric("Humedad Máxima", f"{h_ser.max():.1f} %")
    else:
        st.warning("No se encontró el registro de humedad.")

with tabs[2]:
    if "bmp_pres" in df.columns:
        st.pyplot(grafico_linea(df, ["bmp_pres"], ["Presión Atmosférica"], "Presión Atmosférica (hPa)", "Presión (hPa)"))
    else:
        st.warning("No se encontró el registro de presión.")

with tabs[3]:
    cols_luz = [c for c in ["vis_light", "uv_light", "ir_light"] if c in df.columns]
    etiquetas_luz = ["Luz Visible", "Radiación UV", "Infrarrojo"]
    if cols_luz:
        st.pyplot(grafico_linea(df, cols_luz, etiquetas_luz[:len(cols_luz)], "Sensores de Radiación e Irradiancia", "Nivel / Cuentas"))
    else:
        st.warning("No se encontraron registros de radiación.")

with tabs[4]:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        fig_viento, info_viento = grafico_viento(df)
        if fig_viento:
            st.pyplot(fig_viento)
            if info_viento and pd.notna(info_viento.get("velocidad_actual", np.nan)):
                st.metric("Última velocidad registrada", f"{info_viento['velocidad_actual']:.2f} m/s",
                          help="Si es 0 m/s, se interpreta como calma.")
            if info_viento and info_viento.get("estado") == "calma":
                st.info("Calma o ausencia de viento medido.")
        else:
            st.warning("No se encontraron las columnas wind_dir y wind_speed en el archivo.")
    with c2:
        if "wind_speed" in df.columns:
            st.pyplot(grafico_linea(df, ["wind_speed"], ["Velocidad"], "Velocidad del Viento", "Velocidad (m/s)"))

with tabs[5]:
    st.markdown("### 🌧️ Precipitación y Control de Calidad (QC)")

    if "tipping" in df.columns:
        aplicar_qc = st.checkbox("Activar filtro heurístico de lluvia", value=True)
        factor = st.number_input("Factor de conversión (mm por vuelco)", min_value=0.001, value=0.2, step=0.001, format="%.3f")

        df["PP_Calculada"] = df["tipping_diff"] * factor

        if aplicar_qc:
            df["PP_Plot"] = df.apply(lambda r: r["PP_Calculada"] if r["Es_Lluvia_Real"] else 0.0, axis=1)
            df["PP_Ficticia"] = df.apply(lambda r: r["PP_Calculada"] if not r["Es_Lluvia_Real"] else 0.0, axis=1)
        else:
            df["PP_Plot"] = df["PP_Calculada"]
            df["PP_Ficticia"] = 0.0

        total_validado = df["PP_Plot"].sum()
        total_ficticio = df["PP_Ficticia"].sum()

        m1, m2 = st.columns(2)
        m1.metric("Precipitación Validada QC", f"{total_validado:.2f} mm")
        m2.metric("Lluvia Ficticia / Ruido Descartado", f"{total_ficticio:.2f} mm")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.pyplot(dibujar_pluviometro(total_validado, 100))

        with c2:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.bar(df["fecha_hora"], df["PP_Plot"], width=0.005, color="tab:blue", label="Lluvia Validada")
            if aplicar_qc and total_ficticio > 0:
                ax.bar(df["fecha_hora"], df["PP_Ficticia"], width=0.005, color="tab:red", label="Lluvia Ficticia (Descartada)")

            ax.set_title("Intensidad de Precipitación (mm)")
            ax.set_xlabel("Hora")
            ax.set_ylabel("Precipitación (mm)")
            ax.legend()
            dar_formato_eje_hora(ax, df)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("No se encontró la variable 'tipping' en el archivo.")

with tabs[6]:
    st.markdown("### 🌱 Evapotranspiración de Referencia (ET₀) y Balance Hídrico")
    st.caption(
        "Para estaciones de bajo costo (3DPaws), la FAO valida el uso del método **Hargreaves-Samani**. "
        "Este método requiere únicamente la temperatura del aire (sensor MCP9808) y la radiación extraterrestre teórica según la latitud."
    )

    if "mcp9808" not in df.columns:
        st.error("No se encuentra el sensor MCP9808; no es posible estimar la evapotranspiración.")
    else:
        c1, c2, c3 = st.columns([1.1, 1, 1])
        with c1:
            latitud_et = st.number_input(
                "Latitud de la estación (°N)", min_value=12.0, max_value=15.0,
                value=13.70, step=0.01, format="%.4f",
                help="13.70° N es una latitud central de referencia para El Salvador."
            )
        with c2:
            dias_analisis = st.selectbox("Periodo reciente", [3, 7, 14, 30], index=1, format_func=lambda x: f"Últimos {x} días")
        with c3:
            capacidad_suelo = st.number_input(
                "Capacidad del suelo / Depósito (mm)", min_value=10.0, max_value=300.0,
                value=100.0, step=10.0,
                help="Reserva máxima de agua retenible en la zona radicular (ej. 100 mm)."
            )

        diario_completo = calcular_et_balance_diario(df, latitud_et, capacidad_suelo, 0.0)

        if diario_completo.empty:
            st.warning("No hay suficientes datos válidos de temperatura para calcular la ET₀.")
        else:
            diario_completo["Fecha"] = pd.to_datetime(diario_completo["Fecha"])
            ultima_fecha = diario_completo["Fecha"].max()
            inicio_periodo = ultima_fecha.normalize() - pd.Timedelta(days=dias_analisis - 1)
            diario_et = diario_completo[diario_completo["Fecha"] >= inicio_periodo].copy().reset_index(drop=True)
            diario_et["Balance_mm"] = diario_et["P_mm"] - diario_et["ET0_mm_dia"]

            almacen = 0.0
            almacen_fin, et_real, deficit, excedente = [], [], [], []
            for _, fila in diario_et.iterrows():
                p_mm, et0 = fila["P_mm"], fila["ET0_mm_dia"]
                if pd.isna(p_mm) or pd.isna(et0):
                    almacen_fin.append(np.nan)
                    et_real.append(np.nan)
                    deficit.append(np.nan)
                    excedente.append(np.nan)
                    continue

                agua_antes_et = almacen + max(0.0, float(p_mm))
                et_cubierta = min(max(0.0, float(et0)), agua_antes_et)
                sobrante = max(0.0, agua_antes_et - et_cubierta - capacidad_suelo)
                almacen = min(capacidad_suelo, max(0.0, agua_antes_et - et_cubierta))

                almacen_fin.append(almacen)
                et_real.append(et_cubierta)
                deficit.append(max(0.0, float(et0) - et_cubierta))
                excedente.append(sobrante)

            diario_et["Agua_estimada_mm"] = almacen_fin
            diario_et["ET_cubierta_mm"] = et_real
            diario_et["Deficit_mm"] = deficit
            diario_et["Excedente_mm"] = excedente

            lluvia_total = diario_et["P_mm"].sum(min_count=1)
            et_total = diario_et["ET0_mm_dia"].sum(min_count=1)
            balance_total = diario_et["Balance_mm"].sum(min_count=1)
            agua_final = diario_et["Agua_estimada_mm"].dropna()

            st.markdown(f"**Periodo analizado:** {diario_et['Fecha'].min():%d/%m/%Y} – {diario_et['Fecha'].max():%d/%m/%Y}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Lluvia acumulada", "—" if pd.isna(lluvia_total) else f"{lluvia_total:.1f} mm")
            k2.metric("ET₀ acumulada", "—" if pd.isna(et_total) else f"{et_total:.1f} mm")
            k3.metric("Balance lluvia − ET₀", "—" if pd.isna(balance_total) else f"{balance_total:+.1f} mm")
            k4.metric("Reserva estimada en suelo", "—" if agua_final.empty else f"{agua_final.iloc[-1]:.1f} mm")

            fig, ax = plt.subplots(figsize=(11, 4.2))
            ax.bar(diario_et["Fecha"], diario_et["P_mm"], color="#3b82f6", alpha=.8, label="Lluvia (mm)")
            ax.plot(diario_et["Fecha"], diario_et["ET0_mm_dia"], color="#f97316", marker="o", linewidth=2, label="ET₀ Hargreaves (mm/día)")
            ax.set_ylabel("mm por día")
            ax.set_xlabel("Fecha")
            ax.set_title("Evapotranspiración de Referencia vs. Lluvia Diaria")
            ax.grid(axis="y", alpha=.22)
            ax.legend(frameon=False, ncol=2)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(11, 3.4))
            ax2.bar(diario_et["Fecha"], diario_et["Agua_estimada_mm"], color="#16a34a", alpha=.8)
            ax2.axhline(capacidad_suelo, color="#475569", linestyle="--", linewidth=1.3, label=f"Capacidad del depósito: {capacidad_suelo:.0f} mm")
            ax2.set_ylim(0, max(capacidad_suelo * 1.12, 1))
            ax2.set_ylabel("Reserva de agua (mm)")
            ax2.set_title("Nivel del Depósito Teórico de Agua en Suelo")
            ax2.grid(axis="y", alpha=.22)
            ax2.legend(frameon=False)
            fig2.autofmt_xdate()
            fig2.tight_layout()
            st.pyplot(fig2)

            tabla = diario_et[["Fecha", "Tmin_C", "Tmax_C", "P_mm", "ET0_mm_dia", "Balance_mm", "Agua_estimada_mm", "Deficit_mm", "Excedente_mm"]].rename(columns={
                "Tmin_C": "T Mín (°C)", "Tmax_C": "T Máx (°C)",
                "P_mm": "Lluvia (mm)", "ET0_mm_dia": "ET₀ Estimada (mm/día)",
                "Balance_mm": "Balance (mm)", "Agua_estimada_mm": "Reserva Suelo (mm)",
                "Deficit_mm": "Déficit (mm)", "Excedente_mm": "Excedente (mm)"
            })
            st.markdown("#### Detalle por Día")
            st.dataframe(tabla.round(2), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Descargar balance hídrico (CSV)", data=convert_df_to_csv(tabla),
                file_name="3DPaws_balance_hidrico.csv", mime="text/csv"
            )

with tabs[7]:
    mostrar = st.multiselect("Variables a mostrar", options=list(df.columns), default=list(df.columns)[:10])
    st.dataframe(df[mostrar], use_container_width=True, height=450)

st.subheader("Exportación de datos")
st.download_button(
    "⬇️ Descargar CSV limpio",
    data=convert_df_to_csv(df),
    file_name="3DPaws_ElSalvador_limpio.csv",
    mime="text/csv",
)
