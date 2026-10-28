import io
import math
import json
import time
import re
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import gdown

st.set_page_config(
    page_title="3DPaws | Estación meteorológica El Salvador",
    page_icon="🌦️",
    layout="wide",
)

MISSING = -999.99
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1ECyuzx0Ec_6g7eoJvBkvHXWH89kgeLpJ"
AUTO_REFRESH_SECONDS = 300

# Directorio de caché persistente dentro del proyecto para evitar bloqueos y descargas lentas
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = CACHE_DIR / "drive_index.json"

st_autorefresh(interval=AUTO_REFRESH_SECONDS * 1000, key="3dpaws_auto_refresh")

# ==========================================================
# CÁLCULOS METEOROLÓGICOS Y CONTROL DE CALIDAD (QC)
# ==========================================================

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

# ==========================================================
# PROCESAMIENTO RAW Y RECONSTRUCCIÓN TEMPORAL
# ==========================================================

def procesar_df_raw(file_bytes):
    try:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=r"\s+",
            engine="python",
        )
    except Exception:
        return pd.DataFrame()

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )

    columnas_fecha = {"year", "mon", "day", "hour", "min"}
    faltan_fecha = columnas_fecha - set(df.columns)
    if faltan_fecha:
        return pd.DataFrame()

    for col in df.columns:
        if col not in columnas_fecha:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Hora UTC registrada por la estación a Hora Local El Salvador (GMT-6 / UTC-6)
    df["fecha_hora"] = pd.to_datetime(
        dict(year=df["year"], month=df["mon"], day=df["day"], hour=df["hour"], minute=df["min"]),
        errors="coerce",
    ) - pd.Timedelta(hours=6)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] <= -999, col] = np.nan

    return df

# ==========================================================
# GESTIÓN Y CACHÉ DE ARCHIVOS DE GOOGLE DRIVE
# ==========================================================

@st.cache_data(ttl=240, show_spinner=False)
def listar_archivos_drive(forzar_refresco=False):
    """
    Lista los archivos recordings_YYYY_MM_DD.dat desde Google Drive (solo lectura).
    Utiliza caché local en disco para arranque instantáneo (0 ms).
    Se refresca automáticamente cada 4 minutos para detectar nuevos archivos (ej. corte de 6 PM).
    """
    if INDEX_FILE.exists() and not forzar_refresco:
        edad = time.time() - INDEX_FILE.stat().st_mtime
        if edad < 240:
            try:
                with open(INDEX_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data:
                    return data
            except Exception:
                pass

    patron = re.compile(r"^recordings_(20\d{2})_(\d{2})_(\d{2})\.dat$", re.IGNORECASE)
    try:
        archivos = gdown.download_folder(
            url=DRIVE_FOLDER_URL,
            quiet=True,
            use_cookies=False,
            skip_download=True,
        )
        encontrados = []
        for item in archivos:
            nombre = Path(getattr(item, "path", str(item))).name
            m = patron.match(nombre)
            if not m:
                continue

            encontrados.append({
                "nombre": nombre,
                "id": getattr(item, "id", None),
                "fecha": f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
            })
        if encontrados:
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(encontrados, f, indent=2)
            return encontrados
    except Exception as e:
        if INDEX_FILE.exists():
            try:
                with open(INDEX_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        raise e

    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def obtener_bytes_archivo(file_info, forzar_recarga=False):
    """
    Obtiene los bytes de un archivo .dat mediante descarga de SOLO LECTURA desde Drive.
    - Los archivos de días pasados cerrados (> 2 días) se leen directamente del disco (0 ms).
    - Los archivos recientes (de hoy y ayer en UTC) se refrescan si tienen más de 4 minutos.
    - Los datos en Google Drive NUNCA se alteran ni modifican.
    """
    nombre = file_info["nombre"]
    file_id = file_info["id"]
    fecha_archivo = pd.to_datetime(file_info["fecha"]).date()
    destino = CACHE_DIR / nombre

    hoy_utc = pd.Timestamp.now(tz="UTC").date()
    # Archivo reciente: puede recibir nuevos datos de la estación en vivo
    es_archivo_reciente = (fecha_archivo >= hoy_utc - pd.Timedelta(days=1))

    # 1. Si ya existe y es del pasado cerrado (> 1 día), nunca cambia
    if destino.exists() and destino.stat().st_size > 0 and not es_archivo_reciente and not forzar_recarga:
        return destino.read_bytes()

    # 2. Si es archivo reciente, reutilizar si fue descargado hace menos de 4 minutos (240s)
    if destino.exists() and es_archivo_reciente and not forzar_recarga:
        edad = time.time() - destino.stat().st_mtime
        if edad < 240:
            return destino.read_bytes()

    # 3. Descargar actualización (solo lectura) desde Drive
    try:
        if file_id:
            gdown.download(id=file_id, output=str(destino), quiet=True, use_cookies=False)
            if destino.exists() and destino.stat().st_size > 0:
                return destino.read_bytes()
    except Exception:
        if destino.exists() and destino.stat().st_size > 0:
            return destino.read_bytes()

    if destino.exists() and destino.stat().st_size > 0:
        return destino.read_bytes()

    raise FileNotFoundError(f"No se pudo descargar el registro: {nombre}")

# ==========================================================
# RECONSTRUCCIÓN DEL DÍA LOCAL COMPLETO (SOLUCIÓN CORTE 6 PM)
# ==========================================================

@st.cache_data(ttl=120)
def cargar_datos_dia_local(fecha_str, archivos_dict, forzar_recarga=False):
    """
    Reconstruye las 24 horas del día seleccionado en El Salvador (UTC-6).
    
    Explicación del corte de las 6:00 PM:
    - La estación en Linux guarda en UTC (00:00 UTC = 18:00 El Salvador del día anterior).
    - Por lo tanto, para cualquier día local D:
      1. El bloque diurno 00:00 a 17:59 se encuentra en recordings_D.dat
      2. El bloque nocturno 18:00 a 23:59 se encuentra en recordings_(D+1).dat
    
    Al combinarlos y filtrar la fecha local D:
    - Ya no hay reinicio ni pérdida de datos a las 6:00 PM.
    - Se visualiza un día completo, continuo y unificado de 00:00 a 24:00.
    """
    target_date = pd.to_datetime(fecha_str).date()
    next_date = target_date + pd.Timedelta(days=1)

    file_curr_name = f"recordings_{target_date.strftime('%Y_%m_%d')}.dat"
    file_next_name = f"recordings_{next_date.strftime('%Y_%m_%d')}.dat"

    info_curr = archivos_dict.get(file_curr_name)
    info_next = archivos_dict.get(file_next_name)

    dfs = []

    # Bloque 1: recordings_D.dat (00:00 a 17:59 hora El Salvador)
    if info_curr:
        bytes_curr = obtener_bytes_archivo(info_curr, forzar_recarga)
        df_curr = procesar_df_raw(bytes_curr)
        if not df_curr.empty:
            df_curr_f = df_curr[df_curr["fecha_hora"].dt.date == target_date]
            if not df_curr_f.empty:
                dfs.append(df_curr_f)

    # Bloque 2: recordings_(D+1).dat (18:00 a 23:59 hora El Salvador)
    if info_next:
        try:
            bytes_next = obtener_bytes_archivo(info_next, forzar_recarga)
            df_next = procesar_df_raw(bytes_next)
            if not df_next.empty:
                df_next_f = df_next[df_next["fecha_hora"].dt.date == target_date]
                if not df_next_f.empty:
                    dfs.append(df_next_f)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["fecha_hora"]).sort_values("fecha_hora").reset_index(drop=True)

    # Filtrar estrictamente la ventana de 00:00 a 23:59:59 del día objetivo
    inicio_dia = pd.Timestamp(target_date).normalize()
    fin_dia = inicio_dia + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df = df[(df["fecha_hora"] >= inicio_dia) & (df["fecha_hora"] <= fin_dia)].reset_index(drop=True)

    if df.empty:
        return df

    # Calcular punto de rocío
    if "mcp9808" in df.columns and "hum_hum" in df.columns:
        df["punto_rocio"] = df.apply(
            lambda r: calcular_punto_rocio_magnus(r["mcp9808"], r["hum_hum"]), axis=1
        )

    # Control de lluvia y balance de vuelcos
    if "tipping" in df.columns:
        s_tip = pd.to_numeric(df["tipping"], errors="coerce")
        s_tip.loc[s_tip < 0] = np.nan
        tip_diff = s_tip.diff().fillna(0)
        tip_diff.loc[tip_diff < 0] = 0
        tip_diff.loc[tip_diff > 50] = 0  # Umbral de seguridad contra saltos artificiales
        df["tipping_diff"] = tip_diff
        df["PP_Calculada"] = df["tipping_diff"] * 0.2
        df["Es_Lluvia_Real"] = df.apply(evaluar_control_calidad_lluvia, axis=1)
        df["PP_Plot"] = df.apply(lambda r: r["PP_Calculada"] if r["Es_Lluvia_Real"] else 0.0, axis=1)

    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def rango_texto(df):
    t = df["fecha_hora"].dropna()
    if t.empty:
        return "Sin fechas válidas"
    return f"{t.min():%d/%m/%Y %I:%M %p} → {t.max():%d/%m/%Y %I:%M %p}"

def dar_formato_eje_hora(ax, df, fecha_objetivo=None):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

    if fecha_objetivo is not None:
        dia_inicio = pd.Timestamp(fecha_objetivo).normalize()
        dia_fin = dia_inicio + pd.Timedelta(hours=23, minutes=59, seconds=59)
        ax.set_xlim(dia_inicio, dia_fin)
    elif not df.empty and pd.notna(df["fecha_hora"].min()):
        dia_inicio = df["fecha_hora"].min().normalize()
        dia_fin = dia_inicio + pd.Timedelta(hours=23, minutes=59, seconds=59)
        ax.set_xlim(dia_inicio, dia_fin)

    ax.grid(True, alpha=.25)

# ==========================================================
# GRÁFICOS
# ==========================================================

def grafico_temperatura(df, fecha_objetivo=None):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if "mcp9808" in df.columns and not df.empty:
        serie = pd.to_numeric(df["mcp9808"], errors="coerce")
        mask = df["fecha_hora"].notna() & serie.notna() & serie.between(-40, 85)
        if mask.any():
            ax.plot(df.loc[mask, "fecha_hora"], serie.loc[mask], linewidth=1.4,
                    color="tab:red", label="Temperatura")
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "Sin registros válidos del MCP9808", transform=ax.transAxes, ha="center")
    else:
        ax.text(0.5, 0.5, "No se encontraron datos para este día", transform=ax.transAxes, ha="center")
    ax.set_title("Temperatura del Aire (°C) — Día Local Completo (00:00 a 24:00)")
    ax.set_xlabel("Hora local (El Salvador)")
    ax.set_ylabel("Grados (°C)")
    dar_formato_eje_hora(ax, df, fecha_objetivo)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def grafico_linea(df, columnas, etiquetas, titulo, ylabel, fecha_objetivo=None):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if not df.empty:
        for col, lbl in zip(columnas, etiquetas):
            if col in df.columns:
                ax.plot(df["fecha_hora"], df[col], linewidth=1.3, label=lbl)
    ax.set_title(titulo)
    ax.set_xlabel("Hora local (El Salvador)")
    ax.set_ylabel(ylabel)
    ax.legend()
    dar_formato_eje_hora(ax, df, fecha_objetivo)
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
    if "wind_dir" not in df.columns or "wind_speed" not in df.columns or df.empty:
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
            last_text = f"Última dirección con viento: {last_dir:.1f}° ({last_name}) — {last['fecha']:%d/%m %I:%M %p}"
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

# ==========================================================
# INTERFAZ PRINCIPAL
# ==========================================================

st.title("🌦️ Estación Meteorológica — El Salvador")
st.caption("Visualización y Control de Calidad Meteorológico (Zona Horaria El Salvador GMT-6 / UTC-6)")

# Barra lateral para control de sincronización
with st.sidebar:
    st.header("⚙️ Configuración y Estado")
    st.info("🕒 **Zona horaria:** El Salvador (GMT-6)")
    st.write(f"🔄 **Auto-refresco:** cada {AUTO_REFRESH_SECONDS // 60} minutos")
    
    forzar_refresco = st.button("🔄 Actualizar lista de Google Drive")
    if forzar_refresco:
        st.cache_data.clear()
        st.rerun()

# Cargar catálogo de archivos
try:
    archivos_drive = listar_archivos_drive(forzar_refresco=forzar_refresco)
except Exception as e:
    st.error(f"No se pudo consultar Google Drive: {e}")
    st.stop()

if not archivos_drive:
    st.warning("No se encontraron archivos recordings_YYYY_MM_DD.dat en Google Drive.")
    st.stop()

archivos_dict = {x["nombre"]: x for x in archivos_drive}
fechas_disponibles = sorted(list(set(x["fecha"] for x in archivos_drive)), reverse=True)

# Cálculo de fechas de referencia local (UTC - 6h)
ahora_local = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=6)
hoy_local = ahora_local.date()
ayer_local = hoy_local - pd.Timedelta(days=1)

def formatear_fecha_opcion(fecha_val):
    if isinstance(fecha_val, str) and ("📅" in fecha_val or "/" in fecha_val):
        return fecha_val
    try:
        d = pd.to_datetime(fecha_val).date()
        if d == hoy_local:
            return f"📅 {d.strftime('%d/%m/%Y')} — Hoy (En curso)"
        elif d == ayer_local:
            return f"📅 {d.strftime('%d/%m/%Y')} — Ayer (Día completo)"
        else:
            return f"📅 {d.strftime('%d/%m/%Y')} (Día completo)"
    except Exception:
        return str(fecha_val)

st.subheader("📅 Selección de Día de Análisis")
fecha_seleccionada = st.selectbox(
    "Selecciona la fecha (Hora Local de El Salvador GMT-6):",
    options=fechas_disponibles,
    format_func=formatear_fecha_opcion,
    index=0,
)

# Extraer fecha limpia independientemente de si llega formateada o como YYYY-MM-DD
if isinstance(fecha_seleccionada, str):
    m_date = re.search(r"(\d{4}[-_/]\d{2}[-_/]\d{2})|(\d{2}[-_/]\d{2}[-_/]\d{4})", fecha_seleccionada)
    if m_date:
        fecha_str_clean = m_date.group(0).replace("_", "-")
        target_date = pd.to_datetime(fecha_str_clean, dayfirst=("/" in fecha_str_clean)).date()
    else:
        target_date = pd.to_datetime(fecha_seleccionada).date()
else:
    target_date = pd.to_datetime(fecha_seleccionada).date()

fecha_iso = target_date.strftime("%Y-%m-%d")
next_date = target_date + pd.Timedelta(days=1)
file_curr_name = f"recordings_{target_date.strftime('%Y_%m_%d')}.dat"
file_next_name = f"recordings_{next_date.strftime('%Y_%m_%d')}.dat"
tiene_bloque_noche = file_next_name in archivos_dict

if tiene_bloque_noche:
    st.caption(f"🔗 **Reconstrucción 24h unificada:** 00:00–17:59 (`{file_curr_name}`) + 18:00–23:59 (`{file_next_name}`).")
else:
    st.caption(f"🔗 **Día en curso:** 00:00–17:59 (`{file_curr_name}`). El bloque nocturno 18:00+ se unificará automáticamente al iniciar `{file_next_name}`.")

with st.spinner("Cargando y procesando datos meteorológicos..."):
    df = cargar_datos_dia_local(fecha_iso, archivos_dict)

if df.empty:
    st.warning(f"No se encontraron observaciones válidas para el día {fecha_iso}.")
    st.stop()

# ==========================================================
# RESUMEN Y MÉTRICAS
# ==========================================================

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
    st.dataframe(qc_df, width="stretch", hide_index=True)

# ==========================================================
# PESTAÑAS DE VISUALIZACIÓN
# ==========================================================

tabs = st.tabs([
    "Temperatura",
    "Humedad Relativa",
    "Presión Atmosférica",
    "Radiación / Luz",
    "Viento",
    "Precipitación / QC",
    "Datos Raw",
])

with tabs[0]:
    st.pyplot(grafico_temperatura(df, fecha_iso))
    if "mcp9808" in df.columns:
        t_ser = pd.to_numeric(df["mcp9808"], errors="coerce").dropna()
        if not t_ser.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Media ", f"{t_ser.mean():.1f} °C")
            c2.metric("Mínima ", f"{t_ser.min():.1f} °C")
            c3.metric("Máxima ", f"{t_ser.max():.1f} °C")

with tabs[1]:
    if "hum_hum" in df.columns:
        st.pyplot(grafico_linea(df, ["hum_hum"], ["Humedad Relativa"], "Humedad Relativa (%)", "Humedad (%)", fecha_iso))
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
        st.pyplot(grafico_linea(df, ["bmp_pres"], ["Presión Atmosférica"], "Presión Atmosférica (hPa)", "Presión (hPa)", fecha_iso))
    else:
        st.warning("No se encontró el registro de presión.")

with tabs[3]:
    cols_luz = [c for c in ["vis_light", "uv_light", "ir_light"] if c in df.columns]
    etiquetas_luz = ["Luz Visible", "Radiación UV", "Infrarrojo"]
    if cols_luz:
        st.pyplot(grafico_linea(df, cols_luz, etiquetas_luz[:len(cols_luz)], "Sensores de Radiación e Irradiancia", "Nivel / Cuentas", fecha_iso))
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
            st.pyplot(grafico_linea(df, ["wind_speed"], ["Velocidad"], "Velocidad del Viento", "Velocidad (m/s)", fecha_iso))

with tabs[5]:
    st.markdown("### 🌧️ Precipitación y Control de Calidad (QC)")

    if "tipping" in df.columns:
        df_qc = df.copy()
        aplicar_qc = st.checkbox("Activar filtro heurístico de lluvia", value=True)
        factor = st.number_input("Factor de conversión (mm por vuelco)", min_value=0.001, value=0.2, step=0.001, format="%.3f")

        df_qc["PP_Calculada"] = df_qc["tipping_diff"] * factor

        if aplicar_qc:
            df_qc["PP_Plot"] = df_qc.apply(lambda r: r["PP_Calculada"] if r["Es_Lluvia_Real"] else 0.0, axis=1)
            df_qc["PP_Ficticia"] = df_qc.apply(lambda r: r["PP_Calculada"] if not r["Es_Lluvia_Real"] else 0.0, axis=1)
        else:
            df_qc["PP_Plot"] = df_qc["PP_Calculada"]
            df_qc["PP_Ficticia"] = 0.0

        total_validado = df_qc["PP_Plot"].sum()
        total_ficticio = df_qc["PP_Ficticia"].sum()

        m1, m2 = st.columns(2)
        m1.metric("Precipitación Validada QC", f"{total_validado:.2f} mm")
        m2.metric("Lluvia Ficticia / Ruido Descartado", f"{total_ficticio:.2f} mm")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.pyplot(dibujar_pluviometro(total_validado, 100))

        with c2:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.bar(df_qc["fecha_hora"], df_qc["PP_Plot"], width=0.005, color="tab:blue", label="Lluvia Validada")
            if aplicar_qc and total_ficticio > 0:
                ax.bar(df_qc["fecha_hora"], df_qc["PP_Ficticia"], width=0.005, color="tab:red", label="Lluvia Ficticia (Descartada)")

            ax.set_title("Intensidad de Precipitación (mm)")
            ax.set_xlabel("Hora local")
            ax.set_ylabel("Precipitación (mm)")
            ax.legend()
            dar_formato_eje_hora(ax, df_qc, fecha_iso)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("No se encontró la variable 'tipping' en el archivo.")

with tabs[6]:
    mostrar = st.multiselect("Variables a mostrar", options=list(df.columns), default=list(df.columns)[:10])
    st.dataframe(df[mostrar], width="stretch", height=450)

st.subheader("Exportación de datos")
st.download_button(
    "⬇️ Descargar CSV limpio",
    data=convert_df_to_csv(df),
    file_name=f"3DPaws_ElSalvador_{fecha_iso}.csv",
    mime="text/csv",
)
