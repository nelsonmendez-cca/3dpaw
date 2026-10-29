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
AUTO_REFRESH_SECONDS = 60

# Parámetros geográficos de la estación (El Salvador)
LATITUD_ESTACION = 13.68   # grados Norte
ELEVACION_ESTACION = 800   # metros sobre el nivel del mar (aprox.)

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
# EVAPOTRANSPIRACIÓN (ET₀) Y BALANCE HÍDRICO
# ==========================================================

def calcular_ra(latitud_grados, dia_juliano):
    """Radiación extraterrestre (Ra) en MJ/m²/día — ecuaciones FAO-56."""
    lat = math.radians(latitud_grados)
    dr = 1 + 0.033 * math.cos(2 * math.pi * dia_juliano / 365)
    delta = 0.409 * math.sin(2 * math.pi * dia_juliano / 365 - 1.39)
    ws = math.acos(max(-1, min(1, -math.tan(lat) * math.tan(delta))))
    Gsc = 0.0820  # constante solar, MJ m⁻² min⁻¹
    Ra = (24 * 60 / math.pi) * Gsc * dr * (
        ws * math.sin(lat) * math.sin(delta)
        + math.cos(lat) * math.cos(delta) * math.sin(ws)
    )
    return max(Ra, 0)


def calcular_horas_sol(latitud_grados, dia_juliano):
    """Duración máxima de insolación (N) en horas — FAO-56."""
    lat = math.radians(latitud_grados)
    delta = 0.409 * math.sin(2 * math.pi * dia_juliano / 365 - 1.39)
    ws = math.acos(max(-1, min(1, -math.tan(lat) * math.tan(delta))))
    return 24 / math.pi * ws


def calcular_et0_hargreaves(tmin, tmax, tmean, ra):
    """
    ET₀ por Hargreaves-Samani (1985) en mm/día.
    Solo requiere temperatura y radiación extraterrestre.
    """
    if any(pd.isna(x) for x in [tmin, tmax, tmean]) or tmax <= tmin:
        return np.nan
    Ra_mm = ra / 2.45  # MJ/m²/día → mm/día equivalente de evaporación
    return max(0, 0.0023 * (tmean + 17.8) * math.sqrt(tmax - tmin) * Ra_mm)


def calcular_et0_penman_monteith_fao56(
    tmean, tmin, tmax, rh_mean, u2_mean, presion_kpa, ra,
    elevacion=ELEVACION_ESTACION,
):
    """
    ET₀ por FAO Penman-Monteith (FAO-56) en mm/día.
    Requiere temperatura, humedad relativa, viento, presión y Ra.
    La radiación solar incidente (Rs) se estima internamente con la
    fórmula de Hargreaves-Radiation cuando no se dispone de piranómetro.
    """
    if any(pd.isna(x) for x in [tmean, tmin, tmax, rh_mean, u2_mean, presion_kpa]):
        return np.nan

    T = tmean
    P = presion_kpa
    gamma = 0.000665 * P  # constante psicrométrica (kPa/°C)

    es_max = 0.6108 * math.exp(17.27 * tmax / (tmax + 237.3))
    es_min = 0.6108 * math.exp(17.27 * tmin / (tmin + 237.3))
    es = (es_max + es_min) / 2  # presión de saturación media
    ea = es * rh_mean / 100       # presión de vapor real

    delta_svp = (
        4098 * 0.6108 * math.exp(17.27 * T / (T + 237.3))
        / ((T + 237.3) ** 2)
    )

    # Radiación solar incidente estimada (Hargreaves radiation)
    kRs = 0.16  # coeficiente para regiones interiores
    Rs = kRs * math.sqrt(max(tmax - tmin, 0)) * ra

    Rso = (0.75 + 2e-5 * elevacion) * ra  # radiación en cielo despejado
    Rns = (1 - 0.23) * Rs                  # radiación neta de onda corta

    sigma = 4.903e-9  # Stefan-Boltzmann (MJ m⁻² día⁻¹ K⁻⁴)
    Rs_Rso = min(Rs / max(Rso, 0.01), 1.0)
    Rnl = (
        sigma
        * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2)
        * (0.34 - 0.14 * math.sqrt(max(ea, 0)))
        * (1.35 * Rs_Rso - 0.35)
    )

    Rn = Rns - Rnl  # radiación neta
    G = 0            # flujo de calor del suelo ≈ 0 a escala diaria

    num = (
        0.408 * delta_svp * (Rn - G)
        + gamma * (900 / (T + 273)) * u2_mean * (es - ea)
    )
    den = delta_svp + gamma * (1 + 0.34 * u2_mean)
    return max(0, num / den)


def calcular_balance_hidrico(
    df, metodo="hargreaves", latitud=LATITUD_ESTACION, elevacion=ELEVACION_ESTACION, kc=1.0
):
    """
    Calcula ET₀ diaria, ETc (evapotranspiración del cultivo = ET0 * Kc)
    y genera la serie acumulada de balance hídrico (P - ETc).
    Retorna un dict con métricas clave y un DataFrame con la serie temporal acumulada.
    """
    if df.empty or "fecha_hora" not in df.columns:
        return None

    fecha = df["fecha_hora"].dt.date.mode()[0]
    doy = fecha.timetuple().tm_yday
    ra = calcular_ra(latitud, doy)
    N = calcular_horas_sol(latitud, doy)

    temp_col = "mcp9808" if "mcp9808" in df.columns else None
    if temp_col is None:
        return None

    t_ser = pd.to_numeric(df[temp_col], errors="coerce").dropna()
    if t_ser.empty:
        return None

    tmin = float(t_ser.min())
    tmax = float(t_ser.max())
    tmean = float(t_ser.mean())

    # —— Hargreaves-Samani ——
    et0_hg = calcular_et0_hargreaves(tmin, tmax, tmean, ra)

    # —— Penman-Monteith FAO-56 ——
    et0_pm = np.nan
    hum_col = "hum_hum" if "hum_hum" in df.columns else None
    wind_col = "wind_speed" if "wind_speed" in df.columns else None
    pres_col = "bmp_pres" if "bmp_pres" in df.columns else None
    if all(c is not None for c in [hum_col, wind_col, pres_col]):
        rh_mean = float(pd.to_numeric(df[hum_col], errors="coerce").mean())
        u2_mean = float(pd.to_numeric(df[wind_col], errors="coerce").mean())
        p_mean_kpa = float(pd.to_numeric(df[pres_col], errors="coerce").mean()) / 10
        et0_pm = calcular_et0_penman_monteith_fao56(
            tmean, tmin, tmax, rh_mean, u2_mean, p_mean_kpa, ra, elevacion,
        )

    et0 = et0_pm if metodo == "penman_monteith" and pd.notna(et0_pm) else et0_hg
    etc = (et0 * kc) if pd.notna(et0) else np.nan

    # Precipitación acumulada
    pp_col = "PP_Plot" if "PP_Plot" in df.columns else None
    precip_total = float(df[pp_col].sum()) if pp_col else 0.0

    balance = (precip_total - etc) if pd.notna(etc) else np.nan

    # ——— Serie acumulada para gráficos ———
    df_bal = df[["fecha_hora"]].copy()
    df_bal["PP_acum"] = df[pp_col].cumsum().values if pp_col else 0.0

    # Distribución sinusoidal de ET durante horas de luz
    sunrise_h = 12 - N / 2
    sunset_h = 12 + N / 2
    hora_dec = df_bal["fecha_hora"].dt.hour + df_bal["fecha_hora"].dt.minute / 60.0
    peso = np.where(
        (hora_dec >= sunrise_h) & (hora_dec <= sunset_h),
        np.sin(np.pi * (hora_dec - sunrise_h) / max(N, 0.1)),
        0.0,
    )
    peso_acum = np.cumsum(peso)
    peso_total = peso_acum[-1] if len(peso_acum) and peso_acum[-1] > 0 else 1.0

    df_bal["ET0_acum"] = (peso_acum / peso_total) * et0 if pd.notna(et0) else 0.0
    df_bal["ETc_acum"] = (peso_acum / peso_total) * etc if pd.notna(etc) else 0.0
    df_bal["Balance_acum"] = df_bal["PP_acum"] - df_bal["ETc_acum"]

    return {
        "et0_hargreaves": et0_hg,
        "et0_penman_monteith": et0_pm,
        "et0_usado": et0,
        "kc": kc,
        "etc_usado": etc,
        "metodo": metodo,
        "precipitacion": precip_total,
        "balance": balance,
        "ra": ra,
        "N": N,
        "tmin": tmin,
        "tmax": tmax,
        "tmean": tmean,
        "doy": doy,
        "df_balance": df_bal,
    }

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

@st.cache_data(ttl=50, show_spinner=False)
def listar_archivos_drive(forzar_refresco=False):
    """
    Lista los archivos recordings_YYYY_MM_DD.dat desde Google Drive (solo lectura).
    Utiliza caché local en disco para arranque instantáneo (0 ms).
    Se refresca automáticamente cada ~50 segundos para detectar nuevos archivos.
    """
    if INDEX_FILE.exists() and not forzar_refresco:
        edad = time.time() - INDEX_FILE.stat().st_mtime
        if edad < 50:
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
    - Los archivos recientes (de hoy y ayer en UTC) se refrescan si tienen más de 50 s.
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

    # 2. Si es archivo reciente, reutilizar si fue descargado hace menos de 50 s
    if destino.exists() and es_archivo_reciente and not forzar_recarga:
        edad = time.time() - destino.stat().st_mtime
        if edad < 50:
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

@st.cache_data(ttl=50)
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
    st.write(f"🔄 **Auto-refresco:** cada {AUTO_REFRESH_SECONDS} segundos (1 min)")
    
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
    "Balance Hídrico / ET₀",
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
    st.markdown("### 💧 Balance Hídrico y Evapotranspiración (ET₀ / ETc)")

    # Detección inteligente de elevación a partir de los datos del sensor barométrico/altimétrico
    elev_default = ELEVACION_ESTACION
    if "bmp_alt" in df.columns:
        alt_val = pd.to_numeric(df["bmp_alt"], errors="coerce").dropna()
        if not alt_val.empty and alt_val.median() > 0:
            elev_default = int(round(alt_val.median()))

    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
    with col_cfg1:
        metodo_et = st.selectbox(
            "Método de cálculo de ET₀",
            ["FAO Penman-Monteith", "Hargreaves-Samani"],
            help="FAO Penman-Monteith (estándar FAO-56) integra radiación estimada, temperatura, humedad, viento y presión. Hargreaves usa temperatura y radiación extraterrestre.",
        )
    with col_cfg2:
        opciones_cultivo = {
            "🌱 Referencia / Pasto (Kc = 1.00)": 1.00,
            "☕ Café en producción (Kc = 1.05)": 1.05,
            "🌽 Maíz grano (Kc = 1.15)": 1.15,
            "🌾 Frijol (Kc = 1.10)": 1.10,
            "🎋 Caña de azúcar (Kc = 1.25)": 1.25,
            "🍅 Tomate / Hortalizas (Kc = 1.05)": 1.05,
            "🍌 Plátano / Banano (Kc = 1.10)": 1.10,
            "✏️ Coeficiente Personalizado": None,
        }
        cultivo_sel = st.selectbox(
            "Cultivo / Coeficiente Kc",
            options=list(opciones_cultivo.keys()),
            help="El coeficiente Kc ajusta la evapotranspiración de referencia (ET₀) a las necesidades específicas del cultivo: ETc = ET₀ × Kc.",
        )
        if opciones_cultivo[cultivo_sel] is not None:
            kc_valor = opciones_cultivo[cultivo_sel]
        else:
            kc_valor = st.slider("Kc personalizado", min_value=0.20, max_value=1.60, value=1.00, step=0.05)

    with col_cfg3:
        lat_input = st.number_input(
            "Latitud (°N)", value=LATITUD_ESTACION, format="%.2f",
            help="Latitud geográfica de la estación.",
        )
    with col_cfg4:
        elev_input = st.number_input(
            "Elevación (m s.n.m.)", value=elev_default, step=10,
            help=f"Elevación sobre el nivel del mar. Detectado por barómetro: ~{elev_default} m.",
        )

    metodo_key = "penman_monteith" if "Penman" in metodo_et else "hargreaves"
    resultado_bh = calcular_balance_hidrico(
        df, metodo=metodo_key, latitud=lat_input, elevacion=elev_input, kc=kc_valor
    )

    if resultado_bh and pd.notna(resultado_bh["et0_usado"]):
        et0_val = resultado_bh["et0_usado"]
        etc_val = resultado_bh["etc_usado"]
        precip_val = resultado_bh["precipitacion"]
        bal_val = resultado_bh["balance"]

        # ── Métricas principales ──
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ET₀ (Referencia)", f"{et0_val:.2f} mm/día")
        m2.metric(f"ETc (Kc={kc_valor:.2f})", f"{etc_val:.2f} mm/día")
        m3.metric("Precipitación Validada", f"{precip_val:.2f} mm")

        if pd.notna(bal_val):
            m4.metric(
                "Balance Hídrico (P − ETc)",
                f"{bal_val:+.2f} mm",
                delta="Superávit" if bal_val >= 0 else "Déficit",
                delta_color="normal" if bal_val >= 0 else "inverse",
            )
        else:
            m4.metric("Balance Hídrico", "N/A")

        riego_necesario_mm = max(0.0, -bal_val) if pd.notna(bal_val) else 0.0
        m5.metric("Lámina de Riego Neta", f"{riego_necesario_mm:.2f} mm")

        # ── Diagnóstico Agronómico ──
        if bal_val >= 0:
            st.success(
                f"💧 **Superávit hídrico:** +{bal_val:.2f} mm (+{bal_val * 10:.1f} m³/ha). "
                "Las precipitaciones cubrieron completamente la demanda hídrica diaria del cultivo. "
                "No se requiere riego el día de hoy."
            )
        else:
            vol_riego_m3 = abs(bal_val) * 10  # 1 mm = 10 m3/ha
            st.warning(
                f"☀️ **Déficit hídrico:** {bal_val:.2f} mm. "
                f"La demanda atmosférica y del cultivo superó a la precipitación. "
                f"**Riego neto recomendado:** {abs(bal_val):.2f} mm ({vol_riego_m3:.1f} m³/ha) para reponer la humedad del suelo."
            )

        # Comparar ambos métodos
        if pd.notna(resultado_bh["et0_penman_monteith"]):
            st.caption(
                f"📊 **Hargreaves-Samani:** {resultado_bh['et0_hargreaves']:.2f} mm/día  |  "
                f"**Penman-Monteith FAO-56:** {resultado_bh['et0_penman_monteith']:.2f} mm/día  |  "
                f"**Radiación extraterrestre (Ra):** {resultado_bh['ra']:.1f} MJ/m²/día  |  "
                f"**Insolación astronómica (N):** {resultado_bh['N']:.1f} horas"
            )

        df_bal = resultado_bh["df_balance"]

        # ── Gráfico 1: Precipitación acumulada vs ETc acumulada ──
        fig_bh, ax_bh = plt.subplots(figsize=(12, 4.8))
        ax_bh.plot(
            df_bal["fecha_hora"], df_bal["PP_acum"],
            linewidth=2.2, color="tab:blue", label="Precipitación acumulada (P) [mm]",
        )
        ax_bh.plot(
            df_bal["fecha_hora"], df_bal["ETc_acum"],
            linewidth=2.2, color="tab:orange", linestyle="--",
            label=f"ETc acumulada ({cultivo_sel.split('(')[0].strip()}) [mm]",
        )
        if kc_valor != 1.0:
            ax_bh.plot(
                df_bal["fecha_hora"], df_bal["ET0_acum"],
                linewidth=1.2, color="gray", linestyle=":",
                label=f"ET₀ de referencia [mm]",
            )
        ax_bh.fill_between(
            df_bal["fecha_hora"], df_bal["PP_acum"], df_bal["ETc_acum"],
            where=df_bal["PP_acum"] >= df_bal["ETc_acum"],
            alpha=0.25, color="tab:blue", label="Superávit hídrico (P ≥ ETc)",
        )
        ax_bh.fill_between(
            df_bal["fecha_hora"], df_bal["PP_acum"], df_bal["ETc_acum"],
            where=df_bal["PP_acum"] < df_bal["ETc_acum"],
            alpha=0.25, color="tab:red", label="Déficit hídrico (P < ETc)",
        )
        ax_bh.set_title(f"Dinámica del Balance Hídrico Acumulado — Precipitación vs ETc ({metodo_et})")
        ax_bh.set_xlabel("Hora local (El Salvador)")
        ax_bh.set_ylabel("Lámina de agua (mm)")
        ax_bh.legend(loc="upper left", fontsize=9)
        ax_bh.axhline(y=0, color="black", linewidth=0.5)
        dar_formato_eje_hora(ax_bh, df_bal, fecha_iso)
        fig_bh.autofmt_xdate()
        fig_bh.tight_layout()
        st.pyplot(fig_bh)

        # ── Gráficos secundarios en 2 columnas ──
        col_g1, col_g2 = st.columns([1.6, 1])

        with col_g1:
            fig_bn, ax_bn = plt.subplots(figsize=(8, 4))
            ax_bn.fill_between(
                df_bal["fecha_hora"], 0, df_bal["Balance_acum"],
                where=df_bal["Balance_acum"] >= 0, alpha=0.35,
                color="tab:blue", label="Superávit",
            )
            ax_bn.fill_between(
                df_bal["fecha_hora"], 0, df_bal["Balance_acum"],
                where=df_bal["Balance_acum"] < 0, alpha=0.35,
                color="tab:red", label="Déficit",
            )
            ax_bn.plot(
                df_bal["fecha_hora"], df_bal["Balance_acum"],
                color="navy", linewidth=1.5, label="Balance neto acumulado",
            )
            ax_bn.axhline(y=0, color="gray", linewidth=1, linestyle="--")
            ax_bn.set_title("Evolución Horaria del Balance Neto Acumulado (P − ETc)")
            ax_bn.set_xlabel("Hora local")
            ax_bn.set_ylabel("Balance (mm)")
            ax_bn.legend(loc="best", fontsize=8)
            dar_formato_eje_hora(ax_bn, df_bal, fecha_iso)
            fig_bn.autofmt_xdate()
            fig_bn.tight_layout()
            st.pyplot(fig_bn)

        with col_g2:
            fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
            cat = ["P Validada", "ET₀", "ETc", "Balance"]
            vals = [precip_val, et0_val, etc_val, bal_val]
            cols = [
                "tab:blue",
                "gray",
                "tab:orange",
                "tab:green" if bal_val >= 0 else "tab:red",
            ]
            bars = ax_bar.bar(cat, vals, color=cols, width=0.55, edgecolor="black", linewidth=0.8)
            for bar, val in zip(bars, vals):
                y_pos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.25
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:+.2f}" if bar == bars[-1] else f"{val:.2f}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontweight="bold", fontsize=9,
                )
            ax_bar.axhline(y=0, color="black", linewidth=0.8)
            ax_bar.set_ylabel("Lámina (mm / día)")
            ax_bar.set_title("Balance Diario Consolidado")
            val_max = max(max(vals), 0.5)
            val_min = min(min(vals), 0.0)
            ax_bar.set_ylim(val_min * 1.3 - 0.2, val_max * 1.3 + 0.2)
            fig_bar.tight_layout()
            st.pyplot(fig_bar)

        # ── Exportación de la serie de Balance Hídrico ──
        csv_bh = df_bal.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar serie temporal de Balance Hídrico (CSV)",
            data=csv_bh,
            file_name=f"3DPaws_Balance_Hidrico_{fecha_iso}.csv",
            mime="text/csv",
        )

        # ── Información metodológica y científica ──
        with st.expander("📖 Fundamentos Científicos y Metodología (FAO-56)"):
            st.markdown(f"""
**Condiciones del día ({fecha_iso}):**

| Variable | Valor |
|---|---|
| Día del año (Día Juliano) | {resultado_bh['doy']} |
| Radiación extraterrestre ($R_a$) | {resultado_bh['ra']:.2f} MJ/m²/día |
| Insolación astronómica máxima ($N$) | {resultado_bh['N']:.1f} horas |
| Temperatura mínima ($T_{{min}}$) | {resultado_bh['tmin']:.1f} °C |
| Temperatura máxima ($T_{{max}}$) | {resultado_bh['tmax']:.1f} °C |
| Temperatura media ($T_{{mean}}$) | {resultado_bh['tmean']:.1f} °C |
| Latitud geográfica | {lat_input:.2f}°N |
| Elevación sobre el nivel del mar | {elev_input} m |
| Coeficiente del cultivo ($K_c$) | {kc_valor:.2f} ({cultivo_sel}) |

---
""")
            st.markdown(r"""
### 1. Método Estándar FAO-56 Penman-Monteith
Recomendado mundialmente por la FAO y la OMM como el método físico de mayor exactitud:

$$ET_0 = \frac{0.408 \, \Delta \, (R_n - G) + \gamma \, \frac{900}{T+273} \, u_2 \, (e_s - e_a)}{\Delta + \gamma \, (1 + 0.34 \, u_2)}$$

Donde:
- $R_n$: Radiación neta en la superficie del cultivo ($MJ \cdot m^{-2} \cdot d^{-1}$).
- $G$: Flujo de calor en el suelo (≈ 0 a escala diaria).
- $T$: Temperatura media del aire a 2 m (°C).
- $u_2$: Velocidad del viento a 2 m de altura ($m/s$).
- $e_s - e_a$: Déficit de presión de vapor de saturación ($kPa$).
- $\Delta$: Pendiente de la curva de presión de saturación de vapor ($kPa/°C$).
- $\gamma$: Constante psicrométrica ($kPa/°C$).

### 2. Método Hargreaves-Samani (1985)
Método empírico alternativo de alta confiabilidad cuando no se dispone de viento o humedad completos:

$$ET_0 = 0.0023 \times (T_{mean} + 17.8) \times (T_{max} - T_{min})^{0.5} \times R_a$$

### 3. Evapotranspiración del Cultivo ($ET_c$) y Balance Hídrico
$$ET_c = K_c \times ET_0$$

$$Balance = P - ET_c$$

- **$Balance > 0$ (Superávit):** Hay exceso de humedad en el suelo; posible percolación o escorrentía superficial.
- **$Balance < 0$ (Déficit):** El cultivo consume más agua de la aportada por la lluvia. Requiere riego de reposición equivalente a $|Balance| \times 10 \text{ m}^3/\text{ha}$.
""")
    else:
        st.warning(
            "No hay datos suficientes de temperatura para calcular la evapotranspiración. "
            "Se requiere al menos la columna `mcp9808` con valores válidos."
        )

with tabs[7]:
    mostrar = st.multiselect("Variables a mostrar", options=list(df.columns), default=list(df.columns)[:10])
    st.dataframe(df[mostrar], width="stretch", height=450)

st.subheader("Exportación de datos")
st.download_button(
    "⬇️ Descargar CSV limpio",
    data=convert_df_to_csv(df),
    file_name=f"3DPaws_ElSalvador_{fecha_iso}.csv",
    mime="text/csv",
)
