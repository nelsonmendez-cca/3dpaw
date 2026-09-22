
# modulo_3dpaws.py
# Dashboard para analizar archivos .dat de estaciones atmosféricas 3DPaws
#
# Ejecutar:
#   pip install streamlit pandas numpy matplotlib
#   streamlit run modulo_3dpaws.py
#
# El archivo .dat esperado tiene columnas separadas por espacios, por ejemplo:
# year mon day hour min int bmp_temp bmp_pres bmp_slp bmp_alt
# bme2_hum hum_temp hum_hum mcp9808 tipping vis_light ir_light
# uv_light wind_dir wind_speed

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(
    page_title="3DPaws | Estación meteorológica",
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

    # Fecha/hora local del registro.
    df["fecha_hora"] = pd.to_datetime(dict(year=df["year"], month=df["mon"], day=df["day"], hour=df["hour"], minute=df["min"]), errors="coerce") - pd.Timedelta(hours=6)

    # Valores usados por 3DPaws para indicar dato no disponible.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] <= -999, col] = np.nan

    df = df.sort_values("fecha_hora").reset_index(drop=True)

    # Variables derivadas útiles.
    if "hum_temp" in df and "mcp9808" in df:
        df["dif_temp_sensores"] = df["hum_temp"] - df["mcp9808"]

    return df


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


def calcular_hr_psicrometrica(df):
    if "hum_temp" not in df.columns or "mcp9808" not in df.columns: return pd.Series(np.nan, index=df.index)
    seco=pd.to_numeric(df["mcp9808"], errors="coerce"); hum=pd.to_numeric(df["hum_temp"], errors="coerce"); P=1013.25
    es=lambda T: 6.112*np.exp((17.62*T)/(243.12+T))
    e=es(hum)-0.00066*P*(seco-hum)
    return (100*e/es(seco)).clip(0,100)

def grafico_psicrometro(df):
    fig,ax=plt.subplots(figsize=(12,4.8)); ax.plot(df["fecha_hora"],df["mcp9808"],label="Bulbo seco"); ax.plot(df["fecha_hora"],df["hum_temp"],label="Bulbo húmedo"); ax.set_ylabel("Temperatura (°C)"); ax.grid(True,alpha=.25); ax.set_title("Psicrómetro y humedad relativa calculada")
    ax2=ax.twinx(); ax2.plot(df["fecha_hora"],df["HR_psicrometrica"],"--",label="HR psicrométrica"); ax2.set_ylabel("Humedad relativa (%)"); ax2.set_ylim(0,100); l1,a1=ax.get_legend_handles_labels(); l2,a2=ax2.get_legend_handles_labels(); ax.legend(l1+l2,a1+a2,loc="upper left"); fig.autofmt_xdate(); fig.tight_layout(); return fig

def dibujar_pluviometro(valor_mm, capacidad=100):
    fig,ax=plt.subplots(figsize=(4,6)); v=max(0,float(valor_mm)); n=min(v/capacidad,1); x0,x1,y0,y1=.3,.7,.08,.88
    ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,linewidth=2)); ax.add_patch(plt.Rectangle((x0,y0),x1-x0,(y1-y0)*n,alpha=.55)); ax.plot([x0-.04,x1+.04],[y1,y1],linewidth=3)
    for mm in range(0,capacidad+1,10):
        y=y0+(y1-y0)*mm/capacidad; ln=.07 if mm%20==0 else .045; ax.plot([x1,x1+ln],[y,y]);
        if mm%20==0: ax.text(x1+ln+.025,y,str(mm),va="center",fontsize=9)
    ax.text(.5,.96,"PLUVIÓGRAFO 3DPaws",ha="center",fontsize=13,fontweight="bold"); ax.text(.5,.02,f"Precipitación: {v:.1f} mm",ha="center",fontsize=12,fontweight="bold"); ax.set_xlim(.18,.95); ax.set_ylim(0,1.03); ax.axis("off"); fig.tight_layout(); return fig

def grafico_linea(df, columnas, titulo, ylabel):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for col in columnas:
        if col in df.columns:
            ax.plot(df["fecha_hora"], df[col], linewidth=1.3, label=col)
    ax.set_title(titulo)
    ax.set_xlabel("Fecha y hora")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def grafico_viento(df):
    w = df[["wind_dir", "wind_speed"]].dropna().copy()
    if w.empty:
        return None

    # Rosa de viento simple por sectores de 22.5°.
    sectores = np.arange(0, 360, 22.5)
    nombres = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
               "S","SSO","SO","OSO","O","ONO","NO","NNO"]

    # Solo observaciones con velocidad > 0.
    w = w[w["wind_speed"] > 0].copy()
    if w.empty:
        return None

    idx = ((w["wind_dir"] + 11.25) // 22.5).astype(int) % 16
    frecuencias = idx.value_counts().reindex(range(16), fill_value=0).sort_index()
    porcentajes = frecuencias / frecuencias.sum() * 100

    theta = np.deg2rad(sectores)
    width = np.deg2rad(22.5)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.bar(theta, porcentajes.values, width=width, align="center", alpha=.75)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels(nombres)
    ax.set_title("Rosa de viento — frecuencia por dirección (%)", pad=20)
    ax.set_ylabel("%")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------
st.title("🌦️ 3DPaws — Analizador de estación meteorológica")
st.caption(
    "Visualización y control exploratorio de los datos registrados por "
    "la estación atmosférica de bajo costo."
)

archivo = st.file_uploader(
    "Carga un archivo .dat de 3DPaws",
    type=["dat", "txt"],
)

if archivo is None:
    st.info(
        "Carga el archivo recordings_YYYY_MM_DD.dat para comenzar. "
        "El módulo reconoce automáticamente las columnas del formato 3DPaws."
    )
    st.stop()

df = cargar_dat(archivo.getvalue())
df["HR_psicrometrica"] = calcular_hr_psicrometrica(df)

# ---------------------------------------------------------------------
# Información general
# ---------------------------------------------------------------------
st.subheader("Resumen del registro")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observaciones", f"{len(df):,}")
c2.metric("Inicio", df["fecha_hora"].min().strftime("%d/%m %H:%M"))
c3.metric("Fin", df["fecha_hora"].max().strftime("%d/%m %H:%M"))
duracion = df["fecha_hora"].max() - df["fecha_hora"].min()
c4.metric("Duración", str(duracion).split(".")[0])

st.write(
    f"**Periodo:** {rango_texto(df)}  |  "
    f"**Intervalo nominal:** {df['int'].dropna().mode().iloc[0] if not df['int'].dropna().empty else 'N/D'}"
)

# ---------------------------------------------------------------------
# Control de calidad
# ---------------------------------------------------------------------
st.subheader("Control de calidad")

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
        "% faltantes": round(faltantes / total * 100, 2),
    })

qc_df = pd.DataFrame(qc).sort_values("% faltantes", ascending=False)
st.dataframe(qc_df, use_container_width=True, hide_index=True)

problemas = qc_df[qc_df["Faltantes"] > 0]
if not problemas.empty:
    st.warning(
        "Hay variables con valores ausentes o marcadores de dato inválido. "
        "Los -999.99 fueron convertidos a NaN y no se utilizan en los cálculos."
    )

# ---------------------------------------------------------------------
# Variables meteorológicas
# ---------------------------------------------------------------------
st.subheader("Variables meteorológicas")

tabs = st.tabs([
    "Temperatura",
    "Humedad",
    "Presión",
    "Radiación / luz",
    "Viento",
    "Precipitación",
    "Datos",
])

with tabs[0]:
    st.pyplot(grafico_psicrometro(df), use_container_width=True)
    st.info("La HR se estima psicrométricamente usando mcp9808 como bulbo seco, hum_temp como bulbo húmedo y 1013.25 hPa como presión estándar.")
    hr=df["HR_psicrometrica"].dropna()
    if not hr.empty:
        c1,c2,c3=st.columns(3); c1.metric("HR media",f"{hr.mean():.1f} %"); c2.metric("HR mínima",f"{hr.min():.1f} %"); c3.metric("HR máxima",f"{hr.max():.1f} %")

with tabs[1]:
    if "hum_hum" in df:
        st.pyplot(
            grafico_linea(df, ["hum_hum"], "Humedad relativa", "Humedad (%)"),
            use_container_width=True,
        )
        st.write(
            f"**Humedad media:** {df['hum_hum'].mean():.2f}%  |  "
            f"**mínima:** {df['hum_hum'].min():.2f}%  |  "
            f"**máxima:** {df['hum_hum'].max():.2f}%"
        )

with tabs[2]:
    cols = [c for c in ["bmp_pres", "bmp_slp"] if c in df]
    st.pyplot(
        grafico_linea(
            df, cols,
            "Presión registrada",
            "Presión (unidad del archivo)",
        ),
        use_container_width=True,
    )
    st.warning(
        "Nota: `bmp_pres` y `bmp_slp` deben interpretarse según la "
        "calibración/configuración específica del firmware 3DPaws. "
        "El módulo no cambia las unidades originales."
    )

with tabs[3]:
    cols = [c for c in ["vis_light", "ir_light", "uv_light"] if c in df]
    st.pyplot(
        grafico_linea(
            df, cols,
            "Sensores de luz y radiación",
            "Lectura del sensor",
        ),
        use_container_width=True,
    )

with tabs[4]:
    c1, c2 = st.columns(2)

    with c1:
        st.pyplot(grafico_viento(df), use_container_width=True)

    with c2:
        if "wind_speed" in df:
            st.pyplot(
                grafico_linea(
                    df,
                    ["wind_speed"],
                    "Velocidad del viento",
                    "Velocidad (unidad del archivo)",
                ),
                use_container_width=True,
            )

            viento = df["wind_speed"].dropna()
            if not viento.empty:
                st.write(
                    f"**Media:** {viento.mean():.3f}  |  "
                    f"**Máxima:** {viento.max():.3f}"
                )

with tabs[5]:
    if "tipping" in df:
        st.markdown("### Pluviógrafo / tipping bucket")
        mm_por_vuelco=st.number_input("Milímetros por vuelco",min_value=0.01,value=0.254,step=0.001,format="%.3f")
        precip=float(pd.to_numeric(df["tipping"],errors="coerce").max())*mm_por_vuelco
        c1,c2=st.columns([1,2])
        with c1: st.pyplot(dibujar_pluviometro(precip,100),use_container_width=True)
        with c2:
            fig,ax=plt.subplots(figsize=(10,4.8)); ax.plot(df["fecha_hora"],pd.to_numeric(df["tipping"],errors="coerce")*mm_por_vuelco); ax.set_title("Precipitación acumulada"); ax.set_xlabel("Fecha y hora"); ax.set_ylabel("mm"); ax.grid(True,alpha=.25); fig.autofmt_xdate(); fig.tight_layout(); st.pyplot(fig,use_container_width=True)
        st.metric("Precipitación acumulada",f"{precip:.2f} mm")
        st.warning("Se asume que tipping es un contador acumulado. Ajusta mm/vuelco a la calibración real del 3DPaws.")
    else: st.warning("El archivo no contiene la columna tipping.")

with tabs[6]:
    mostrar = st.multiselect(
        "Variables a mostrar",
        options=list(df.columns),
        default=[
            "fecha_hora",
            "bmp_temp",
            "bmp_pres",
            "bmp_slp",
            "hum_temp",
            "hum_hum",
            "mcp9808",
            "wind_dir",
            "wind_speed",
        ],
    )
    st.dataframe(df[mostrar], use_container_width=True, height=500)

# ---------------------------------------------------------------------
# Estadísticas
# ---------------------------------------------------------------------
st.subheader("Estadísticas")

variables = st.multiselect(
    "Selecciona variables para estadísticas",
    options=numeric_cols,
    default=[
        c for c in ["bmp_temp", "hum_temp", "hum_hum", "bmp_slp", "wind_speed"]
        if c in numeric_cols
    ],
)

stats = []
for col in variables:
    r = resumen_variable(df, col)
    if r:
        stats.append({
            "Variable": col,
            "N": r["n"],
            "Mínimo": r["min"],
            "Máximo": r["max"],
            "Media": r["media"],
            "Mediana": r["mediana"],
            "Desv. estándar": r["std"],
            "P05": r["p05"],
            "P95": r["p95"],
        })

if stats:
    st.dataframe(
        pd.DataFrame(stats).round(3),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------
# Descarga de datos limpios
# ---------------------------------------------------------------------
st.subheader("Exportación")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar CSV limpio",
    data=csv,
    file_name="3DPaws_datos_limpios.csv",
    mime="text/csv",
)
