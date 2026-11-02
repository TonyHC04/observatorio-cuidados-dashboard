"""Dashboard – Observatorio de Autonomía y Cuidados (Proyecto Analítico Integral).
Audiencia: dirección de Sembrando Esperanza para una Mañana Brillante, A.C.
Datos: los MISMOS del Anexo_Calculos_Observatorio_completo.xlsx (simulación sintética de 50 solicitudes,
histórico sintético de 18 meses y flujos financieros de las hojas 3-5). Solicitudes e histórico: carpeta data/.
"""
from pathlib import Path
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.express as px, streamlit as st

st.set_page_config(page_title="Observatorio de Autonomía y Cuidados", layout="wide")
DATA = Path(__file__).parent / "data"
sol = pd.read_csv(DATA / "solicitudes_sinteticas.csv", keep_default_na=False)
hist = pd.read_csv(DATA / "historico_sintetico.csv")

# ---- cifras de diagnóstico (hoja 1 del anexo / fichas del repositorio)
CANSANCIO, AFECT_LAB, N_ENASIC = 284, 192, 636
ENUT_M, ENUT_H = 23.77, 13.13
ENIGH_DIF = 51532.24 / 56059.63 - 1
# ---- finanzas (hojas 3, 4 y 5 del anexo)
INV, FLUJOS, TASA = 47300, [55000, 60000, 65000], 0.10
VAN = {"Pesimista": -25036.66, "Realista": 101122.24, "Optimista": 209470.47}  # hoja 4 (flujos con factor 0.15 / 1 / 1.73)
PROB = {"Pesimista": .20, "Realista": .60, "Optimista": .20}
ADOP = {"Pesimista": 30, "Realista": 60, "Optimista": 90}
BASE_DIAS = 4.6   # SUPUESTO sintético: tiempo del proceso manual (hoja 7)

# ---- filtros (slicers)
st.sidebar.header("Filtros de la simulación de matching")
zona = st.sidebar.multiselect("Zona", sorted(sol.Zona.unique()), default=sorted(sol.Zona.unique()))
tipo = st.sidebar.multiselect("Tipo de apoyo", sorted(sol.Tipo_Apoyo.unique()), default=sorted(sol.Tipo_Apoyo.unique()))
cans = st.sidebar.multiselect("Nivel de cansancio", ["Alto", "Medio", "Bajo"], default=["Alto", "Medio", "Bajo"])
acc = st.sidebar.multiselect("Requiere accesibilidad", ["Sí", "No"], default=["Sí", "No"])
f = sol[sol.Zona.isin(zona) & sol.Tipo_Apoyo.isin(tipo) & sol.Nivel_Cansancio.isin(cans) & sol.Requiere_Accesibilidad.isin(acc)]
n = len(f); vinc = (f.Estatus == "Vinculado").sum()
tasa_v = vinc / n if n else 0; t_med = f.Tiempo_Respuesta_Dias.mean() if n else 0
red = 1 - t_med / BASE_DIAS if n else 0

# ---- pronóstico (regresión lineal sobre los 18 meses; igual que la hoja 8)
x, y = hist.Mes.values, hist.Solicitudes.values
b, a = np.polyfit(x, y, 1); res = y - (a + b * x); se = np.sqrt((res ** 2).sum() / (len(x) - 2)); TCRIT = 2.120
fut = np.arange(19, 22); yh = a + b * fut
pi = TCRIT * se * np.sqrt(1 + 1 / len(x) + (fut - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())

# ---- finanzas
ve = sum(VAN[k] * PROB[k] for k in VAN)
acum = np.cumsum([-INV] + FLUJOS)

def sem(ok, warn): return "" if ok else ("" if warn else "")

st.title("Casi la mitad de las personas cuidadoras reporta cansancio: la vinculación de apoyos puede atender a 8 de cada 10 solicitudes")
st.caption("Audiencia: dirección de Sembrando Esperanza para una Mañana Brillante, A.C. · Diagnóstico: INEGI (ENASIC 2022, ENUT 2024, ENIGH 2024). "
           " Las solicitudes, el histórico y el desempeño del matching son DATOS SINTÉTICOS; el diagnóstico local aún no se levanta.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tasa de vinculación (simulada)", f"{tasa_v:.0%}", help="Vinculadas / solicitudes filtradas. Meta: 80 %.")
c1.write(sem(tasa_v >= .8, tasa_v >= .7) + " vs. meta 80 %")
c2.metric("Tiempo medio de respuesta", f"{t_med:.1f} días", f"{-red:.0%} vs. base {BASE_DIAS} d", delta_color="inverse")
c2.write(sem(red >= .3, red >= .2) + " vs. objetivo −30 %")
c3.metric("Cansancio atribuido al cuidado (ENASIC)", f"{CANSANCIO / N_ENASIC:.1%}", help="284 de 636 registros, sin ponderar.")
c3.write(sem(CANSANCIO / N_ENASIC < .3, CANSANCIO / N_ENASIC < .5) + " vs. umbral de alerta 50 %")
c4.metric("VAN esperado (3 años)", f"${ve:,.0f}", help="Probabilidades 20/60/20 de la hoja 4.")
c4.write(sem(ve > 0 and VAN['Realista'] > 0, ve > 0) + " vs. meta VAN > 0")

t1, t2, t3 = st.tabs(["1 · Diagnóstico", "2 · Matching y demanda", "3 · Decisión financiera"])
with t1:
    a1, a2 = st.columns(2)
    d = pd.DataFrame({"Indicador": ["Cansancio atribuido", "≥1 afectación laboral"], "%": [CANSANCIO / N_ENASIC * 100, AFECT_LAB / N_ENASIC * 100]})
    fig = px.bar(d, x="%", y="Indicador", orientation="h", text=d["%"].map("{:.1f}%".format), title="ENASIC 2022 (n=636, sin ponderar)")
    fig.update_layout(height=320, yaxis=dict(autorange="reversed")); a1.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Bar(x=["Mujeres", "Hombres"], y=[ENUT_M, ENUT_H], marker_color=["#1F4E78", "#9C6B30"], text=[f"{ENUT_M} h", f"{ENUT_H} h"]))
    fig.update_layout(title="ENUT 2024: horas/sem de cuidados especiales (con cuidados pasivos)", height=320); a2.plotly_chart(fig, width="stretch")
    st.info(f"Brecha mujeres/hombres: {ENUT_M / ENUT_H:.2f}× (misma variante). ENIGH 2024 (sin ponderar): ingreso mediano {ENIGH_DIF:.1%} en hogares con dificultad grave. "
            "Son contextos nacionales, no describen a las familias de la A.C.")
with t2:
    b1, b2 = st.columns(2)
    g = f.groupby(["Zona", "Estatus"]).size().reset_index(name="Solicitudes")
    fig = px.bar(g, x="Zona", y="Solicitudes", color="Estatus", barmode="stack", title="Resultado del matching por zona",
                 color_discrete_map={"Vinculado": "#2E8B57", "Sin Coincidencia": "#C00000"}); b1.plotly_chart(fig, width="stretch")
    m = f[f.Estatus != "Vinculado"].Motivo_Rechazo.value_counts().reset_index(); m.columns = ["Motivo", "Casos"]
    fig = px.bar(m, x="Casos", y="Motivo", orientation="h", title="Motivos de “sin coincidencia”"); b2.plotly_chart(fig, width="stretch")
    fig = px.box(f, x="Nivel_Cansancio", y="Tiempo_Respuesta_Dias", color="Tipo_Apoyo", category_orders={"Nivel_Cansancio": ["Alto", "Medio", "Bajo"]},
                 title="Tiempo de respuesta (días) por nivel de cansancio y tipo de apoyo"); st.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name="Histórico sintético", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=fut, y=yh, name="Pronóstico", mode="lines+markers", line=dict(color="#C00000")))
    fig.add_trace(go.Scatter(x=np.r_[fut, fut[::-1]], y=np.r_[yh + pi, (yh - pi)[::-1]], fill="toself", line=dict(width=0), name="IC 95 %", opacity=.25))
    fig.update_layout(title=f"Solicitudes mensuales: próximo mes {yh[0]:.0f} · próximos 3 meses {yh.sum():.0f} (regresión lineal, MAPE prueba 2.8 %)", xaxis_title="Mes", height=380)
    st.plotly_chart(fig, width="stretch")
with t3:
    tab = pd.DataFrame({k: {"Adopción": f"{ADOP[k]}%", "VAN": f"${VAN[k]:,.0f}", "Probabilidad": f"{PROB[k]:.0%}"} for k in VAN}); st.table(tab)
    d1, d2 = st.columns(2)
    fig = go.Figure(go.Bar(x=list(VAN), y=list(VAN.values()), marker_color=["#C00000", "#1F4E78", "#2E8B57"], text=[f"${v:,.0f}" for v in VAN.values()]))
    fig.update_layout(title=f"VAN por escenario · esperado ${ve:,.0f}", height=340); d1.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Scatter(x=[0, 1, 2, 3], y=acum, mode="lines+markers")); fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="Flujo acumulado, escenario realista (años)", xaxis_title="Año", height=340); d2.plotly_chart(fig, width="stretch")
    st.success(f"Recomendación: INVERTIR CONDICIONADO a validación institucional y dictamen LFPDPPP. Capital requerido ${INV:,.0f}.")
st.caption("Supuestos sin dato real: línea base de 4.6 días, motivos asignados a las no coincidencias y flujos de los escenarios pesimista/optimista.")
