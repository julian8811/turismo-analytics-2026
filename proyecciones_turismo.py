# -*- coding: utf-8 -*-
"""Proyecciones de turismo Colombia 2026-2027.
Fuente: DANE, Encuesta Mensual de Alojamiento (EMA), anexos agosto 2026
https://www.dane.gov.co/files/operaciones/EMA/anex-EMA-ago2026.xlsx
Modelos: SARIMA, Prophet, LightGBM, Ensamble (pesos 1/MAPE), Chronos-Bolt-Small zero-shot, SARIMAX con TRM.
TRM: Superfinanciera vía datos.gov.co (dataset 32sa-8pi3), promedio mensual.
Google Trends (pytrends): falló con HTTP 429 el 2 nov 2026; no se usa.
"""
import warnings, logging; warnings.filterwarnings("ignore"); logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
import re
import numpy as np, pandas as pd, openpyxl
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import lightgbm as lgb

MESES=["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
wb=openpyxl.load_workbook("ema.xlsx",read_only=True,data_only=True)

def leer(hoja, cols, fila_hdr_txt="Año"):
    rows=list(wb[hoja].iter_rows(values_only=True)); out=[]; y=None
    for r in rows:
        if r[0] is not None:
            mm=re.match(r"\s*(\d{4})",str(r[0]))
            if mm and int(mm.group(1))>1990: y=int(mm.group(1))
        if y and isinstance(r[1],str) and r[1].strip() in MESES and r[cols[0]] is not None:
            out.append([pd.Timestamp(y,MESES.index(r[1].strip())+1,1)]+[float(r[c]) for c in cols])
    return out

o=pd.DataFrame(leer("4.2 Porc Mens Ocupación.reg",[2,8]),columns=["fecha","ocup_nacional","ocup_antioquia"]).set_index("fecha")
i=pd.DataFrame(leer("8.2 Ind empalmados.MMH",[3]),columns=["fecha","ingresos_reales_idx"]).set_index("fecha")
i=i[i.index>="2015-01-01"]
SERIES={"Ocupación hotelera nacional (%)":o["ocup_nacional"],
        "Ocupación hotelera Antioquia (%)":o["ocup_antioquia"],
        "Índice de ingresos reales de alojamiento, nacional (2019=100)":i["ingresos_reales_idx"]}
for s in SERIES.values(): s.index.freq="MS"

def covid(idx):  # choque COVID: mar-2020 a jun-2021 = 1; jul-dic 2021 recuperación = 0.5
    d=pd.Series(0.0,index=idx); d[(idx>="2020-03-01")&(idx<="2021-06-01")]=1.0; d[(idx>="2021-07-01")&(idx<="2021-12-01")]=0.5
    return d.values.reshape(-1,1)

def f_sarima(y,h):
    fidx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(),periods=h,freq="MS")
    m=SARIMAX(y,exog=covid(y.index),order=(1,0,1),seasonal_order=(1,1,0,12),trend="c").fit(disp=False)
    f=m.get_forecast(h,exog=covid(fidx)); ci=f.conf_int(alpha=0.2)
    return pd.DataFrame({"pred":f.predicted_mean.values,"lo":ci.iloc[:,0].values,"hi":ci.iloc[:,1].values},index=fidx)

def f_prophet(y,h):
    df=pd.DataFrame({"ds":y.index,"y":y.values,"covid":covid(y.index).ravel()})
    m=Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False,interval_width=0.8)
    m.add_regressor("covid"); m.fit(df)
    fidx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(),periods=h,freq="MS")
    fut=pd.DataFrame({"ds":fidx,"covid":covid(fidx).ravel()}); p=m.predict(fut)
    return pd.DataFrame({"pred":p.yhat.values,"lo":p.yhat_lower.values,"hi":p.yhat_upper.values},index=fidx)

LAGS=[1,2,3,12]
def feats(hist,t):
    return [hist[-l] for l in LAGS]+[t.month,covid(pd.DatetimeIndex([t]))[0,0]]
def f_lgbm(y,h):
    v=list(y.values); X=[];Y=[]
    for k in range(12,len(v)): X.append(feats(v[:k],y.index[k])); Y.append(v[k])
    m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=8,min_child_samples=5,verbose=-1).fit(np.array(X),Y)
    res=np.array(Y)-m.predict(np.array(X)); q10,q90=np.quantile(res,[0.1,0.9])
    fidx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(),periods=h,freq="MS"); hist=v[:]; P=[]
    for t in fidx: p=m.predict(np.array([feats(hist,t)]))[0]; P.append(p); hist.append(p)
    P=np.array(P); w=np.sqrt(np.arange(1,h+1))  # banda crece con el horizonte
    return pd.DataFrame({"pred":P,"lo":P+q10*w,"hi":P+q90*w},index=fidx)


import json, torch
from chronos import BaseChronosPipeline
# ---- TRM mensual (datos.gov.co 32sa-8pi3, Superfinanciera) ----
trm=pd.Series({pd.Timestamp(d["m"][:10]):float(d["trm"]) for d in json.load(open("trm.json"))}).sort_index()
TRM_ULT=trm.index[-1]; TRM_VAL=trm.iloc[-1]
def trm_x(idx):  # supuesto horizonte: TRM plana en el último promedio mensual observado (oct 2026)
    v=trm.reindex(idx); v[idx>TRM_ULT]=TRM_VAL
    return np.log(v.values)
def f_sarimax_trm(y,h):
    fidx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(),periods=h,freq="MS")
    X=lambda ix: np.column_stack([covid(ix).ravel(),trm_x(ix)])
    m=SARIMAX(y,exog=X(y.index),order=(1,0,1),seasonal_order=(1,1,0,12),trend="c").fit(disp=False)
    f=m.get_forecast(h,exog=X(fidx)); ci=f.conf_int(alpha=0.2)
    return pd.DataFrame({"pred":f.predicted_mean.values,"lo":ci.iloc[:,0].values,"hi":ci.iloc[:,1].values},index=fidx)
CH=BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small",device_map="cpu",torch_dtype=torch.float32)
def f_chronos(y,h):
    fidx=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(),periods=h,freq="MS")
    q,_=CH.predict_quantiles(inputs=torch.tensor(y.values,dtype=torch.float32),prediction_length=h,quantile_levels=[0.1,0.5,0.9])
    q=q[0].numpy(); return pd.DataFrame({"pred":q[:,1],"lo":q[:,0],"hi":q[:,2]},index=fidx)
BASE={"SARIMA":f_sarima,"Prophet":f_prophet,"LightGBM":f_lgbm}
MODELOS=dict(BASE,**{"Chronos-Bolt":f_chronos,"SARIMAX+TRM":f_sarimax_trm})
H_TEST=12
mape_=lambda a,p: float(np.mean(np.abs((a-p)/a))*100)
metricas=[];pron={};resumen=[];pesos={}
for nom,y in SERIES.items():
    tr,te=y[:-H_TEST],y[-H_TEST:]; res={}
    for mn,fn in MODELOS.items(): res[mn]=fn(tr,H_TEST)["pred"].values
    mp={mn:mape_(te.values,p) for mn,p in res.items()}
    w={k:(1/mp[k]) for k in BASE}; t=sum(w.values()); w={k:v/t for k,v in w.items()}; pesos[nom]=w
    res["Ensamble"]=sum(w[k]*res[k] for k in BASE)
    for mn,p in res.items():
        metricas.append([nom,mn,f"{tr.index[0]:%Y-%m} a {tr.index[-1]:%Y-%m}",f"{te.index[0]:%Y-%m} a {te.index[-1]:%Y-%m}",round(mape_(te.values,p),2),round(float(np.sqrt(np.mean((te.values-p)**2))),2)])
    best=min(res,key=lambda k: mape_(te.values,res[k]))
    h=(2027-y.index[-1].year)*12+(12-y.index[-1].month)
    if best=="Ensamble":
        fs={k:BASE[k](y,h) for k in BASE}; f=sum(w[k]*fs[k] for k in BASE)
    else: f=MODELOS[best](y,h)
    f["escenario_base"]=f["pred"]; f["escenario_optimista"]=f["hi"]; f["escenario_pesimista"]=f["lo"]
    pron[nom]=(y,f,(best,))
    for a in [2025,2026,2027]:
        real=y[y.index.year==a]; fc=f[f.index.year==a]
        resumen.append([nom,a,best,round(pd.concat([real,fc["pred"]]).mean(),2),round(pd.concat([real,fc["lo"]]).mean(),2),round(pd.concat([real,fc["hi"]]).mean(),2),len(real)])
M=pd.DataFrame(metricas,columns=["Serie","Modelo","Entrenamiento","Prueba","MAPE %","RMSE"])
comp=M.pivot(index="Serie",columns="Modelo",values="MAPE %"); comp["Ganador"]=comp.idxmin(axis=1)
with pd.ExcelWriter("proyecciones_turismo.xlsx") as xw:
    pd.DataFrame([["Fuente series","DANE, Encuesta Mensual de Alojamiento (EMA), anexos agosto 2026 (publicado 16 oct 2026). Datos ene 2005/2015 a ago 2026"],
        ["URL","https://www.dane.gov.co/files/operaciones/EMA/anex-EMA-ago2026.xlsx"],
        ["Fuente TRM","Superintendencia Financiera, TRM diaria vía datos.gov.co (https://www.datos.gov.co/resource/32sa-8pi3.json), promedio mensual ene 2015 a oct 2026"],
        ["Google Trends","pytrends 'turismo Colombia' y 'Medellín': Google respondió HTTP 429 (TooManyRequests) en 2 intentos el 2 nov 2026. No se incluyó"],
        ["Modelos","SARIMA(1,0,1)(1,1,0)12+COVID; Prophet+COVID; LightGBM rezagos 1,2,3,12; Ensamble ponderado 1/MAPE de los tres anteriores; Chronos-Bolt-Small (amazon/chronos-bolt-small) zero-shot CPU, cuantiles 0,1/0,5/0,9; SARIMAX igual a SARIMA + log(TRM)"],
        ["Backtest","Últimos 12 meses (sep 2025 a ago 2026). En SARIMAX+TRM el backtest usa la TRM observada (ex post), lo que lo favorece"],
        ["Supuesto exógeno","TRM: sep y oct 2026 observadas; nov 2026 a dic 2027 plana en "+f"{TRM_VAL:,.0f}"+" COP/USD (promedio oct 2026). COVID = 0 en el horizonte"],
        ["Pesos ensamble",json.dumps({k:{m:round(v,3) for m,v in w.items()} for k,w in pesos.items()},ensure_ascii=False)],
        ["Escenarios","Base = mejor modelo por MAPE; optimista/pesimista = banda 80 % (ensamble: promedio ponderado de bandas)"],
        ["Limitaciones","Cifras DANE 2026 provisionales. Sin visitantes, pasajeros ni Trends"]],
        columns=["Campo","Detalle"]).to_excel(xw,sheet_name="Metodología",index=False)
    pd.concat(SERIES,axis=1).join(trm.rename("TRM promedio mensual"),how="outer").to_excel(xw,sheet_name="Series históricas")
    comp.to_excel(xw,sheet_name="Comparación MAPE")
    M.to_excel(xw,sheet_name="Métricas",index=False)
    pd.DataFrame(resumen,columns=["Serie","Año","Mejor modelo","Promedio anual base","Promedio pesimista","Promedio optimista","Meses reales incluidos"]).to_excel(xw,sheet_name="Resumen anual",index=False)
    for k,(nom,(y,f,b)) in enumerate(pron.items()): f.round(2).to_excel(xw,sheet_name=f"Pronóstico {k+1}")
pd.set_option("display.width",250); print(comp); print(pd.DataFrame(resumen)); print(pesos)
json.dump({nom:{"hist":{f"{d:%Y-%m}":round(v,2) for d,v in y[y.index>="2025-01-01"].items()},
 "fc":{f"{d:%Y-%m}":[round(r.pred,2),round(r.lo,2),round(r.hi,2)] for d,r in f.iterrows()},"best":b[0]} for nom,(y,f,b) in pron.items()},open("proy.json","w"),ensure_ascii=False)
