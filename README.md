# Turismo Analytics 2026: Mundo, Colombia y Medellín

Proyecciones de turismo para Colombia y Antioquia de 2026 a 2027, con datos abiertos.

**Dashboard:** https://cursor.com/canvas/store-0515972d-5e6f-43ef-b1af-f35dcf3e6d7c/05d806a7-4584-452d-9d2e-1f999f75274a

## Archivos
- `proyecciones_turismo.py`: el script completo (lectura, modelos, backtest y exportación).
- `proyecciones_turismo.xlsx`: los resultados, con las hojas Metodología, Series históricas, Comparación MAPE, Métricas, Resumen anual y Pronóstico 1–3.
- `fuentes_turismo_2026.xlsx`: el catálogo de 55 fuentes de turismo 2026 (mundo, Colombia y Medellín).
- `ema.xlsx`: los anexos de la Encuesta Mensual de Alojamiento del DANE, agosto 2026.
- `trm.json`: la TRM promedio mensual de enero de 2015 a octubre de 2026.

## Fuentes
- **DANE, EMA:** anexos de agosto 2026, publicados el 16 oct 2026 (https://www.dane.gov.co/files/operaciones/EMA/anex-EMA-ago2026.xlsx).
  - Ocupación hotelera nacional y de Antioquia, de enero de 2019 a agosto de 2026.
  - Índice de ingresos reales de alojamiento (2019=100), de enero de 2015 a agosto de 2026.
- **TRM:** de la Superintendencia Financiera, vía datos.gov.co (https://www.datos.gov.co/resource/32sa-8pi3.json). Es el promedio mensual de los valores diarios.
- **Google Trends** (pytrends): no se incluyó, porque Google respondió con el error HTTP 429 el 2 nov 2026.

## Modelos
La prueba usa los últimos 12 meses reales (sep 2025 a ago 2026).
1. **SARIMA(1,0,1)(1,1,0)12**, con una variable para el choque COVID.
2. **Prophet**, con estacionalidad anual y el regresor COVID.
3. **LightGBM**, con rezagos 1, 2, 3 y 12, el mes y la variable COVID. Pronostica de forma recursiva.
4. **Ensamble ponderado** de los modelos 1 a 3, con pesos iguales a 1/MAPE.
5. **Chronos-Bolt-Small** (`amazon/chronos-bolt-small`), sin entrenamiento (zero-shot) y en CPU. Usa los cuantiles 0,1, 0,5 y 0,9.
6. **SARIMAX**: el mismo SARIMA más log(TRM). Para el horizonte se supone la TRM fija en 3.199 COP/USD, el promedio de oct 2026.

## MAPE en la prueba
| Serie | SARIMA | Prophet | LightGBM | Ensamble | Chronos | SARIMAX+TRM | Ganador |
|---|---|---|---|---|---|---|---|
| Ocupación nacional | 3,81 % | 2,43 % | 6,29 % | **1,87 %** | 3,93 % | 8,79 % | Ensamble |
| Ocupación Antioquia | 6,30 % | 3,43 % | 5,89 % | **3,08 %** | 8,43 % | 10,58 % | Ensamble |
| Ingresos reales | 7,00 % | 6,93 % | 6,61 % | **6,56 %** | 6,58 % | 11,23 % | Ensamble |

## Proyecciones (promedios anuales, ensamble, banda del 80 %)
| Serie | 2025 real | 2026 | 2027 base | 2027 pesimista | 2027 optimista |
|---|---|---|---|---|---|
| Ocupación nacional | 50,4 % | 48,9 % | 48,6 % | 42,6 % | 54,6 % |
| Ocupación Antioquia | 50,7 % | 48,4 % | 47,2 % | 42,0 % | 52,5 % |
| Ingresos reales (2019=100) | 97,4 | 91,2 | 93,4 | 81,0 | 105,7 |

El dato de 2026 combina 8 meses reales con el pronóstico. Las cifras del DANE de 2026 son provisionales, y ningún modelo anticipa choques nuevos.

## Uso
```bash
pip install -r requirements.txt
python proyecciones_turismo.py
```
