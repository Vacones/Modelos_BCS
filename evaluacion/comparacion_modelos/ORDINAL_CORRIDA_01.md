# CORAL ordinal: nueva corrida Mac de validación

**Revisión:** 12 de noviembre de 2026. **Carpetas privadas:** `resultados_ordinal/coral_holdout_v1` y `resultados_ordinal/coral_pipeline_val_v1`. **Partición evaluada:** `valid`; `test` no se evaluó. Esta ejecución conserva siete niveles BCS de 2.25 a 3.75 y no equivale al experimento ordinal v4 histórico, que fusionó 2.25 con 2.50.

## Procedencia

- Se entrenó con 239 fotos de 46 vacas en `train`; cuatro fotos con BCS ambiguo quedaron excluidas del predictor. La selección de checkpoint usó únicamente las 36 fotos de 10 vacas en `valid` y el **MAE continuo por vaca**.
- La cabeza alcanzó mejor MAE de validación por vaca **0.2650** en la época 9; el ajuste fino, **0.2646** en la época 1. Se eligió el ajuste fino. La diferencia de 0.0004 BCS es mínima y no demuestra una mejora real.
- El archivo de entrenamiento conserva en su registro las huellas de COCO y del código. El código actual coincide con esa huella. El checkpoint seleccionado tiene SHA-256 `ea1c09339761e93b16ca6320e651a1cb4a0f183070962ea5ea63623dd8991388`.
- Para el sistema completo se reutilizó el detector YOLO de la nueva corrida DINOv2, también usado por la nueva EfficientNet. Su SHA-256 es `28c95b485dba90aec93f4c90c3abbafbecbb8948c40062daad6dd71b27b0dd6b`. El recorte ordinal aplica 10% de margen a la caja `ass` predicha y rellena hasta 224 × 224 px.

## Resultados

| Evaluación | Fotos cubiertas | Vacas cubiertas | MAE categórico por foto cubierta | MAE continuo por foto cubierta | MAE continuo por vaca cubierta |
|---|---:|---:|---:|---:|---:|
| Recortes con cajas anotadas | 36/36 | 10/10 | 0.2847 | 0.2828 | 0.2646 |
| Sistema con cajas predichas | **24/36** | **8/10** | **0.2813** | **0.2875** | **0.2397** |

El sistema completo se abstuvo en 12 fotos: **9 asociaciones ambiguas** y **3 sin par válido**. Entre sus 24 predicciones categóricas hubo 7 exactas, 17 dentro de ±0.25 y 21 dentro de ±0.50. Con todas las fotos elegibles como denominador, son **7/36**, **17/36** y **21/36**. La salida continua estuvo dentro de ±0.25 en **13/36** y de ±0.50 en **18/36**. El MAE se calcula solo en fotos cubiertas; su cambio frente a cajas anotadas no mide el efecto causal de las cajas porque los subconjuntos difieren.

La agregación **nativa** ordinal promedia las seis probabilidades de umbral de las fotos disponibles de cada vaca y luego produce una categoría; dio MAE categórico **0.2500** en 8 vacas. Para la **regla común** de comparación, se promediaron los BCS categóricos por foto y se dio igual peso a cada vaca: MAE **0.2243** en esas mismas 8. La salida continua da MAE **0.2397** con ambas reglas porque promediar las probabilidades y luego sumarlas equivale a sumar y promediar en este caso.

## Comparación pareada de los tres clasificadores

CORAL, DINOv2 categórico y EfficientNet categórico tienen exactamente las **mismas 36 fotos**, las mismas etiquetas BCS y los mismos estados de cobertura: emiten predicción en las **mismas 24 fotos de 8 vacas**. Comparten detector; sus predictores y formas de recortar y convertir la salida BCS difieren.

| MAE sobre el mismo subconjunto | DINOv2 categórico | CORAL categórico | EfficientNet categórico |
|---|---:|---:|---:|
| Por fotografía, 24 fotos | 0.2083 | 0.2813 | 0.3646 |
| Por vaca, media común de BCS por foto, 8 vacas | 0.1920 | 0.2243 | 0.2623 |

La referencia constante BCS 3.0, calculada sobre **esas mismas 24 fotos y 8 vacas**, obtuvo MAE **0.3750 por foto** y **0.3438 por vaca**. Es una referencia del subconjunto cubierto, no una corrida adicional del detector.

Como diagnóstico de clasificación, el **F1 macro de las siete clases** en las 24 fotos cubiertas fue **0.2431** para DINO, **0.2250** para CORAL y **0.2602** para EfficientNet. Al contar las 12 abstenciones como predicciones fuera del catálogo y mantener las 36 referencias, los F1 macro fueron **0.2134**, **0.1589** y **0.2063**. La clase 2.25 no aparece en validación; aporta cero al F1 de siete clases. El F1 cambia el orden de los candidatos frente al MAE porque no distingue un error de 0.25 de otro mayor una vez que la clase es incorrecta. Los valores y denominadores están en [clasificadores_pareados_validacion.csv](clasificadores_pareados_validacion.csv).

El orden descriptivo en estas 24 fotos no determina un ganador final: los modelos se seleccionaron usando esta validación, son solo ocho vacas cubiertas y faltan aún la conciliación de dos fotos frente a ResNet18 y una prueba independiente no usada durante desarrollo. La comparación con la corrida ordinal v4 histórica permanece separada por diferencias de catálogo, pesos y protocolo.
