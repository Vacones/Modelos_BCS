# DINOv2/LDL: nueva corrida Mac de validación

**Corrida privada:** `resultados_dino/comparacion_dino_01`. **Código:** commit local `abd22cd`. **Partición evaluada:** `val`; `test` no fue evaluado. La carpeta contiene `FINALIZADO.txt`, checkpoints, configuración, métricas y predicciones privadas. Esta corrida no reemplaza ni reproduce numéricamente la ejecución histórica documentada en Drive.

## Trazabilidad comprobada

- COCO local aprobado: 243 fotos `train` y 36 `valid`. Las 36 fotos elegibles de validación tienen archivos existentes y su SHA-256 actual coincide con el manifiesto de esta corrida.
- La cabeza LDL seleccionó el checkpoint de la época **9** mediante MAE continuo de validación con cajas anotadas. El detector nuevo completó **80 épocas**; no se reutilizaron pesos históricos.
- Huella SHA-256 de la cabeza seleccionada: `37872530a19951840cb2df762eece0a9707e2df9b15ead6e5d40e91c501a3ef7`. Huella del detector `best.pt`: `28c95b485dba90aec93f4c90c3abbafbecbb8948c40062daad6dd71b27b0dd6b`.
- Las predicciones individuales siguen fuera del repositorio. Los agregados de abajo se recalcularon leyendo sus CSV y JSON privados.

## Resultados por fotografía

| Evaluación | Fotos con predicción | MAE categórico entre cubiertas | ±0.25 global categórico | MAE continuo entre cubiertas | ±0.25 global continuo |
|---|---:|---:|---:|---:|---:|
| Clasificador con cajas anotadas | 36/36 | 0.2083 | 29/36 | 0.2216 | 23/36 |
| Sistema completo con cajas predichas | **24/36** | **0.2083** | **21/36** | **0.2014** | **18/36** |

El sistema completo se abstuvo en **12/36** fotos: 9 asociaciones ambiguas y 3 sin par válido. Entre sus 24 predicciones, obtuvo 8 aciertos exactos de categoría, 21 dentro de ±0.25 y 23 dentro de ±0.50; al usar todas las elegibles como denominador, esos conteos son 8/36, 21/36 y 23/36. La referencia constante 3.0 en las 36 fotos obtuvo MAE 0.3472 y 21/36 dentro de ±0.25. El MAE condicionado del sistema completo no se interpreta sin su cobertura. La aparente reducción del MAE continuo frente a cajas anotadas corresponde a subconjuntos distintos y no demuestra que las cajas predichas mejoren el predictor.

## Resultados por vaca y pareo

El sistema cubrió **8/10** vacas de validación. Al promediar los BCS disponibles por foto dentro de cada vaca y dar el mismo peso a cada vaca, el MAE fue **0.1920** para la categoría y **0.1949** para la esperanza continua. Esta agregación es una lectura común de comparación, no una métrica seleccionada durante el entrenamiento.

Las 36 fotos de este COCO tienen los mismos hashes que la fuente COCO de la corrida YOLO + regresión Mac. Ambas emitieron predicción en **20 fotos de 7 vacas**. En ese subconjunto estricto, el MAE por vaca continuo fue **0.1936** para DINO y **0.2332** para la regresión. Son pocas vacas y el subconjunto depende de ambas coberturas; no se declara ganador. Frente a ResNet18, solo 34/36 fotos tienen archivos byte idénticos; el pareo categórico exige además que ambos sistemas predigan, por lo que se reduce a 22 fotos de 8 vacas y se presenta únicamente como diagnóstico.

## Próxima verificación

Revisar errores y abstenciones por nivel BCS sin publicar identificadores; después completar las nuevas corridas de EfficientNet clasificador y CORAL ordinal. Para evaluar los sistemas completos con cajas predichas bajo el mismo protocolo, CORAL todavía necesita esa etapa; su nueva ruta actual evalúa cajas anotadas.
