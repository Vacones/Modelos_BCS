# Modelo final contra DINOv2 + LDL

Fecha: 13 de noviembre de 2026. Usamos las 331 fotos de las 66 vacas del CAETEC con las cajas de grupa anotadas. Todas las variantes se midieron con los mismos pliegues de la validación cruzada por vaca (5 pliegues repetidos 10 veces), con métricas por vaca (media de sus fotos) e intervalos de confianza de 95 % por bootstrap sobre vacas. El script es `comparar_ldl.py` y los números completos están en `resultados_cierre.json`. Las características se calcularon en memoria y no se guardaron.

Hicimos esta comparación porque el modelo DINOv2 + LDL de [`modelos/con_cajas_yolo/dinov2_ldl_mac`](../../con_cajas_yolo/dinov2_ldl_mac/) y el modelo final usan la misma red base, pero cambian el recorte, el número de vistas y la cabeza. Queríamos saber cuál de esos cambios explica la diferencia.

## Variantes

- **a, modelo final**: recorte de grupa con 10 % de margen llevado a 224, DINOv2-S, CLS y media de parches, promedio con la foto espejada y ridge con alfa 100.
- **b**: ridge sobre las dos vistas de DINOv2 + LDL (1536 características), con alfa elegido por una validación cruzada interna agrupada por vaca.
- **c**: cabeza LDL sobre las dos vistas, con 3 semillas.
- **d**: cabeza LDL sobre la vista del modelo final (768 características), con 3 semillas.
- **e**: ridge con alfa 100 sobre la vista completa de DINOv2 + LDL, para aislar el efecto del recorte.

DINOv2 + LDL usa la caja exacta sin margen, llevada a 224 con relleno. Su segunda vista es el 65 % superior de ese recorte (`DEFAULT_ROI = [0, 0, 1, 0.65]` en `regions.py`), también con relleno. La cabeza tiene LayerNorm, Dense 128, GELU, Dropout 0.3 y 7 salidas (2.25 a 3.75). La pérdida es entropía cruzada con etiquetas gaussianas (σ 0.125) más 0.5 × la diferencia de CDF, cada foto pesa 1 entre el número de fotos de su vaca y se entrena con AdamW (lr 1e-3, weight decay 1e-4, lote 32, hasta 100 épocas con paciencia 15). La predicción es el valor esperado. La reimplementamos en PyTorch. En la versión original la época se elegía con las vacas de validación. Aquí la elegimos con el 20 % de las vacas de entrenamiento de cada pliegue, para no ver las vacas evaluadas.

## Resultados (66 vacas)

| Variante | MAE [IC 95 %] | ±0.25 | ±0.50 | Pendiente | AUC ≤2.75 | MAE ≥3.5 | MAE ≤2.5 | ΔMAE contra a [IC 95 %] |
|---|---|---|---|---|---|---|---|---|
| Constante (mediana) | 0.318 [0.261, 0.375] | 0.59 | 0.89 | 0.00 | 0.50 | 0.62 | 0.52 | |
| **a, modelo final** | **0.206 [0.173, 0.241]** | **0.84** | **0.99** | **0.50** | **0.95** | 0.37 | **0.24** | ref |
| b, ridge con 2 vistas | 0.208 [0.170, 0.251] | 0.82 | 0.97 | 0.48 | 0.86 | 0.35 | 0.27 | +0.003 [-0.025, 0.029] |
| e, ridge con la vista completa | 0.215 [0.175, 0.258] | 0.80 | 0.97 | 0.49 | 0.86 | 0.37 | 0.27 | +0.010 [-0.017, 0.038] |
| c, LDL con 2 vistas (media de 3 semillas) | 0.226 [0.188, 0.265] | 0.81 | 0.97 | 0.43 | 0.85 | 0.37 | 0.28 | +0.020 [-0.008, 0.050] |
| d, LDL con 1 vista (media de 3 semillas) | 0.244 [0.210, 0.281] | 0.81 | 0.98 | 0.40 | 0.88 | 0.41 | 0.28 | +0.038 [0.018, 0.057] |

Entre semillas, c va de 0.228 a 0.237 y d de 0.246 a 0.252. Con una sola semilla, c queda entre 0.022 y 0.032 por encima de a. La parada temprana eligió en c una mediana de 6 épocas (percentil 10 de 1 y percentil 90 de 36) y en d una mediana de 17. Con 50 vacas de entrenamiento la cabeza LDL sobreajusta en pocas épocas. En b la validación interna eligió alfa 1000 en 23 pliegues y alfa 100 en 21.

## Conclusión

1. Ninguna variante mejora al modelo final. La mejor, ridge con las dos vistas, empata (+0.003) y detecta peor a las vacas flacas (AUC 0.86 contra 0.95).
2. La cabeza LDL empeora el resultado, +0.020 con dos vistas y +0.038 con una (en este caso el intervalo no incluye el 0). Con 66 vacas, una red con parada temprana es menos estable que un ridge.
3. Con ridge, la segunda vista no aporta una mejora medible. El recorte sin margen y sin foto espejada (e) queda apenas por debajo del modelo final.
4. Si la época se elige sin ver las vacas evaluadas, DINOv2 + LDL da 0.23 por vaca. Las cifras medidas con las mismas vacas que se usaron para elegir el checkpoint salen optimistas.
5. Por eso nos quedamos con el modelo final (una vista, foto espejada, ridge con alfa 100 y calibración), que es el más simple, el más estable y el mejor o igual en todas las métricas.
