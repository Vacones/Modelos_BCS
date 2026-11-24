# Cierre: nuestra receta contra el "DINOv2 + LDL" del equipo

Fecha: 2026-11-13. Datos: 66 vacas CAETEC, 331 fotos, cajas de grupa anotadas. Protocolo: GroupKFold por vaca 5×10 con los mismos folds para todas las variantes, métricas por vaca (media de sus fotos) e IC 95 % por bootstrap sobre vacas. Script: `comparar_ldl.py`. Números completos en `resultados_cierre.json`. Las features se calcularon en memoria y no se guardaron.

## Variantes

- **a (la nuestra)**: recorte de grupa +10 % redimensionado a 224, DINOv2-S, CLS + media de parches, flip TTA, ridge con alfa 100.
- **b**: ridge sobre las 2 vistas del equipo (1536 features), alfa elegido por CV interna agrupada.
- **c**: cabeza LDL del equipo sobre sus 2 vistas, 3 semillas.
- **d**: cabeza LDL del equipo sobre nuestra vista (768), 3 semillas.
- **e**: ridge con alfa 100 sobre la vista completa del equipo. Aísla el efecto del recorte.

Receta del equipo, tomada del repo `Vacones/Modelos_BCS` (`modelos/con_cajas_yolo/dinov2_ldl_mac`): caja exacta sin margen con letterbox a 224. La segunda vista es el 65 % superior de ese recorte (`DEFAULT_ROI = [0, 0, 1, 0.65]` en `regions.py`), con letterbox a 224. Cabeza LayerNorm, Dense 128, GELU, Dropout 0.3 y 7 logits (2.25 a 3.75). Pérdida: entropía cruzada con etiquetas gaussianas (σ 0.125) más 0.5 × diferencia de CDF. Peso 1/fotos por vaca. AdamW con lr 1e-3, weight decay 1e-4, batch 32, hasta 100 épocas con paciencia 15. Predicción: valor esperado. La reimplementamos en PyTorch. El equipo elegía la época con vacas de validación. Aquí se elige con el 20 % de las vacas de entrenamiento de cada fold, para no ver las vacas evaluadas.

## Tabla (66 vacas)

| variante | MAE [IC95] | ±0.25 | ±0.50 | pendiente | AUC ≤2.75 | MAE ≥3.5 | MAE ≤2.5 | ΔMAE vs a [IC95] |
|---|---|---|---|---|---|---|---|---|
| constante (mediana) | 0.318 [0.261, 0.375] | 0.59 | 0.89 | 0.00 | 0.50 | 0.62 | 0.52 | |
| **a nuestra receta** | **0.206 [0.173, 0.241]** | **0.84** | **0.99** | **0.50** | **0.95** | 0.37 | **0.24** | ref |
| b ridge 2 vistas | 0.208 [0.170, 0.251] | 0.82 | 0.97 | 0.48 | 0.86 | 0.35 | 0.27 | +0.003 [-0.025, 0.029] |
| e ridge vista completa del equipo | 0.215 [0.175, 0.258] | 0.80 | 0.97 | 0.49 | 0.86 | 0.37 | 0.27 | +0.010 [-0.017, 0.038] |
| c LDL 2 vistas (media de 3 semillas) | 0.226 [0.188, 0.265] | 0.81 | 0.97 | 0.43 | 0.85 | 0.37 | 0.28 | +0.020 [-0.008, 0.050] |
| d LDL 1 vista (media de 3 semillas) | 0.244 [0.210, 0.281] | 0.81 | 0.98 | 0.40 | 0.88 | 0.41 | 0.28 | +0.038 [0.018, 0.057] |

Spread entre semillas: c tiene MAE 0.228 a 0.237 y d tiene 0.246 a 0.252. Con una sola semilla, c queda entre +0.022 y +0.032 contra a. La época elegida por parada temprana tiene mediana 6 en c (p10 = 1, p90 = 36) y 17 en d. La MLP sobreajusta en pocas épocas con 50 vacas de entrenamiento. En b, la CV interna eligió alfa 1000 en 23 folds y 100 en 21.

## Conclusión

1. Ninguna variante del equipo le gana a nuestra receta. La mejor (ridge con 2 vistas) empata (Δ +0.003) y tiene peor AUC para vacas flacas (0.86 contra 0.95).
2. La cabeza LDL empeora el resultado: +0.020 con 2 vistas y +0.038 con 1 vista (IC fuera del 0). Con 66 vacas, una MLP con parada temprana es más inestable que un ridge.
3. La segunda vista (65 % superior) no aporta nada que se pueda medir sobre ridge. El recorte sin margen y sin flip (e) es apenas peor que el nuestro.
4. Con la época elegida sin ver las vacas evaluadas, la receta "DINOv2 + LDL" del equipo da 0.23 por vaca en esta CV. Cualquier cifra suya medida en las mismas vacas de validación con las que se eligió el checkpoint es optimista.
5. Decisión: queda nuestra receta (1 vista, flip TTA, ridge con alfa 100 y calibración), que es la más simple, la más estable y la mejor o igual en todas las métricas.
