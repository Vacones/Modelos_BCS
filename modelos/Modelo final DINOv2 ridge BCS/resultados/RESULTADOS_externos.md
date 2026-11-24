# Prueba con datasets externos

Fecha: 12 de noviembre de 2026. Con pocas vacas en los extremos de la escala, probamos si sumar fotos de datasets públicos ayudaba al modelo. El modelo base es el recorte de grupa con 10 % de margen, DINOv2 ViT-S/14 congelado (224, CLS y media de parches, promedio con la foto espejada) y una cabeza ridge o CORAL con PCA.

Siempre evaluamos **solo con vacas del CAETEC** (66 vacas, 331 fotos), con la validación cruzada por vaca de 5 pliegues repetida 10 veces. Los hiperparámetros, y el peso w de las fotos externas cuando aplica, se eligieron con una validación interna de 3 pliegues sobre las vacas CAETEC de entrenamiento. El MAE es por vaca (media de sus fotos) y los intervalos de 95 % salen de 2000 remuestreos bootstrap sobre vacas. Los scripts son `extraer_externos.py` y `evaluar_externos.py`, y los números completos están en `resultados_externos.json`. Las características se borraron después.

## Muestra de Huang (ScienceDB)

Tomamos 3000 imágenes, 150 por cada combinación de vista (GS, YM, L-i y R-i) y nivel (3.25, 3.5, 3.75, 4.0 y 4.25), con máximo 2 por vaca y sin duplicados. Casi todas las imágenes traen una sola caja, que marca la grupa. La caja cubre en mediana el 23 % del cuadro en GS, L-i y R-i, y el 9 % en YM.

Al revisarlas vimos que YM es una cámara cenital en el carril de salida que ve la vaca completa desde atrás y arriba, y es la más parecida a nuestras fotos. GS es un primer plano cenital de la grupa con marcas de pintura, y L-i y R-i son tomas muy cercanas en diagonal desde arriba. Las fotos del CAETEC son traseras y a la altura del animal. Además Huang va de 3.25 a 4.25 y CAETEC de 2.25 a 3.75, así que solo coinciden en 3.25 a 3.75.

## Resultados en las 66 vacas del CAETEC

| Configuración | MAE | ±0.25 | Pendiente | AUC ≤2.75 | MAE ≥3.5 | ΔMAE contra a [IC 95 %] |
|---|---|---|---|---|---|---|
| Constante (mediana) | 0.318 | 0.59 | 0.00 | 0.50 | 0.62 | |
| a, solo CAETEC (ridge) | 0.210 | 0.84 | 0.49 | 0.95 | 0.36 | ref |
| b, más Huang completo con w = 0.1 | 0.226 | 0.82 | 0.47 | 0.90 | 0.36 | +0.016 [-0.007, 0.039] |
| b, más Huang completo con w = 1 | 0.275 | 0.76 | 0.37 | 0.82 | 0.38 | +0.065 [0.033, 0.101] |
| c, más solo la vista YM con w = 0.1 | 0.223 | 0.83 | 0.47 | 0.90 | 0.36 | +0.013 [-0.007, 0.033] |
| d, más Huang con indicador de dominio, w = 0.3 | 0.235 | 0.83 | 0.43 | 0.87 | 0.39 | +0.025 [0.004, 0.047] |
| e, solo CAETEC con calibración lineal | 0.205 | 0.85 | 0.58 | 0.95 | 0.32 | -0.005 [-0.015, 0.005] |

La tabla muestra las variantes con ridge. Con CORAL el patrón es el mismo: solo CAETEC da 0.206, y sumar Huang completo lo sube a 0.248 con w = 0.1 y a 0.300 con w = 1. Cuando dejamos que la validación interna eligiera w, eligió no usar Huang (w = 0) en los 50 pliegues con CORAL y en unos 40 de 50 con ridge. El archivo JSON tiene las 32 configuraciones.

## Transferencia entre datasets

Un ridge entrenado solo con Huang califica a nuestras vacas en promedio 0.82 puntos más alto (MAE 0.824), con una pendiente de 0.18. Aun así las ordena en parte (Spearman 0.49), y la vista GS es la que mejor lo hace (Spearman 0.60 y AUC de 0.81 para vacas flacas). Si se le quita ese sesgo, el MAE baja a 0.29, que sigue siendo peor que el 0.21 del modelo entrenado con CAETEC.

En sentido contrario, un ridge entrenado con todo CAETEC no ordena las vacas de Huang. Su Spearman es 0.04 con todas las vistas y 0.18 en GS, y en el rango que comparten (3.25 a 3.75) el MAE va de 0.32 en GS a 0.58 en YM.

## Zenodo 20290988 y Criollo (Dryad)

Zenodo viene de una sala rotativa con cámara superior y trasera, 3 o más vacas por cuadro, RGB de 1920 × 1080 y profundidad sin registrar, sin cajas y con BCS de 2.75 a 5.0 por pase. Habría que detectar y recortar la vaca central, y la misma diferencia de escala y de vista que vimos con Huang hace esperar que tampoco ayude, así que no entrenamos con él. Criollo es ganado de carne con escala de 1 a 9 e imágenes de profundidad, gris y bordes de 224 px tomadas desde arriba, con unos 100 animales. No es compatible en escala, raza ni tipo de imagen, y no lo usamos.

## Conclusión

1. Huang no ayuda a entrenar. Con cualquier peso fijo el MAE por vaca empeora entre 0.01 y 0.10, las predicciones se comprimen más hacia la media (la pendiente baja de 0.5 a entre 0.26 y 0.47) y empeoran justo las vacas de 3.5 y 3.75. Ni el indicador de dominio ni usar solo YM lo arreglan, y cuando la validación interna elige el peso, elige no usarlo.
2. Tampoco sirve para validar. Nuestro modelo no ordena a las vacas de Huang y el modelo de Huang califica a las nuestras unos 0.8 puntos más alto. Los veterinarios, el rango de BCS y el ángulo de la cámara son distintos, así que una métrica sobre Huang no dice nada de nuestro caso.
3. La única señal compartida es que Huang GS ordena en parte a nuestras vacas, pero no lo bastante alineada como para sumar sus fotos al entrenamiento.
4. Sin datos externos, la calibración lineal (e) corrige parte de la compresión. La pendiente pasa de 0.49 a 0.58 y el MAE en vacas de 3.5 o más baja de 0.36 a 0.32, aunque el MAE global casi no cambia.
5. Por eso el modelo final se entrena solo con fotos del CAETEC, con calibración lineal. El error en los extremos se debe a que hubo una ligera falta de vacas de BCS 2.5 o menos y de 3.5 o más.

Con CPU y foto espejada, DINOv2-S tardó 0.095 s por imagen en CAETEC y 0.102 s en Huang. La validación cruzada de las 32 configuraciones por 50 pliegues tardó 34 minutos.
