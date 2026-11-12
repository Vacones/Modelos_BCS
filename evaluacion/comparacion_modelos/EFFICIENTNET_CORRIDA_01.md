# YOLO + EfficientNet: nueva corrida Mac de validación

**Fecha de revisión:** 12 de noviembre de 2026. **Corrida privada:** `resultados_yolo_eff_mac`. **Partición evaluada:** `val` (36 fotos, 10 vacas). No se evaluó `test` en esta corrida. Esta ejecución es distinta de la histórica documentada en Drive.

## Procedencia y selección

- La preparación `preparacion_v1` tiene 239 fotos de entrenamiento con BCS utilizable, 36 de validación y 56 de prueba. El manifiesto de la preparación, las cajas predichas y los clasificadores tienen la misma huella SHA-256: `39e192896d1107e9a5b0f1e73b9744d6d5c0a082ae995a9e61dfbab646bd49b7`.
- Se entrenó una cabeza de clasificación y después un ajuste fino. Cada fase eligió su checkpoint por `val_loss`. Para evaluar el sistema completo se eligió el ajuste fino por su menor MAE BCS de validación con cajas anotadas (0.4306 frente a 0.4514). Esta elección también consultó validación; sus métricas son de desarrollo.
- El sistema completo reutilizó el detector YOLO entrenado para la nueva corrida DINOv2/LDL, con las mismas fotos COCO aprobadas. Huella SHA-256 del detector: `28c95b485dba90aec93f4c90c3abbafbecbb8948c40062daad6dd71b27b0dd6b`. Umbral de confianza: 0.25; asociación mínima: 0.8. No se entrenó un detector propio para esta variante.
- Huella SHA-256 del checkpoint EfficientNet de ajuste fino: `9ac8763a2a0e35f53416649d6a60e00da104ef7a62a96d4246fcc7ac1a5b897e`. Predicciones y pesos permanecen fuera del repositorio.

## Resultados por fotografía

| Evaluación | Fotos clasificadas | Abstenciones | MAE entre clasificadas | Exactitud global | Dentro de ±0.25 global | Dentro de ±0.50 global |
|---|---:|---:|---:|---:|---:|---:|
| Cabeza, cajas anotadas | 36/36 | 0 | 0.4514 | 8/36 | 19/36 | 22/36 |
| Ajuste fino, cajas anotadas | 36/36 | 0 | 0.4306 | 10/36 | 20/36 | 22/36 |
| Ajuste fino, cajas predichas | **24/36** | **12** | **0.3646** | **7/36** | **14/36** | **18/36** |

Las 12 abstenciones del sistema completo fueron 9 asociaciones ambiguas y 3 fotos sin par válido. El MAE de 0.3646 **solo** se calcula sobre las 24 fotos cubiertas. La reducción aparente frente a 0.4306 con cajas anotadas compara subconjuntos diferentes: sobre esas mismas 24 fotos, el ajuste fino con cajas anotadas tuvo MAE 0.4063. Ninguna de estas cifras indica que las cajas predichas mejoren el clasificador. La referencia constante BCS 3.0 obtuvo MAE 0.3472 en las 36 fotos de validación; no tiene el mismo denominador que el MAE condicionado del pipeline.

## Unidad vaca y comparación controlada con DINOv2

El pipeline clasificó al menos una foto de **8/10 vacas**. Si se promedian los BCS predichos de las fotos disponibles de cada vaca y se da el mismo peso a cada vaca, su MAE es **0.2623** sobre esas ocho. Esta regla común es diagnóstica y no reemplaza las predicciones individuales ni la métrica de cada modelo.

La nueva corrida DINOv2 y esta corrida EfficientNet tienen las **mismas 36 fotos, etiquetas y estados de cobertura**: ambos usan el mismo detector y se abstuvieron en las mismas 12 fotos. Sobre las **mismas 24 fotos**, el MAE categórico fue 0.3646 para EfficientNet y 0.2083 para DINOv2. Sobre las **mismas 8 vacas cubiertas**, el MAE categórico bajo la media por vaca fue 0.2623 y 0.1920, respectivamente. Es una comparación de validación pequeña y usada durante desarrollo; el detector compartido controla sus detecciones, pero los recortes y el tratamiento del BCS deben conservarse en la interpretación de cada flujo. No se declara un ganador final.

## Limitaciones y siguiente paso

El ajuste fino quedó por debajo de la referencia constante en MAE con cajas anotadas sobre las 36 fotos. La clase BCS 2.25 tiene solo dos fotos de entrenamiento y ninguna de validación, por lo que el F1 macro sobre las siete clases debe leerse junto con los conteos. La advertencia de Keras sobre `shuffle=True` es del adaptador de `tf.data`; el conjunto de entrenamiento ya se baraja explícitamente.

Completar la nueva evaluación ordinal y después actualizar la tabla pareada principal y las gráficas de errores por nivel BCS. Antes de cualquier resultado final, revisar la procedencia de los dos archivos que no son byte idénticos entre ResNet18 y COCO, y definir una prueba independiente que no haya intervenido en decisiones de desarrollo.
