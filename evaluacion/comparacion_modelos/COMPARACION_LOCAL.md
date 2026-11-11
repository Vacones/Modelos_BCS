# Comprobación pareada local: ResNet18 y YOLO + regresión Mac

**Fecha:** 10 de noviembre de 2026. **Partición:** validación. Esta comprobación es diagnóstica y no selecciona un modelo.

El adaptador leyó el manifiesto privado, las predicciones por imagen de ResNet18, los metadatos COCO y las predicciones del pipeline YOLO + regresión. No guardó identificadores ni predicciones individuales en el repositorio. Las dos fuentes contienen 36 fotos de validación y 10 vacas con BCS concordantes. **34 fotos tienen el mismo hash de archivo**; las otras dos quedaron fuera de este análisis foto a foto hasta aclarar su transformación o procedencia.

| Métrica en las 34 fotos idénticas | ResNet18 | YOLO + regresión |
|---|---:|---:|
| Fotos con predicción | 34/34 | 25/34 |
| Vacas con al menos una predicción | 10/10 | 8/10 |
| MAE por vaca, entre vacas cubiertas de cada modelo | 0.2339 | 0.2629 |

Para el **pareo estricto**, se usaron solo las **25 fotos con predicción de ambos sistemas**, correspondientes a **8 vacas**. Al promediar las predicciones disponibles por vaca y dar igual peso a cada vaca, el MAE fue **0.1563** para ResNet18 y **0.2629** para la regresión. La diferencia media de error absoluto fue **0.1066 BCS** a favor de ResNet18 en este subconjunto. Es una observación de validación pequeña y seleccionada por la cobertura de la regresión; no representa las diez vacas completas ni establece superioridad general.

Estas cifras difieren de los MAE de las corridas originales porque se restringen a fotos byte idénticas y, en el pareo, a fotos donde ambos sistemas emitieron predicción. **Aquí se usa la media de los BCS predichos por foto para ambos sistemas**; la corrida original de ResNet18 agrega probabilidades por vaca y después elige una categoría. Por tanto, este resultado no sustituye su métrica nativa por vaca. Además, ResNet18 produce categorías BCS y la regresión un valor continuo; la diferencia numérica se muestra como diagnóstico, no como clasificación directa de arquitecturas.

Para reproducir **solo agregados**, desde la raíz del repositorio:

```bash
python3 -m evaluacion.comparacion_modelos.adaptadores_locales \
  --manifiesto-resnet "$HOME/Desktop/vacones_reg/dataset_sin_cajas_v2_aprobado/_control/dataset_manifest.csv" \
  --predicciones-resnet "$HOME/Desktop/vacones_reg/resultados_sin_cajas/modelo-sin-cajas-v1/predicciones_por_imagen_validation.csv" \
  --datos-regresion "$HOME/Desktop/vacones_reg/resultados_yolo_regresion/bcs_secuencial_v2_regresion_mac/datos.json" \
  --predicciones-regresion "$HOME/Desktop/vacones_reg/resultados_yolo_regresion/bcs_secuencial_v2_regresion_mac/predicciones_val.csv"
```

El comando imprime conteos y métricas sin nombres de vacas ni fotos. Se detiene si hay etiquetas o identidades discordantes; no modifica datos ni resultados privados.
