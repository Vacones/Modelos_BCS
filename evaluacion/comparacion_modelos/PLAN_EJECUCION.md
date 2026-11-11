# Plan de ejecución para la comparación maestra

**Estado:** preparación técnica, 10 de noviembre de 2026. Las corridas nuevas tendrán identificadores y resultados propios; no reemplazan las métricas de los informes históricos.

## Objetivo y reglas comunes

Evaluar los cinco enfoques con las diez vacas de validación del conjunto local aprobado. La unidad principal será la vaca; las fotografías y las cajas se analizarán por separado. Cada ejecución guardará fuera del repositorio un registro por fotografía elegible, incluso cuando el sistema se abstenga: clave estable, vaca, BCS real, BCS predicho (o vacío), estado, partición y origen de cajas. Para resultados categóricos se guardará además la clase y, cuando existan, las probabilidades; para continuos, el valor sin redondear. Ningún identificador o predicción individual se publicará. El benchmark común puede promediar valores BCS por foto; se reportará por separado la regla nativa de cada modelo, como el promedio de probabilidades de ResNet18.

Antes de comparar, el evaluador común deberá comprobar unicidad de claves, una sola etiqueta por vaca, mismas vacas y BCS, catálogos compatibles, y correspondencia de fotografías. Si dos archivos difieren, se documentará la causa; no se asumirán equivalentes por compartir vaca. Toda métrica indicará el denominador y la cobertura. Las cajas anotadas y las predichas darán resultados separados. El conjunto `test` previamente consultado no se usará para seleccionar arquitectura o umbrales ni se presentará como prueba independiente.

## Entregables por línea de trabajo

| Línea | Trabajo previo | Puerta antes de entrenar | Resultado privado requerido |
|---|---|---|---|
| ResNet18 sin cajas | Pesos y predicciones locales disponibles. | Confirmar manifiesto, huella y correspondencia con el conjunto común. | Predicciones por foto y por vaca ya existentes; adaptar solo el lector común. |
| YOLO + regresión v2 Mac | Pesos y predicciones locales disponibles. | Conciliar dos fotografías con hash distinto frente a ResNet18. | Predicciones de `val`, configuración, resumen por vaca ya existentes. |
| YOLO + DINOv2/LDL | **Nueva corrida `comparacion_dino_01` terminada y verificada localmente**; pesos históricos siguen ausentes. | Revisar abstenciones y agregación por vaca antes de decidir. | Checkpoint, `val_pipeline/predicciones.csv`, `por_vaca.csv`, configuración y métricas privados disponibles; `val_gt` separado. [Resumen](DINO_CORRIDA_01.md). |
| YOLO + EfficientNet clasificador | Código WSL/CUDA; faltan pesos históricos y entrada preparada. | Adaptar carga COCO y soporte Mac, demostrar inferencia/ETL sintéticos y separación por vaca. | Corrida **nueva** en Mac; predicciones de pipeline y de cajas anotadas por separado, con abstenciones. |
| YOLO + CORAL ordinal | Código usa export YOLO/metadata que no está localmente; faltan pesos históricos. | Adaptar COCO aprobado sin filtrar ni mover particiones; pruebas de etiquetas/cajas/identidades. | Corrida **nueva** entrenada solo con `train`; predicciones de `val` por foto y vaca, configuración y checkpoint. |

## Orden de trabajo

1. Implementar y probar el contrato de entrada y evaluador común con datos ficticios. Recalcular los dos modelos locales conocidos sin publicar las filas privadas.
2. Preparar DINOv2/LDL para ejecución supervisada por Eduardo: comando de prevalidación, salida nueva, revisión de recursos. Eduardo ejecuta el entrenamiento; se verifica el resultado antes de avanzar.
3. Adaptar en paralelo EfficientNet clasificador y CORAL ordinal a Mac/COCO, cada uno con pruebas propias. Eduardo ejecuta las corridas después de ver las verificaciones y las instrucciones exactas.
4. Conciliar claves y fotos de las cinco salidas. Calcular métricas por vaca en las mismas diez vacas, cobertura y MAE condicionado; para comparación pareada usar únicamente las mismas observaciones y declarar diferencias de fotos efectivamente usadas.
5. Generar tablas y figuras finales: cobertura, errores por BCS, diferencias pareadas por vaca, matrices de confusión para salidas categóricas, y detector por separado. Registrar versión, datos y limitaciones de cada cifra.

## Criterio de cierre

La comparación se considerará completa cuando las cinco corridas tengan predicciones individuales auditables y un evaluador único reproduzca los agregados, sin fuga entre particiones ni datos privados en Git. Si falta una corrida o su correspondencia de datos, quedará señalada como pendiente y no se completará con supuestos.
