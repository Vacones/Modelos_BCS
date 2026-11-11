# Inventario de evidencia para la comparación maestra

Revisión: 10 de noviembre de 2026. `Verificado localmente` significa que se encontraron archivos de predicción y se recalculó al menos una métrica. `Documentado` significa que el informe describe una ejecución, pero sus predicciones privadas no están disponibles en este espacio para recalcularla. El inventario no elige un ganador.

| Candidato | Evidencia encontrada | Unidad y cajas | Estado para comparación por vaca |
|---|---|---|---|
| ResNet18 sin cajas | Ejecución local `modelo-sin-cajas-v1`: 10 vacas y 36 fotos de validación; predicciones por vaca y foto, configuración, versiones y checkpoint. Informe de Drive: [Assess Model](https://docs.google.com/document/d/1YmY9kGQ2c9JnxkFMDT1s-jhlR3Au9jOKbnqOYB4uwHI/edit). | Vaca principal, foto secundaria; sin cajas. | Verificado localmente. `test` ya consultado. Falta registrar el commit y la huella del manifiesto en la ejecución. |
| YOLO + EfficientNet de clasificación | [Model Assessment](https://docs.google.com/document/d/1qMms6LQeQLAYyLBn95H1RIyBzHFf3j1btDGeGGLMO_w/edit) documenta 36 fotos de validación, 30 clasificadas y 6 abstenciones en el flujo ajustado. Describe evidencia transcrita de la ejecución `noche_20261028_032148_WLyisH`. | Foto; cajas anotadas y predichas separadas. | Documentado. Faltan predicciones por vaca/foto y configuración/checkpoint exactos para recalcular MAE conjunto por vaca. |
| EfficientNet + CORAL ordinal | [Informe ordinal](https://docs.google.com/document/d/1BscInm74v1ZIgLbp0hvS3-i2FRl-KsHNPMiUo3Xu1-M/edit) documenta validación cruzada sobre 56 vacas y prueba sobre 10. El [informe DINOv2](https://docs.google.com/document/d/1Wl5ib2npN-AwFp9jeMp5FPZjZKEYmLwc5PV9oU3hLZ8/edit) añade una comparación de `ordinal v4` en 36 fotos de validación: 29 clasificadas por el sistema automático. | Vaca en validación cruzada; foto en comparación v4. Cajas anotadas y predichas, según ejecución. | Documentado. No mezclar validación cruzada, `test` y v4 como una sola corrida. Faltan predicciones privadas de v4, correspondencia de checkpoint/commit y reevaluación de las 10 vacas de validación por vaca. `test` ya consultado. |
| YOLO + DINOv2/LDL | [Informe de evaluación](https://docs.google.com/document/d/1Wl5ib2npN-AwFp9jeMp5FPZjZKEYmLwc5PV9oU3hLZ8/edit) documenta la ejecución `mac_20261103_112530_hGbsPu`: 36 fotos de validación, 28 clasificadas y 8 abstenciones; distingue salida categórica y continua. | Foto; cajas anotadas y predichas separadas. | Documentado con evidencia experimental en Drive. Faltan aquí los CSV privados y el checkpoint para recalcular por vaca. No reporta evaluación de `test` para esa ejecución. |
| YOLO + EfficientNet de regresión continua | Ejecución Mac local `bcs_secuencial_v2_regresion_mac`: 36 fotos de validación, 27 con predicción; 8 de 10 vacas cubiertas. Predicciones individuales, configuración y checkpoints locales; código integrado en Git. | Foto y resumen por vaca; cajas anotadas y predichas separadas. | Verificado localmente. El informe histórico [YOLO + regresión](https://docs.google.com/document/d/1ZeMOOpm-claJ_hW65twWe0mInZ_eewsqsjMIi3_0TWk/edit) consultó `test`; no confundir esa corrida con la nueva de Mac. |

## Comprobaciones de correspondencia ya realizadas

- ResNet18 y YOLO + regresión contienen las mismas diez identidades de validación y los BCS coinciden.
- De 36 fotos de validación por fuente, 34 tienen hash de archivo exactamente igual. La identidad de las otras dos fotos aún requiere conciliarse; no se declararán fotos idénticas por compartir vaca y BCS.
- El MAE por vaca de ResNet18 en las diez vacas es 0.150. En las ocho vacas cubiertas por el pipeline de regresión es 0.125, pero la selección de fotos usada por ambos sistemas no es idéntica; esas cifras son diagnósticas.

## Pendientes para completar la tabla por vaca

1. Obtener localmente los CSV de predicciones y las configuraciones de YOLO + EfficientNet, ordinal v4 y DINOv2/LDL. Los documentos de Drive no sustituyen esos archivos para recalcular por vaca.
2. Conciliar las dos fotos de validación con hashes distintos mediante identidad original y revisión de transformaciones; no mover particiones ni excluirlas automáticamente.
3. Registrar commit, huella de datos, checkpoint y versión de entorno en cada corrida nueva o reconocer la procedencia desconocida de las históricas.
4. Calcular una tabla pareada con idénticas vacas, fotos y reglas, y documentar aparte los resultados históricos de validación cruzada y `test`.
