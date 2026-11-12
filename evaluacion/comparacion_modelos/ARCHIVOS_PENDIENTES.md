# Archivos necesarios para cerrar la comparación por vaca

Estos insumos se revisarán **localmente**. No deben agregarse a GitHub ni a documentos o carpetas públicas. Si un archivo contiene nombres de personas, credenciales o rutas internas innecesarias, se limita su acceso y se extraen solo las columnas requeridas para el análisis.

## Para cada ejecución de validación

1. Predicciones **por fotografía**: identificador estable de foto y de vaca, partición, BCS de referencia, predicción BCS o estado de abstención, y si se usaron cajas anotadas o predichas. Para salidas categóricas, incluir catálogo y etiqueta predicha; para continuas, conservar el valor sin redondear.
2. Manifiesto privado usado por esa ejecución, con correspondencia de fotos y vacas. Si los nombres cambiaron al exportar Roboflow, aportar la tabla de correspondencia. No mover fotos entre particiones para hacer coincidir informes.
3. Configuración y trazabilidad: variante y fecha de corrida, commit del código, semilla, preprocesamiento, umbrales de detección/asociación/abstención, regla de agregación por vaca, versión del entorno y huellas del conjunto y checkpoint. Un dato histórico desconocido se marca como desconocido.
4. Métricas originales y, si existen, errores por foto o resúmenes por vaca. Sirven para cotejar el recálculo, no para sustituir las predicciones individuales.

## Carpetas concretas por localizar

| Ejecución | Archivos prioritarios |
|---|---|
| YOLO + EfficientNet de clasificación, corrida histórica `noche_20261028_032148_WLyisH` | Sus predicciones siguen pendientes **solo si se pretende reproducir esa corrida histórica**. La nueva `resultados_yolo_eff_mac` ya generó predicciones privadas de `val`, configuración y checkpoint; véase [EFFICIENTNET_CORRIDA_01.md](EFFICIENTNET_CORRIDA_01.md). No mezclar ambas. |
| YOLO + CORAL ordinal **v4 histórico** | Sus predicciones siguen pendientes **solo si se pretende reproducir v4**. La nueva `coral_pipeline_val_v1` ya generó predicciones privadas de `valid`, configuración y checkpoint; véase [ORDINAL_CORRIDA_01.md](ORDINAL_CORRIDA_01.md). No mezclar ambas. |
| YOLO + DINOv2/LDL, corrida histórica `mac_20261103_112530_hGbsPu` | Sus predicciones siguen pendientes **solo si se pretende reproducir esa corrida histórica**. La nueva `comparacion_dino_01` ya generó predicciones privadas de `val`, configuración y checkpoints; véase [DINO_CORRIDA_01.md](DINO_CORRIDA_01.md). No mezclar ambas. |

ResNet18, YOLO + regresión v2 Mac y las nuevas corridas DINO, EfficientNet y CORAL ya tienen predicciones locales. Falta conciliar dos archivos de validación entre ResNet18 y el COCO con contenido binario distinto antes de llamar a sus resultados una comparación foto a foto idéntica.

**Entrega al análisis:** rutas locales o de una carpeta privada autorizada, no copias en el repositorio. Se puede compartir primero solo la ruta y el esquema de columnas; el equipo revisará datos e identidades antes de publicar cualquier agregado.
