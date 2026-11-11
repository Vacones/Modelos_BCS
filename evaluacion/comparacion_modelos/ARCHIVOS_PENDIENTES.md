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
| YOLO + EfficientNet de clasificación, corrida `noche_20261028_032148_WLyisH` | CSV/JSON de predicciones de `val` del **pipeline con cajas predichas**, abstenciones y configuración de umbrales. Si hubo evaluación con cajas anotadas, conservarla separada. |
| YOLO + CORAL ordinal **v4** | Predicciones individuales de las 36 fotos de validación del flujo automático, con abstenciones; checkpoint/configuración que distingan v4 de validación cruzada y `test`. |
| YOLO + DINOv2/LDL, corrida `mac_20261103_112530_hGbsPu` | Predicciones de `val` del pipeline, estado de asociación por foto y valor categórico y continuo por separado; configuración y métricas originales. |

ResNet18 y YOLO + regresión v2 Mac ya tienen predicciones locales. Falta conciliar dos archivos de validación con contenido binario distinto antes de llamar a sus resultados una comparación foto a foto idéntica.

**Entrega al análisis:** rutas locales o de una carpeta privada autorizada, no copias en el repositorio. Se puede compartir primero solo la ruta y el esquema de columnas; el equipo revisará datos e identidades antes de publicar cualquier agregado.
