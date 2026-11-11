# Comparación de modelos BCS

Esta carpeta reunirá la comparación de los candidatos desarrollados en `modelos/` y servirá como base para la fase de evaluación de CRISP-DM. La comparación debe mostrar qué resultados son compatibles entre sí, qué limitaciones tienen y cómo responden a los objetivos de negocio del proyecto.

## Estado

Todavía no existe una comparación maestra aprobada ni se ha seleccionado un modelo para uso operativo. Los informes actuales utilizan distintas unidades de análisis, fuentes de cajas y conjuntos de evaluación. Sus cifras no deben ordenarse en una sola clasificación sin comprobar primero que fueron obtenidas bajo condiciones comparables.

## Candidatos

La revisión incluirá las implementaciones disponibles en:

- `modelos/sin_cajas_resnet/`
- `modelos/con_cajas_yolo/`, incluidas sus variantes EfficientNet y DINOv2 + LDL
- `modelos/con_cajas_ordinal/`
- `modelos/con_cajas_yolo_regresion/`

Cada versión y ejecución se registrará por separado. Una carpeta puede contener más de un candidato.

## Información mínima por evaluación

Para cada resultado se registrará:

- versión o commit del código y checkpoint utilizado;
- configuración y versión o huella de los datos autorizados;
- partición evaluada y si ya se había consultado para tomar decisiones;
- número de vacas e imágenes elegibles;
- unidad de análisis: vaca o fotografía;
- origen de las cajas: anotadas o predichas;
- cobertura, abstenciones y sus causas;
- métricas, denominadores y regla de agregación;
- resultado de una referencia constante calculada sin usar las etiquetas de prueba;
- enlace al informe o artefacto que respalda la cifra.

Las métricas por fotografía no se presentarán como si correspondieran a animales independientes. El MAE calculado solo entre predicciones emitidas siempre se acompañará de la cobertura.

## Reglas de comparación

Una diferencia directa entre modelos requiere comprobar que coincidan la partición, las observaciones, la unidad de análisis, el origen de las cajas, la definición de la salida y el tratamiento de abstenciones. Si estas condiciones difieren, se mostrarán los resultados por separado y se explicará la limitación.

Se distinguirá el desempeño del predictor con cajas anotadas del desempeño del sistema completo con detección automática. Cuando sea posible, la evaluación principal del BCS conjunto se expresará por vaca; las métricas por fotografía servirán también para diagnosticar errores.

## Evaluación y decisión

La comparación técnica alimentará una evaluación posterior frente a los criterios de éxito definidos en Business Understanding. Esa evaluación deberá documentar los hallazgos, revisar las limitaciones del proceso y justificar si corresponde continuar una iteración, reunir nuevos datos o aprobar un candidato. Completar un entrenamiento o superar a otro modelo en una sola métrica no constituye aprobación para uso operativo.

## Privacidad

Esta carpeta no debe contener datasets, fotografías, manifiestos o etiquetas reales, identificadores de animales, predicciones individuales, credenciales ni checkpoints. Cualquier resumen agregado que se publique debe revisarse antes de integrarlo en GitHub.

## Corte preliminar de validación

Se preparó un [resumen descriptivo con gráficas](RESUMEN_PRELIMINAR.md), un [inventario de evidencia](INVENTARIO.md), un [protocolo](PROTOCOLO.md) y una [lista de archivos pendientes](ARCHIVOS_PENDIENTES.md). Las cifras agregadas están en [datos_agregados_validacion.csv](datos_agregados_validacion.csv). Este corte no selecciona un modelo: faltan predicciones individuales de tres enfoques para la comparación pareada por vaca.

La [preparación de las corridas nuevas](PLAN_EJECUCION.md) fija las verificaciones y las salidas privadas requeridas. Una [comprobación pareada local](COMPARACION_LOCAL.md) usa únicamente fotografías byte idénticas de ResNet18 y YOLO + regresión Mac; sus cifras son diagnósticas. `evaluador_comun.py` y `adaptadores_locales.py` realizan esa comprobación sin escribir identificadores ni predicciones individuales en Git.
