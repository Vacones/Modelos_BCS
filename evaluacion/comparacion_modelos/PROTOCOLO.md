# Protocolo de comparación maestra BCS

Estado: borrador de trabajo, 10 de noviembre de 2026. No implica selección de un modelo.

## Candidatos

Se registran por separado ResNet18 sin cajas, YOLO + EfficientNet de clasificación, EfficientNet + CORAL ordinal, YOLO + DINOv2/LDL y YOLO + EfficientNet de regresión continua. Una variante histórica y una ejecución nueva son observaciones distintas.

## Comparación principal

La unidad principal es **vaca**: primero se obtiene una estimación conjunta por animal y luego cada vaca aporta el mismo peso al MAE y a las tolerancias. Para un sistema que puede abstenerse, se informa (a) vacas elegibles, (b) vacas con al menos una predicción, (c) fotos elegibles y (d) fotos con predicción. El MAE condicionado se calcula solo sobre vacas cubiertas y siempre aparece junto a la cobertura. La comparación pareada usa las mismas vacas; cuando difieren las fotos utilizadas se indica expresamente.

Las métricas secundarias por fotografía sirven para diagnosticar el flujo y permiten comparar informes históricos que solo reportaron esa unidad. Se muestran aparte de las métricas por vaca. En tolerancias globales, una abstención cuenta como no acierto; en el MAE no se le imputa un BCS artificial.

Métricas: MAE, dentro de ±0.25 y ±0.50 BCS, cobertura y abstenciones; RMSE y sesgo cuando existan predicciones individuales. Exactitud y F1 macro son secundarios para salidas categóricas; se informa el catálogo, el soporte de cada clase y si el F1 excluye abstenciones. Si una salida continua se convierte a categoría, la regla debe fijarse y anotarse antes de calcular F1. No se compara directamente MAE continuo con MAE categórico redondeado. Precisión, recall y mAP del detector forman un panel separado y no prueban calidad BCS.

La referencia constante se fija con la mediana de **train** de cada protocolo. Se muestra en el mismo conjunto y con el mismo denominador que el candidato. Para MAE condicionado se calcula además sobre las mismas vacas o fotos cubiertas.

## Condiciones de comparabilidad

Una diferencia directa requiere identidad de vacas y BCS coincidentes, partición y fotos verificadas, misma unidad, mismo catálogo o escala de salida, mismo origen de cajas y mismo tratamiento de abstenciones. Las cajas anotadas (evaluación del predictor) y las predichas (sistema completo) se presentan por separado. Se conservan hashes o manifiestos privados para verificar las fuentes; no se publican los identificadores.

El conjunto de validación ya intervino en selección de modelos y umbrales; sus cifras son de desarrollo. Hay consultas históricas de `test` en al menos ResNet18 y CORAL, y en el informe histórico de YOLO + regresión. Ningún resultado posterior de `test` se llamará prueba independiente sin un nuevo conjunto de vacas. Los intervalos bootstrap por vaca, si se calculan, serán descriptivos y se advertirá que no corrigen la selección repetida.

## Trazabilidad mínima por ejecución

Guardar localmente: modelo y variante, commit, entorno, configuración, huella de datos, particiones, checkpoint y huella, origen de cajas, catálogo/salida, regla de agregación, denominadores, cobertura, métricas y ruta privada a predicciones. Los documentos de Drive o las cifras transcritas se identifican como **documentadas**, no como recalculadas, si falta el artefacto individual.

## Gráficas

1. Cobertura y tolerancia global ±0.25 por fotografía, con el mismo denominador y etiqueta de salida continua/categórica.
2. MAE condicionado por fotografía, separado por tipo de salida y con `n` visible; los faltantes quedan como tales.
3. MAE y cobertura por vaca, solo para ejecuciones con predicciones verificables por animal.
4. Diferencia pareada de error por vaca y errores por nivel BCS cuando estén disponibles los archivos privados de todos los candidatos pertinentes.
5. Matriz de confusión y F1 por clase para candidatos categóricos bajo un catálogo común; mAP del detector en panel aparte.

Cada figura indicará fuente, partición, unidad, origen de cajas y limitaciones. No habrá una clasificación única de modelos basada en fuentes incompletas.

## Publicación

Las predicciones individuales, fotos, etiquetas, identificadores y checkpoints quedan fuera de Git y Drive público. Solo se considerarán para publicación las tablas y figuras agregadas revisadas. Los resultados locales del equipo se procesan en la computadora autorizada; no se suben al repositorio como entradas del comparador.
