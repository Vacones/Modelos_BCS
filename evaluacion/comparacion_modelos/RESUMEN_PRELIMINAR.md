# Comparación maestra BCS: corte preliminar

**Fecha:** 10 de noviembre de 2026. **Partición:** validación. **Estado:** inventario descriptivo; todavía no es una selección de modelo.

## Qué se puede comparar hoy

Los cinco enfoques tienen alguna cifra de validación sobre 36 fotografías, pero la evidencia no es uniforme. Las predicciones individuales de ResNet18, YOLO + regresión v2 Mac y la **nueva corrida** DINOv2/LDL se encontraron localmente y permitieron recalcular métricas. YOLO + EfficientNet y ordinal v4 siguen documentados mediante informes, sin archivos privados de predicciones disponibles aquí para reproducir su comparación por vaca. Véase [INVENTARIO.md](INVENTARIO.md) para la procedencia de cada fila.

| Sistema | Salida | Fotos con predicción / 36 | Dentro de ±0.25 / 36 | MAE en fotos con predicción |
|---|---|---:|---:|---:|
| Constante 3.0 | Categoría | 36 | 21 | 0.3472 |
| ResNet18 sin cajas | Categoría | 36 | 31 | 0.2292 |
| YOLO + EfficientNet | Categoría | 30 | 16 | 0.4917 |
| YOLO + CORAL ordinal v4 | Categoría | 29 | 21 | No documentado para el flujo automático |
| YOLO + DINOv2/LDL histórico | Categoría | 28 | 25 | 0.2143 |
| YOLO + DINOv2/LDL **nueva corrida** | Categoría | 24 | 21 | 0.2083 |
| La misma corrida DINO, esperanza continua | Continua | 24 | 18 | 0.2014 |
| YOLO + regresión v2 Mac | Continua | 27 | 8 | 0.3299 |

El acierto global incluye las abstenciones en el denominador; el MAE solo incluye fotografías con predicción. La cifra de ResNet18 es secundaria, por fotografía; el modelo se diseñó para una predicción conjunta por vaca. Las salidas continuas no se deben ordenar directamente contra el MAE de categorías. Los flujos con cajas usan detecciones predichas en estas filas. Las dos filas de la nueva corrida DINO son **dos decodificaciones del mismo modelo**, no dos entrenamientos. La histórica tuvo pesos y condiciones diferentes. Las 36 fotografías no garantizan todavía identidad de archivo en todos los enfoques.

![Cobertura y tolerancia de validación](figuras/cobertura_y_tolerancia_val.png)

![MAE condicionado de validación](figuras/mae_condicionado_val.png)

Los datos de ambas figuras están en [datos_agregados_validacion.csv](datos_agregados_validacion.csv). Se regeneran con `python -m evaluacion.comparacion_modelos.generar_graficas` desde la carpeta raíz del repositorio.

## Unidad principal: vaca

Hay resultados por vaca verificables para tres sistemas. ResNet18 obtuvo MAE nativo **0.150** sobre 10 vacas de validación. El flujo YOLO + regresión v2 Mac cubrió **8 de 10** vacas, con MAE **0.2214** sobre esas 8. La nueva corrida DINO cubrió **8 de 10**, con MAE por vaca **0.1920** categórico y **0.1949** continuo bajo la media de predicciones por foto. Una comparación estricta entre DINO continuo y regresión usa **20 fotos de 7 vacas** donde ambos predijeron: MAE **0.1936** frente a **0.2332**. Son cifras de desarrollo y del subconjunto cubierto; no justifican una conclusión de superioridad. ResNet18 usa otra regla nativa de agregación y tiene dos fotos aún no conciliadas con las otras fuentes. Véase [DINO_CORRIDA_01.md](DINO_CORRIDA_01.md) y [COMPARACION_LOCAL.md](COMPARACION_LOCAL.md).

## Lectura crítica

- El detector de YOLO + regresión alcanzó métricas altas de cajas en validación, pero su sistema completo emitió predicción BCS en 27 de 36 fotos. La calidad del detector y la calidad BCS son resultados distintos.
- La corrida histórica DINO cubrió 28 fotos; la nueva cubrió 24. Esta última se abstuvo en 12 (9 asociaciones ambiguas y 3 sin par válido). Su error condicionado y cobertura deben leerse juntos; no se atribuye la diferencia entre corridas a una sola causa.
- El informe ordinal v4 aporta cobertura y aciertos del flujo automático, pero no el MAE condicionado de esa ejecución. No se sustituye por el MAE con cajas anotadas ni por cifras de validación cruzada o `test`.
- Solo se verificó coincidencia de las 10 vacas y sus BCS entre ResNet18 y la regresión Mac. De 36 fotos, 34 coincidieron byte por byte; quedan dos archivos por conciliar. No se infiere equivalencia de todas las fotografías por compartir etiqueta o vaca.
- `test` fue consultado antes en algunas líneas de trabajo. Las cifras de este documento son de **validación de desarrollo** y no una prueba final independiente.

## Próximo cierre

Completar las corridas nuevas de EfficientNet y ordinal, obtener los archivos privados indicados en [ARCHIVOS_PENDIENTES.md](ARCHIVOS_PENDIENTES.md) y verificar su identidad con el manifiesto local. Después generar diferencias pareadas, errores por BCS y matrices de confusión para salidas comparables. La elección de un modelo debe esperar ese análisis y una revisión de la procedencia de cada ejecución.
