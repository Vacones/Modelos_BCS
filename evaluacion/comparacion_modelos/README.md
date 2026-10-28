# Comparación de modelos

Espacio para comparar de manera reproducible el modelo `sin_cajas_resnet` y el modelo `con_cajas_yolo` cuando ambos cuenten con versiones revisadas y evaluaciones compatibles.

## Estado

No existen resultados comparativos aprobados. Esta etapa comenzará después de definir una versión evaluable de cada modelo y un protocolo común que evite fugas de información entre vacas.

## Contenido esperado

La futura comparación deberá documentar:

- versiones exactas de los modelos comparados;
- particiones y unidad de evaluación por vaca;
- métricas comunes y reglas para interpretarlas;
- condiciones de entrenamiento y evaluación;
- limitaciones y diferencias entre las entradas de cada modelo;
- criterio para decidir si se estudia posteriormente la incorporación del dataset chino.

Los resultados agregados que se publiquen deberán revisarse para evitar exponer identificadores, etiquetas o predicciones individuales.

## Restricciones

Esta carpeta no debe contener datasets, imágenes, manifiestos reales, credenciales, checkpoints ni resultados sensibles. Tampoco debe combinar resultados obtenidos con particiones incompatibles o metodologías no documentadas.
