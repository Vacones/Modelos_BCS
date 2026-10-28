# Modelos BCS — Vacones

Repositorio común del equipo Vacones para desarrollar y comparar modelos de estimación del **Body Condition Score (BCS)** de vacas a partir de imágenes.

Este repositorio publica únicamente código, pruebas, ejemplos ficticios y documentación técnica. Los datasets, fotografías, etiquetas reales, identificadores, resultados por animal y modelos entrenados permanecen fuera de GitHub.

## Organización

| Área | Propósito | Estado documentado |
|---|---|---|
| [`modelos/sin_cajas_resnet`](modelos/sin_cajas_resnet/) | Clasificación conjunta por vaca con fotografías completas y ResNet18. | Implementación inicial disponible y probada. |
| [`modelos/con_cajas_yolo`](modelos/con_cajas_yolo/) | Modelo basado en *bounding boxes* y YOLO. Roboflow se utiliza como herramienta de etiquetado. | Espacio preparado; implementación pendiente de integración. |
| [`evaluacion/comparacion_modelos`](evaluacion/comparacion_modelos/) | Comparación reproducible de los modelos aprobados. | Etapa posterior; sin resultados comparativos aprobados. |

El dataset chino se mantiene como fuente externa diferida. Su posible incorporación dependerá de los resultados de los modelos principales y no forma parte del contenido público de este repositorio.

## Flujo de colaboración

1. Crear una rama para el área correspondiente:
   - `modelo/sin-cajas-resnet`
   - `modelo/con-cajas-yolo`
   - `evaluacion/comparacion-modelos`
2. Trabajar únicamente dentro de la carpeta del modelo o evaluación correspondiente.
3. Ejecutar las pruebas aplicables y revisar que no se incluyan archivos privados.
4. Abrir un Pull Request para integrar los cambios en `main`.
5. Integrar únicamente versiones revisadas y funcionales.

Las reglas completas están en [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Modelo disponible

Para instalar, validar o entrenar el modelo sin cajas, consulte [`modelos/sin_cajas_resnet/README.md`](modelos/sin_cajas_resnet/README.md).

## Uso responsable

Los modelos de este repositorio son prototipos académicos. Sus resultados no deben utilizarse como diagnóstico veterinario ni como única base para decisiones clínicas o productivas.
