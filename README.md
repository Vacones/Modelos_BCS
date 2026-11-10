# Modelos BCS — Vacones

Repositorio común del equipo Vacones para desarrollar y comparar modelos de estimación del **Body Condition Score (BCS)** de vacas a partir de imágenes.

Aquí se publica código, pruebas, ejemplos ficticios, documentación técnica y algunos resultados agregados. Los datasets, fotografías, etiquetas e identificadores reales, predicciones por animal y modelos entrenados no deben añadirse al repositorio.

## Organización

| Área | Contenido | Estado |
|---|---|---|
| [`modelos/sin_cajas_resnet`](modelos/sin_cajas_resnet/) | ResNet18 con fotografías completas y predicción conjunta por vaca. | Baseline entrenado y evaluado; código y pruebas disponibles. |
| [`modelos/con_cajas_yolo`](modelos/con_cajas_yolo/) | YOLO + EfficientNet y una variante YOLO + DINOv2/LDL. | Código y pruebas disponibles; cada implementación tiene su propio entorno. |
| [`modelos/con_cajas_ordinal`](modelos/con_cajas_ordinal/) | EfficientNet-B0 + CORAL con recorte de grupa. | Código y notebooks disponibles; faltan pruebas automatizadas propias. |
| [`modelos/con_cajas_yolo_regresion`](modelos/con_cajas_yolo_regresion/) | YOLO26n + EfficientNet-B0 para estimar BCS continuo. | Notebooks de entrenamiento v2 y evaluación disponibles; falta verificar una ejecución limpia y añadir pruebas. |
| [`evaluacion/comparacion_modelos`](evaluacion/comparacion_modelos/) | Comparación de candidatos bajo un protocolo común. | Pendiente; no hay una comparación final aprobada. |


## Flujo de colaboración

1. Crear una rama para el área correspondiente, por ejemplo:
   - `modelo/sin-cajas-resnet`
   - `modelo/con-cajas-yolo`
   - `modelo/con-cajas-ordinal`
   - `modelo/con-cajas-yolo-regresion`
   - `evaluacion/comparacion-modelos`
2. Trabajar dentro de la carpeta del modelo o evaluación correspondiente.
3. Ejecutar las pruebas aplicables y revisar que no se incluyan archivos privados ni salidas guardadas en notebooks.
4. Abrir un Pull Request para integrar los cambios en `main`.
5. Integrar versiones revisadas y funcionales.

Las reglas completas están en [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Guías por modelo

- [Modelo sin cajas: ResNet18](modelos/sin_cajas_resnet/README.md)
- [Modelo con cajas: YOLO + EfficientNet](modelos/con_cajas_yolo/README.md)
- [Variante con cajas para Mac: DINOv2 + LDL](modelos/con_cajas_yolo/dinov2_ldl_mac/README.md)
- [Modelo ordinal: EfficientNet-B0 + CORAL](modelos/con_cajas_ordinal/README.md)
- [Modelo con cajas y regresión continua](modelos/con_cajas_yolo_regresion/README.md)

## Uso responsable

Estos modelos son prototipos académicos. Sus resultados no deben utilizarse como diagnóstico veterinario ni como única base para decisiones clínicas o productivas.
