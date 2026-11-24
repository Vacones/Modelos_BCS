# Guía de contribución

## Principio de organización

El repositorio se organiza por modelo, no por integrante. Cada contribución debe permanecer dentro de la carpeta del modelo que modifica, salvo los cambios comunes de documentación o configuración.

## Ramas de trabajo

Use una rama descriptiva basada en el área:

- `modelo/sin-cajas-resnet`
- `modelo/con-cajas-yolo`
- `evaluacion/comparacion-modelos`

No desarrolle directamente sobre `main`. Esa rama debe contener únicamente versiones revisadas y funcionales.

Suba los cambios con `git` desde una rama y un Pull Request. No use la opción de subir archivos de la página de GitHub, porque escribe directo en `main` y no respeta `.gitignore`, así que puede publicar fotos, etiquetas o pesos sin que nadie lo revise.

## Mensajes de commit

Escriba los mensajes en español, en modo imperativo o infinitivo y con una categoría breve. Ejemplos:

```text
modelo: agregar validación de entradas para YOLO
pruebas: cubrir imágenes sin anotaciones
documentación: explicar la ejecución local
corrección: evitar fuga de vacas entre particiones
```

Cada commit debe representar un cambio coherente y verificable.

## Pruebas

Ejecute las pruebas del modelo que modificó antes de abrir un Pull Request. Para el modelo sin cajas, desde la raíz:

```bash
python -m pytest modelos/sin_cajas_resnet/tests -q
```

El Pull Request debe indicar qué se probó, qué cambió y qué limitaciones permanecen.

## Privacidad y seguridad

Antes de crear un commit, ejecute `git status` y revise el contenido preparado con `git diff --cached`.

No publique:

- Fotografías o datasets reales.
- Manifiestos, particiones o etiquetas reales.
- Identificadores reales de animales.
- Exportaciones privadas de Roboflow.
- Credenciales, tokens, llaves o archivos `.env`.
- Checkpoints, predicciones o resultados por animal.
- Rutas locales, documentos académicos privados o datos personales.

Si un cambio necesita alguno de estos materiales para ejecutarse, documente cómo proporcionarlo localmente sin incorporarlo al repositorio.

## Pesos y modelos entrenados

Los pesos se guardan en el Drive del equipo, en una carpeta por modelo y versión, compartida solo con el equipo y los profesores. El README del modelo debe incluir el enlace, la ruta donde va cada archivo dentro de la carpeta del modelo, su tamaño y su SHA-256, y un archivo `HUELLAS.txt` que se pueda comprobar con `sha256sum -c`. Si los pesos cambian, se publica una versión nueva en otra carpeta y no se reemplaza la anterior, para que los resultados reportados se puedan reproducir.

## Pull Requests

Un Pull Request debe:

1. Concentrarse en un solo modelo o propósito.
2. Conservar la separación de datos por vaca cuando corresponda.
3. Incluir o actualizar pruebas cuando cambie el comportamiento.
4. Actualizar la documentación del modelo si cambia su uso.
5. Pasar las pruebas aplicables antes de integrarse en `main`.

## Mínimo por modelo

Cada carpeta de `modelos/` debe indicar en su README:

- Qué predice y cuál es la unidad de evaluación: foto o vaca.
- Qué datos privados necesita y cómo proporcionarlos fuera del repositorio.
- Cómo preparar el entorno y ejecutar entrenamiento y evaluación.
- Qué código, pruebas y artefactos faltan para reproducir los resultados publicados.
- Si el conjunto de prueba ya fue consultado.

Cada ejecución debe registrar el commit, la configuración, la versión o huella de los datos, las particiones, el checkpoint elegido, las métricas con sus denominadores y las abstenciones. Los ejemplos y pruebas del repositorio usarán datos ficticios. Los datos, pesos y resultados individuales permanecerán fuera de GitHub.

Un notebook puede servir para ejecutar el flujo, pero su README debe distinguir claramente qué etapas contiene realmente. No se presentará una evaluación como entrenamiento reproducible si el código de entrenamiento no está disponible.
