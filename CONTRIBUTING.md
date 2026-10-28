# Guía de contribución

## Principio de organización

El repositorio se organiza por modelo, no por integrante. Cada contribución debe permanecer dentro de la carpeta del modelo que modifica, salvo los cambios comunes de documentación o configuración.

## Ramas de trabajo

Use una rama descriptiva basada en el área:

- `modelo/sin-cajas-resnet`
- `modelo/con-cajas-yolo`
- `evaluacion/comparacion-modelos`

No desarrolle directamente sobre `main`. Esa rama debe contener únicamente versiones revisadas y funcionales.

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

- fotografías o datasets reales;
- manifiestos, particiones o etiquetas reales;
- identificadores reales de animales;
- exportaciones privadas de Roboflow;
- credenciales, tokens, llaves o archivos `.env`;
- checkpoints, predicciones o resultados por animal;
- rutas locales, documentos académicos privados o datos personales.

Si un cambio necesita alguno de estos materiales para ejecutarse, documente cómo proporcionarlo localmente sin incorporarlo al repositorio.

## Pull Requests

Un Pull Request debe:

1. concentrarse en un solo modelo o propósito;
2. conservar la separación de datos por vaca cuando corresponda;
3. incluir o actualizar pruebas cuando cambie el comportamiento;
4. actualizar la documentación del modelo si cambia su uso;
5. pasar las pruebas aplicables antes de integrarse en `main`.

No se asignarán propietarios automáticos mediante `CODEOWNERS` hasta que el equipo confirme responsables y usuarios de GitHub.
