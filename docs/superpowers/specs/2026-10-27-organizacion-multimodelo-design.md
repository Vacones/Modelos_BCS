# Diseño de organización multimodelo

## Propósito

Convertir `Vacones/Modelos_BCS` en el repositorio común del equipo para desarrollar, revisar y comparar modelos de estimación de BCS, conservando el modelo sin cajas ya funcional y preparando espacios claros para el modelo con cajas y la comparación posterior.

## Alcance aprobado

El repositorio tendrá dos modelos principales:

1. `sin_cajas_resnet`: clasificación conjunta por vaca a partir de fotografías completas, usando ResNet.
2. `con_cajas_yolo`: detección o análisis con *bounding boxes* usando YOLO. Roboflow será una herramienta de etiquetado, no un tercer modelo.

La comparación entre modelos será una etapa posterior en `evaluacion/comparacion_modelos`. El dataset chino seguirá siendo una fuente externa y no se incorporará automáticamente al código ni se publicará en GitHub.

## Estructura objetivo

```text
Modelos_BCS/
├── README.md
├── CONTRIBUTING.md
├── .gitignore
├── docs/
├── modelos/
│   ├── sin_cajas_resnet/
│   │   ├── README.md
│   │   ├── entrenar_sin_cajas.py
│   │   ├── preparar_datos.py
│   │   ├── requirements.txt
│   │   ├── modelo_sin_cajas/
│   │   ├── examples/
│   │   └── tests/
│   └── con_cajas_yolo/
│       └── README.md
└── evaluacion/
    └── comparacion_modelos/
        └── README.md
```

## Responsabilidad de los archivos

- El `README.md` raíz será el índice del proyecto, explicará el estado de cada modelo y enlazará sus instrucciones específicas.
- `CONTRIBUTING.md` documentará ramas, Pull Requests, pruebas, mensajes de commit en español y restricciones de privacidad.
- El README de `sin_cajas_resnet` conservará las instrucciones técnicas existentes, ajustadas a su nueva ruta.
- El README de `con_cajas_yolo` será una plantilla sobria: propósito, estado pendiente, contenido esperado y restricciones de seguridad. No atribuirá avances, resultados ni propiedad personal.
- El README de `comparacion_modelos` explicará los insumos y criterios esperados sin inventar resultados.

## Flujo de colaboración

- `main` contendrá únicamente versiones revisadas y funcionales.
- Las ramas de trabajo recomendadas serán `modelo/sin-cajas-resnet`, `modelo/con-cajas-yolo` y `evaluacion/comparacion-modelos`.
- Cada cambio entrará mediante Pull Request después de ejecutar las pruebas aplicables.
- La organización permanente será por modelo, no por persona.
- No se configurará `CODEOWNERS` hasta contar con los usuarios de GitHub y responsables confirmados.

## Historial y lenguaje

Los cuatro commits actuales se reescribirán en español antes de que el equipo clone el repositorio. Sus identificadores cambiarán y la rama remota se actualizará con `--force-with-lease`.

Mensajes aprobados:

1. `configuración: inicializar la estructura segura del proyecto`
2. `modelo: agregar el flujo de entrenamiento BCS`
3. `pruebas: agregar cobertura reproducible del flujo`
4. `documentación: aclarar las pruebas con datos privados`

La reorganización se registrará después como `organización: estructurar el repositorio por modelo`.

## Seguridad y exclusiones

No se publicarán datasets, fotografías, manifiestos reales, identificadores reales, credenciales, archivos `.env`, anotaciones privadas, checkpoints, predicciones ni resultados sensibles. La reorganización deberá mantener y ampliar las reglas de `.gitignore` para que funcionen dentro de carpetas anidadas.

## Criterios de aceptación

- El código actual funciona desde `modelos/sin_cajas_resnet`.
- Las pruebas del modelo sin cajas conservan su resultado esperado.
- Las carpetas de YOLO y comparación contienen únicamente documentación inicial, sin afirmar trabajo no realizado.
- El README raíz y `CONTRIBUTING.md` explican el flujo completo del equipo.
- Todos los mensajes de commit visibles están en español.
- La rama local y `origin/main` terminan en el mismo commit.
- No aparece ningún archivo sensible entre los archivos versionados.
