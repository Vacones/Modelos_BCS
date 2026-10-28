# Plan de implementación de la organización multimodelo

> **Para trabajadores con agentes:** SUBHABILIDAD OBLIGATORIA: usar `superpowers:subagent-driven-development` (recomendado) o `superpowers:executing-plans` para implementar este plan tarea por tarea. Los pasos usan casillas (`- [ ]`) para dar seguimiento.

**Objetivo:** Organizar `Vacones/Modelos_BCS` por modelo, conservar funcional el modelo sin cajas, preparar los espacios de YOLO y comparación, y dejar todo el historial visible en español.

**Arquitectura:** La raíz será un índice y punto de gobierno; cada implementación vivirá debajo de `modelos/`, y los análisis transversales debajo de `evaluacion/`. El modelo sin cajas se moverá como una unidad para mantener juntos código, pruebas, ejemplo y dependencias.

**Tecnologías:** Git, GitHub, Markdown, Python 3.12, pytest.

**Especificación:** `docs/superpowers/specs/2026-10-27-organizacion-multimodelo-design.md`

## Restricciones globales

- No publicar datasets, fotografías, manifiestos reales, etiquetas privadas, credenciales, checkpoints ni resultados sensibles.
- No inventar responsables, métricas, avances ni decisiones técnicas del modelo YOLO o de la comparación.
- Mantener `main` como rama estable y usar mensajes de commit en español.
- Reescribir el remoto únicamente con `git push --force-with-lease`, nunca con `--force`.
- No configurar `CODEOWNERS` hasta recibir usuarios y responsables confirmados.

## Enfoque de revisión

- Las rutas anidadas de resultados deben seguir ignoradas por Git.
- La ejecución de pruebas desde la raíz debe encontrar correctamente el paquete movido.
- Los README iniciales no deben presentar el modelo YOLO ni la comparación como terminados.
- La reescritura del historial no debe perder ninguno de los cuatro cambios originales.
- El commit local final debe coincidir exactamente con `origin/main`.

---

### Task 1: Traducir el historial existente

**Archivos:**
- Crear temporalmente: `.git/codex_reword_commits.sh`
- Modificar: historial Git de los cuatro commits existentes

**Interfaces:**
- Consume: los commits `f4a4c3a`, `4601889`, `c22e2d5` y `e1f5fec`.
- Produce: cuatro commits equivalentes, en el mismo orden y con mensajes en español.

- [ ] **Paso 1: Guardar los identificadores y árboles actuales**

Ejecutar: `git log --reverse --format='%H %T %s'`

Resultado esperado: cuatro líneas que permitan comprobar que la traducción no cambia el contenido de cada commit.

- [ ] **Paso 2: Reescribir los cuatro mensajes desde la raíz**

Usar un editor temporal de Git que sustituya únicamente los asuntos por los cuatro mensajes aprobados en la especificación.

- [ ] **Paso 3: Verificar los asuntos traducidos**

Ejecutar: `git log --reverse --format='%s'`

Resultado esperado: exactamente los cuatro mensajes en español y en el orden aprobado.

- [ ] **Paso 4: Verificar que los árboles de los cuatro commits no cambiaron**

Ejecutar: `git log --reverse --format='%T'`

Resultado esperado: los mismos cuatro identificadores de árbol guardados en el Paso 1.

- [ ] **Paso 5: Eliminar el editor temporal**

Eliminar `.git/codex_reword_commits.sh` y comprobar que no queda ningún auxiliar de autenticación o reescritura dentro de `.git`.

### Task 2: Reorganizar el repositorio por modelo

**Archivos:**
- Mover: `entrenar_sin_cajas.py` a `modelos/sin_cajas_resnet/entrenar_sin_cajas.py`
- Mover: `preparar_datos.py` a `modelos/sin_cajas_resnet/preparar_datos.py`
- Mover: `requirements.txt` a `modelos/sin_cajas_resnet/requirements.txt`
- Mover: `modelo_sin_cajas/` a `modelos/sin_cajas_resnet/modelo_sin_cajas/`
- Mover: `examples/` a `modelos/sin_cajas_resnet/examples/`
- Mover: `tests/` a `modelos/sin_cajas_resnet/tests/`
- Mover y modificar: `README.md` a `modelos/sin_cajas_resnet/README.md`
- Crear: `README.md`
- Crear: `CONTRIBUTING.md`
- Crear: `modelos/con_cajas_yolo/README.md`
- Crear: `evaluacion/comparacion_modelos/README.md`
- Modificar: `.gitignore`
- Incluir: `docs/superpowers/specs/2026-10-27-organizacion-multimodelo-design.md`
- Incluir: `docs/superpowers/plans/2026-10-27-organizacion-multimodelo.md`

**Interfaces:**
- Consume: el modelo sin cajas funcional ubicado actualmente en la raíz.
- Produce: una raíz multimodelo y una unidad ejecutable en `modelos/sin_cajas_resnet`.

- [ ] **Paso 1: Mover el modelo sin cajas como una unidad**

Usar `git mv` para conservar el historial de los archivos y dejar en la raíz solo archivos comunes del repositorio.

- [ ] **Paso 2: Escribir el README raíz y la guía de contribución**

El README debe enumerar los dos modelos, el estado real y la etapa de comparación. `CONTRIBUTING.md` debe exigir ramas por modelo, Pull Requests, pruebas, mensajes en español y revisión de archivos sensibles.

- [ ] **Paso 3: Escribir las plantillas de YOLO y comparación**

Cada README debe explicar propósito, estado pendiente, contenido esperado y prohibiciones de publicación. No debe incluir autores, métricas ni decisiones aún no confirmadas.

- [ ] **Paso 4: Ajustar las instrucciones del modelo sin cajas**

Actualizar los comandos para ejecutarlos desde `modelos/sin_cajas_resnet` y mantener sin cambios el comportamiento técnico documentado.

- [ ] **Paso 5: Generalizar las exclusiones para carpetas anidadas**

Asegurar que `resultados_sin_cajas/`, checkpoints, datasets, imágenes y manifiestos privados se ignoren a cualquier profundidad.

- [ ] **Paso 6: Ejecutar las pruebas desde la raíz**

Ejecutar: `python -m pytest modelos/sin_cajas_resnet/tests -q`

Resultado esperado sin dataset privado: `33 passed, 3 skipped`.

- [ ] **Paso 7: Verificar las exclusiones anidadas**

Ejecutar: `git check-ignore --no-index modelos/sin_cajas_resnet/resultados_sin_cajas/prueba/best_model.pt`

Resultado esperado: la ruta aparece como ignorada.

- [ ] **Paso 8: Verificar que no se versionan artefactos sensibles**

Revisar `git status --short`, `git diff --cached` y `git ls-files` antes del commit.

- [ ] **Paso 9: Crear el commit de organización**

```bash
git add .gitignore README.md CONTRIBUTING.md docs modelos evaluacion
git commit -m "organización: estructurar el repositorio por modelo"
```

### Task 3: Publicar y verificar el historial reorganizado

**Archivos:**
- Modificar: `origin/main` en GitHub

**Interfaces:**
- Consume: los cinco commits locales en español y todas las verificaciones aprobadas.
- Produce: `origin/main` organizado, verificable y sin credenciales temporales.

- [ ] **Paso 1: Comprobar el estado previo a publicación**

Ejecutar: `git status --short --branch`, `git log --reverse --format='%s'` y la suite completa.

Resultado esperado: árbol limpio, cinco mensajes en español y pruebas aprobadas.

- [ ] **Paso 2: Actualizar el remoto con protección contra sobrescrituras inesperadas**

Ejecutar: `git push --force-with-lease origin main`

Resultado esperado: GitHub acepta el historial y no reporta cambios remotos ajenos.

- [ ] **Paso 3: Comparar los commits local y remoto**

Ejecutar: `git rev-parse HEAD` y `git ls-remote origin refs/heads/main`.

Resultado esperado: ambos identificadores son iguales.

- [ ] **Paso 4: Inspeccionar la lista pública de archivos**

Ejecutar: `git ls-tree -r --name-only origin/main`.

Resultado esperado: solo código, pruebas, ejemplos ficticios y documentación aprobada.

- [ ] **Paso 5: Confirmar limpieza de credenciales**

Comprobar que no existen tokens, llaves de despliegue, scripts de autenticación ni auxiliares de reescritura creados para la operación.
