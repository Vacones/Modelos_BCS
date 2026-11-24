# ¿Ayudan los datasets externos al modelo BCS de CAETEC?

Fecha: 2026-11-12. Modelo base: recorte de grupa con 10 % de margen, DINOv2 ViT-S/14 congelado (224, CLS + media de parches, foto espejada) y ridge o CORAL+PCA.
Evaluación **solo sobre vacas CAETEC** (66 vacas, 331 fotos): GroupKFold por vaca 5×10, hiperparámetros (y peso w de Huang cuando aplica) por GroupKFold(3) interno sobre vacas CAETEC de entrenamiento, MAE por vaca (media de las fotos de la vaca), IC 95 % por bootstrap sobre vacas (2000).
Scripts: `extraer_externos.py`, `evaluar_externos.py`. Números completos en `resultados_externos.json`. Las features se borraron.

## Muestra de Huang (ScienceDB)
3000 imágenes: 150 por (vista, nivel) en las vistas GS, YM, L-i y R-i, y los niveles 3.25, 3.5, 3.75, 4.0 y 4.25. Como máximo 2 por id, sin duplicados MD5. Huang trae 1 caja por imagen (1 de 3000 con más de una). La caja es la región de la grupa, no la vaca completa: cubre la mediana del 23 % del cuadro en GS, L-i y R-i, y el 9 % en YM.
Vistas (revisión visual): **YM** es una cámara cenital en el carril de salida que ve la vaca completa desde atrás y arriba. Es la más parecida a nuestras fotos. **GS** es un primer plano cenital de la grupa, con marcas de pintura. **L-i/R-i** son tomas muy cercanas en diagonal desde arriba, de un costado de la grupa. CAETEC son fotos traseras a la altura del animal.
Rango BCS: Huang va de 3.25 a 4.25 y CAETEC de 2.25 a 3.75. Solo se solapan en 3.25-3.75.

## Tabla (66 vacas CAETEC, ΔMAE pareado contra (a) con la misma cabeza, negativo = mejor)

| config | MAE [IC95] | ±0.25 | ±0.50 | pendiente | AUC ≤2.75 | MAE ≥3.5 (n=13) | MAE ≤2.5 (n=14) | sesgo | ΔMAE vs a [IC95] |
|---|---|---|---|---|---|---|---|---|---|
| constante (mediana) | 0.318 [0.261,0.375] | 0.591 | 0.894 | 0.00 | 0.500 | 0.615 | 0.518 | -0.008 | |
| **a ridge (solo CAETEC)** | 0.210 [0.178,0.245] | 0.842 | 0.985 | 0.49 | 0.952 | 0.363 | 0.249 | -0.002 | ref |
| **a CORAL (solo CAETEC)** | 0.206 [0.174,0.239] | 0.838 | 0.985 | 0.53 | 0.950 | 0.365 | 0.191 | -0.017 | ref |
| b ridge +Huang todo w=0.1 | 0.226 | 0.820 | 0.980 | 0.47 | 0.895 | 0.358 | 0.254 | +0.022 | +0.016 [-0.007,0.039] |
| b ridge +Huang todo w=0.3 | 0.245 | 0.802 | 0.974 | 0.43 | 0.862 | 0.369 | 0.272 | +0.030 | +0.036 [0.009,0.064] |
| b ridge +Huang todo w=1 | 0.275 | 0.759 | 0.953 | 0.37 | 0.816 | 0.382 | 0.306 | +0.046 | +0.065 [0.033,0.101] |
| b ridge +Huang todo w=CV | 0.213 | 0.833 | 0.982 | 0.49 | 0.935 | 0.363 | 0.249 | +0.002 | +0.003 [-0.002,0.009] |
| b CORAL +Huang todo w=0.1 | 0.248 | 0.783 | 0.973 | 0.41 | 0.843 | 0.374 | 0.279 | +0.017 | +0.042 [0.015,0.070] |
| b CORAL +Huang todo w=0.3 | 0.269 | 0.767 | 0.962 | 0.35 | 0.817 | 0.397 | 0.316 | +0.033 | +0.063 [0.030,0.096] |
| b CORAL +Huang todo w=1 | 0.300 | 0.735 | 0.964 | 0.26 | 0.787 | 0.416 | 0.388 | +0.068 | +0.095 [0.057,0.134] |
| b CORAL +Huang todo w=CV | 0.206 | 0.838 | 0.985 | 0.53 | 0.950 | 0.365 | 0.191 | -0.017 | 0.000 (CV eligió w=0 en 50/50) |
| c ridge +YM w=0.1 | 0.223 | 0.830 | 0.977 | 0.47 | 0.896 | 0.358 | 0.260 | +0.011 | +0.013 [-0.007,0.033] |
| c ridge +YM w=0.3 | 0.230 | 0.809 | 0.970 | 0.45 | 0.886 | 0.364 | 0.262 | +0.011 | +0.020 [-0.003,0.045] |
| c ridge +YM w=1 | 0.247 | 0.812 | 0.959 | 0.41 | 0.853 | 0.374 | 0.274 | +0.015 | +0.037 [0.009,0.067] |
| c ridge +YM w=CV | 0.213 | 0.839 | 0.982 | 0.49 | 0.943 | 0.365 | 0.250 | +0.000 | +0.003 [-0.000,0.007] |
| c CORAL +YM w=0.1 | 0.213 | 0.835 | 0.991 | 0.48 | 0.904 | 0.365 | 0.224 | -0.008 | +0.008 [-0.013,0.030] |
| c CORAL +YM w=0.3 | 0.224 | 0.829 | 0.994 | 0.45 | 0.887 | 0.370 | 0.241 | -0.004 | +0.019 [-0.006,0.042] |
| c CORAL +YM w=1 | 0.243 | 0.824 | 0.976 | 0.41 | 0.853 | 0.380 | 0.267 | +0.008 | +0.037 [0.005,0.065] |
| c CORAL +YM w=CV | 0.207 | 0.836 | 0.985 | 0.52 | 0.945 | 0.368 | 0.189 | -0.018 | +0.001 [-0.002,0.005] |
| d ridge todo +dominio w=0.3 | 0.235 | 0.829 | 0.982 | 0.43 | 0.874 | 0.389 | 0.246 | -0.005 | +0.025 [0.004,0.047] |
| d ridge todo +dominio w=1 | 0.253 | 0.791 | 0.964 | 0.38 | 0.837 | 0.417 | 0.262 | -0.009 | +0.044 [0.017,0.072] |
| d ridge todo +dominio w=CV | 0.215 | 0.827 | 0.986 | 0.49 | 0.937 | 0.367 | 0.240 | -0.003 | +0.005 [-0.002,0.013] |
| d ridge YM +dominio w=0.3 | 0.225 | 0.818 | 0.973 | 0.45 | 0.886 | 0.369 | 0.252 | -0.002 | +0.015 [-0.005,0.035] |
| d ridge YM +dominio w=1 | 0.240 | 0.805 | 0.964 | 0.42 | 0.851 | 0.382 | 0.260 | -0.003 | +0.030 [0.005,0.057] |
| d ridge YM +dominio w=CV | 0.212 | 0.842 | 0.982 | 0.49 | 0.941 | 0.364 | 0.246 | -0.001 | +0.003 [-0.002,0.008] |
| d CORAL todo +dominio w=0.3 | 0.249 | 0.809 | 0.965 | 0.36 | 0.875 | 0.432 | 0.267 | -0.015 | +0.043 [0.013,0.072] |
| d CORAL todo +dominio w=1 | 0.268 | 0.805 | 0.967 | 0.28 | 0.845 | 0.475 | 0.304 | -0.011 | +0.063 [0.029,0.096] |
| d CORAL todo +dominio w=CV | 0.206 | 0.836 | 0.986 | 0.53 | 0.950 | 0.366 | 0.189 | -0.016 | +0.001 [-0.001,0.002] |
| d CORAL YM +dominio w=0.3 | 0.219 | 0.848 | 0.991 | 0.45 | 0.898 | 0.387 | 0.231 | -0.022 | +0.013 [-0.010,0.035] |
| d CORAL YM +dominio w=1 | 0.234 | 0.827 | 0.970 | 0.40 | 0.870 | 0.407 | 0.248 | -0.020 | +0.028 [0.000,0.058] |
| d CORAL YM +dominio w=CV | 0.208 | 0.833 | 0.985 | 0.52 | 0.947 | 0.374 | 0.194 | -0.020 | +0.002 [-0.003,0.007] |
| e ridge + calibración lineal | 0.205 [0.174,0.237] | 0.853 | 0.985 | 0.58 | 0.954 | 0.324 | 0.231 | -0.001 | -0.005 [-0.015,0.005] |
| e CORAL + calibración lineal | 0.204 [0.172,0.238] | 0.845 | 0.985 | 0.59 | 0.945 | 0.333 | 0.183 | -0.005 | -0.002 [-0.009,0.005] |

En las variantes w=CV, la CV interna eligió w=0 (no usar Huang) en la gran mayoría de los 50 folds: 50/50 en CORAL con Huang completo y ~40/50 en ridge.

## Transferencia (f)
**De Huang a CAETEC** (ridge entrenado solo con Huang, se predicen las 66 vacas):

| entrenado con | MAE | Spearman | pendiente | AUC ≤2.75 | sesgo medio | MAE vacas ≥3.25 (n=24) | MAE tras quitar el sesgo |
|---|---|---|---|---|---|---|---|
| Huang todo | 0.824 | 0.49 | 0.18 | 0.78 | +0.82 | 0.445 | 0.294 |
| GS | 0.850 | 0.60 | 0.15 | 0.81 | +0.85 | 0.477 | 0.292 |
| YM | 0.788 | 0.44 | 0.15 | 0.72 | +0.79 | 0.400 | 0.299 |
| R-i | 0.771 | 0.48 | 0.15 | 0.75 | +0.77 | 0.389 | 0.294 |
| L-i | 0.841 | 0.18 | 0.04 | 0.63 | +0.84 | 0.417 | 0.322 |

**De CAETEC a Huang** (ridge entrenado con todo CAETEC, se predicen grupos de Huang):

| vista | Spearman (todos los niveles) | Spearman 3.25-3.75 | MAE 3.25-3.75 |
|---|---|---|---|
| todas | 0.04 | -0.04 | 0.42 |
| GS | 0.18 | 0.19 | 0.32 |
| YM | 0.00 | -0.02 | 0.58 |
| L-i | -0.06 | -0.24 | 0.39 |
| R-i | 0.07 | -0.15 | 0.42 |

## Zenodo 20290988 y Criollo/Dryad
- **Zenodo** (sala rotativa): cámara superior/trasera con 3 o más vacas por cuadro, RGB 1920×1080 y profundidad Kinect sin registrar, sin cajas, BCS 2.75-5.0 por pase. Las grupas se ven desde arriba y atrás, pero haría falta detectar y recortar la vaca central. Además, la misma desalineación de escala y vista que en Huang predice que tampoco ayudaría. No se entrenó con él.
- **Criollo/Dryad**: raza de carne, escala 1-9, imágenes DGE (profundidad, gris y bordes) de 224 px desde arriba, ~100 animales. Es incompatible en escala, raza y modalidad. No se usó.

## Conclusión
1. **Huang no ayuda a entrenar.** Con cualquier peso fijo empeora el MAE por vaca de forma monótona (+0.01 a +0.10), comprime más (la pendiente baja de 0.5 a 0.26-0.47) y empeora justo las vacas de 3.5-3.75 que queríamos mejorar. Ni el indicador de dominio ni restringirse a YM lo arreglan. Cuando la CV interna elige el peso, elige no usarlo (w=0).
2. **No sirve para validar.** Nuestro modelo no ordena las vacas de Huang (Spearman ≈ 0 en general, 0.19 en GS) y el modelo de Huang califica a nuestras vacas ~0.8 puntos más altas, con pendiente 0.15. Las escalas de los evaluadores, el rango (3.25-4.25 contra 2.25-3.75) y el ángulo de cámara no son comparables, así que una métrica sobre Huang no diría nada de nuestro caso de uso.
3. Lo único con una señal parcial es que Huang-GS ordena a nuestras vacas (Spearman 0.60, AUC flacas 0.81). Hay algo de señal compartida de condición corporal, pero no lo bastante alineada para sumarla con pesos.
4. **Sin datos externos**, la calibración lineal post-hoc (e) corrige parte de la compresión: la pendiente pasa de 0.49-0.53 a 0.58-0.59 y el MAE en vacas ≥3.5 de 0.36 a 0.32-0.33. El MAE global no cambia de forma significativa (Δ -0.002 a -0.005, IC incluye 0). Es barata y opcional.
5. Decisión: queda **DINOv2-S + ridge/CORAL entrenado solo con CAETEC**, opcionalmente con calibración lineal. El error en los extremos se debe a que hubo una ligera falta de vacas de BCS ≤2.5 y ≥3.5.

Tiempos (CPU, flip TTA): DINOv2-S 0.095 s/img en CAETEC y 0.102 s/img en Huang. La CV de 32 configuraciones × 50 folds tardó 34 min.
