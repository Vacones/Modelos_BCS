#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
PY=.venv-mac/bin/python
test -x "$PY" || { echo 'Primero ejecuta bash instalar_mac.sh'; exit 2; }
# CA de certifi para urllib/torch.hub; mantiene la verificación HTTPS activa.
export SSL_CERT_FILE="$("$PY" -m certifi)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
DATASET="${1:-$PWD/dataset_yolo}"
DETECTOR="${2:-}"
CSV_V4="${3:-}"
MODE="${ROI_MODE:-upper}"
EXPLAIN_LIMIT="${EXPLAIN_LIMIT:-0}"
ROI_ARGS=(--roi-mode "$MODE")
if [ -n "${ROI_CSV:-}" ]; then ROI_ARGS+=(--roi-csv "$ROI_CSV"); fi
test -d "$DATASET" || { echo 'No existe el dataset: indica una ruta externa autorizada; consulta README.md.'; exit 2; }
mkdir -p runs
if [ -n "${RUN_DESTINO:-}" ]; then
  "$PY" preflight.py --datos "$DATASET" --salida "$RUN_DESTINO"
  mkdir "$RUN_DESTINO"
  RUN_ROOT="$(cd "$RUN_DESTINO" && pwd)"
else
  RUN_ROOT="$(mktemp -d "$PWD/runs/mac_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
fi
printf '%s\n' "$RUN_ROOT" > ULTIMA_EJECUCION.txt
exec > >(tee -a "$RUN_ROOT/ejecucion.log") 2>&1
trap 'echo "Falló una etapa; conserva esta carpeta y consulta $RUN_ROOT/ejecucion.log. La guía explica cómo continuar."' ERR
"$PY" comprobar_gpu_mac.py --dino
"$PY" pipeline.py etl --data "$DATASET" --work "$RUN_ROOT/datos"
"$PY" pipeline.py eda --work "$RUN_ROOT/datos"
"$PY" revisar_regiones.py --work "$RUN_ROOT/datos" --out "$RUN_ROOT/rois_gt_aproximadas.csv"
"$PY" pipeline.py features --work "$RUN_ROOT/datos" --out "$RUN_ROOT/features_gt" --splits train val --views 1 --batch 4 --device mps "${ROI_ARGS[@]}"
"$PY" pipeline.py train --features "$RUN_ROOT/features_gt" --run "$RUN_ROOT/modelo_ldl_tf" --epochs 100 --patience 15 --device mps
"$PY" pipeline.py evaluate --work "$RUN_ROOT/datos" --features "$RUN_ROOT/features_gt" --model "$RUN_ROOT/modelo_ldl_tf" --out "$RUN_ROOT/val_gt" --device mps
if [ -z "$DETECTOR" ]; then
  echo 'No se proporcionó best.pt: se entrena un detector nuevo en MPS. Esta corrida no aísla el cambio del clasificador respecto al ordinal.'
  "$PY" pipeline.py detector-train --work "$RUN_ROOT/datos" --run "$RUN_ROOT/detector_nuevo" --weights yolo26n.pt --epochs 80 --patience 0 --batch 4 --device mps
  DETECTOR="$RUN_ROOT/detector_nuevo/weights/best.pt"
fi
test -f "$DETECTOR"
if [ -n "$CSV_V4" ]; then
  "$PY" reutilizar_recortes.py --work "$RUN_ROOT/datos" --csv "$CSV_V4" --out "$RUN_ROOT/boxes_val"
else
  "$PY" pipeline.py detector-crops --work "$RUN_ROOT/datos" --weights "$DETECTOR" --out "$RUN_ROOT/boxes_val" --splits val --device mps
fi
"$PY" revisar_regiones.py --work "$RUN_ROOT/datos" --boxes "$RUN_ROOT/boxes_val" --splits val --out "$RUN_ROOT/rois_pred_aproximadas.csv"
"$PY" pipeline.py features --work "$RUN_ROOT/datos" --out "$RUN_ROOT/features_pred" --boxes "$RUN_ROOT/boxes_val" --splits val --batch 4 --device mps "${ROI_ARGS[@]}"
"$PY" pipeline.py evaluate --work "$RUN_ROOT/datos" --features "$RUN_ROOT/features_pred" --model "$RUN_ROOT/modelo_ldl_tf" --out "$RUN_ROOT/val_pipeline" --device mps
"$PY" pipeline.py explain --work "$RUN_ROOT/datos" --features "$RUN_ROOT/features_pred" --boxes "$RUN_ROOT/boxes_val" --model "$RUN_ROOT/modelo_ldl_tf" --out "$RUN_ROOT/explicaciones_val" --limit "$EXPLAIN_LIMIT" --grid 7 --batch 4 --device mps
for STAGE in gt pipeline; do
  if [ "$STAGE" = gt ]; then BASELINE="${BASELINE_GT_CSV:-}"; else BASELINE="${BASELINE_PIPELINE_CSV:-}"; fi
  if [ -z "$BASELINE" ]; then
    echo "Comparación $STAGE omitida: proporciona BASELINE_GT_CSV o BASELINE_PIPELINE_CSV con datos privados externos."
    continue
  fi
  if "$PY" comparar.py --ordinal "$BASELINE" --dinov2 "$RUN_ROOT/val_${STAGE}/predicciones.csv" --work "$RUN_ROOT/datos" --out "$RUN_ROOT/comparacion_${STAGE}"; then
    echo "Comparación $STAGE generada. Revisar si detector/recortes son idénticos."
  else
    echo "Comparación $STAGE no disponible: comprueba coincidencia de fotos/etiquetas. Métricas y mapas guardados."
  fi
done
printf '%s\n' 'Finalizado. No se evaluó test. Revisar val_pipeline/resumen.md y explicaciones_val/.' > "$RUN_ROOT/FINALIZADO.txt"
echo "Resultados: $RUN_ROOT"
