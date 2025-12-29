#!/bin/bash

# Configuration
TRUTH="scripts/v57/cibles.geojson"
RASTER="zone_19.tif"
PREDS_DIR="scripts/v60/preds"
VENV_PY="./venv/bin/python"
EVAL_SCRIPT="scripts/analysis/evaluate_holdout_zone.py"

echo "=========================================================="
echo "v60 Zone 19 Evaluation Results"
echo "=========================================================="

for t in 90 80 50 30 20 10; do
    echo -e "\n--- Threshold: $t% ---"
    PRED_FILE="$PREDS_DIR/preds_t$t.geojson"
    
    if [ -f "$PRED_FILE" ]; then
        echo "1m Buffer:"
        $VENV_PY $EVAL_SCRIPT "$TRUTH" "$PRED_FILE" "$RASTER" --buffer 1.0
        
        echo -e "\n3m Buffer:"
        $VENV_PY $EVAL_SCRIPT "$TRUTH" "$PRED_FILE" "$RASTER" --buffer 3.0
    else
        echo "Prediction file $PRED_FILE not found."
    fi
done
