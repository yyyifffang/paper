#!/bin/bash

# 當任何一個指令發生錯誤 (Non-zero exit status) 時，立即停止整個腳本
set -e

# 定義要測試的隨機種子陣列 (至少 3 個以確保統計顯著性)
SEEDS=(42 43 44)
SCRIPT_NAME="ghost_tracking_BAAI.py"

N_ITERATIONS=40
BATCH_SIZE=40

echo "================================================="
echo "  Starting Automated Experiments (Ghost Tracking)  "
echo "================================================="

# 迴圈依序執行每個 seed
for SEED in "${SEEDS[@]}"; do
    START_TIME=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "\n[$START_TIME] >>> Starting run with SEED: $SEED, ITERATION: $N_ITERATIONS <<<"
    

   python3 "$SCRIPT_NAME" \
        --seed "$SEED" \
        --n-iterations "$N_ITERATIONS" \
        --batch-size "$BATCH_SIZE" \
        --llm-cost 0.03 \
        --verify-cost 0.03 \
        --synthetic-processing-cost 0.005
    
    END_TIME=$(date +'%Y-%m-%d %H:%M:%S')
    echo "[$END_TIME] >>> Finished run with SEED: $SEED <<<"
    
    echo "Cooling down for 10 seconds to flush VRAM..."
    sleep 60
done

echo "================================================="
echo "  All automated experiments completed successfully.  "
echo "================================================="