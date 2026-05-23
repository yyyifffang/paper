#!/bin/bash

# 當任何一個指令發生錯誤 (Non-zero exit status) 時，立即停止整個腳本
set -e

# 定義要測試的隨機種子陣列 (至少 3 個以確保統計顯著性)
SEEDS=(42 43 44)
SCRIPT_NAME="ghost_tracking_BAAI.py"

echo "================================================="
echo "  Starting Automated Experiments (Ghost Tracking)  "
echo "================================================="

# 迴圈依序執行每個 seed
for SEED in "${SEEDS[@]}"; do
    START_TIME=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "\n[$START_TIME] >>> Starting run with SEED: $SEED <<<"
    
    # 執行 Python 腳本並傳入 seed 參數
    python3 $SCRIPT_NAME --seed $SEED
    
    END_TIME=$(date +'%Y-%m-%d %H:%M:%S')
    echo "[$END_TIME] >>> Finished run with SEED: $SEED <<<"
    
    # MLOps 最佳實踐：在兩次極限 GPU 運算之間強制暫停 10 秒
    # 讓作業系統有足夠的時間徹底回收 VRAM，避免下一輪啟動時發生 OOM
    echo "Cooling down for 10 seconds to flush VRAM..."
    sleep 10
done

echo "================================================="
echo "  All automated experiments completed successfully.  "
echo "================================================="