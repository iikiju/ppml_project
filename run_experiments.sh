#!/bin/bash

# Create directories for saving models and logs
mkdir -p models/cl
mkdir -p models/fl
mkdir -p logs/cl
mkdir -p logs/fl

# Default configuration variables
CL_CSV="data/meta_info_with_manufacturer_use_split_cl_new.csv"
FL_CSV="data/meta_info_with_manufacturer_use_federated_new.csv"
DEVICE="cuda"  # Change to cpu if needed
CL_EPOCHS=20
FL_ROUNDS=20  # 5에서 10으로 증가
SEED=42  # 랜덤 시드 설정
TARGET_EPSILON=50.0  # 목표 epsilon 값 (reduced from 100.0 to 10.0 for better privacy-utility balance)
DELTA=1e-1  # delta 값 (reduced from 0.5 to 1e-5 for stronger privacy guarantee)

# Memory management settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# Parse model type from command line
MODEL_TYPE=${2:-"resnet50"}  # Default to resnet50 if not specified
VALID_MODELS=("resnet50" "densenet121" "efficientnet_b0")

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

validate_model() {
    local valid=0
    for model in "${VALID_MODELS[@]}"; do
        if [ "$MODEL_TYPE" == "$model" ]; then
            valid=1
            break
        fi
    done
    
    if [ $valid -eq 0 ]; then
        echo "❌ Error: Invalid model type '$MODEL_TYPE'"
        echo "Valid models: resnet50, densenet121, efficientnet_b0"
        exit 1
    fi
    
    echo "🔍 Selected model: $MODEL_TYPE"
}

# Centralized Learning (CL) Experiments
run_cl() {
    local model_dir="models/cl/$MODEL_TYPE"
    local log_dir="logs/cl/$MODEL_TYPE"
    mkdir -p "$model_dir"
    mkdir -p "$log_dir"
    
    echo "=== 🔄 Running Centralized Learning (CL) Experiments with $MODEL_TYPE ==="
    
    # Create a master log file for this run
    local master_log="$log_dir/cl_${MODEL_TYPE}_${TIMESTAMP}_master.log"
    echo "=== Centralized Learning Experiments with $MODEL_TYPE ($(date)) ===" > "$master_log"
    echo "CSV: $CL_CSV" >> "$master_log"
    echo "Epochs: $CL_EPOCHS" >> "$master_log"
    echo "Device: $DEVICE" >> "$master_log"
    echo "Seed: $SEED" >> "$master_log"
    echo "" >> "$master_log"
    
    # Basic CL run
    echo "🏃‍♂️ Running basic CL with $MODEL_TYPE..."
    local basic_log="$log_dir/cl_${MODEL_TYPE}_basic_${TIMESTAMP}.log"
    { time python main_cl.py --csv $CL_CSV --model $MODEL_TYPE --epochs $CL_EPOCHS --device $DEVICE --seed $SEED --save_path "$model_dir/basic_cl.pth"; } 2>&1 | tee "$basic_log"
    echo "Basic CL completed. Log saved to: $basic_log"
    echo "Basic CL completed. See log: $basic_log" >> "$master_log"
    
    # CL with augmentation
    echo "🏃‍♂️ Running CL with augmentation ($MODEL_TYPE)..."
    local aug_log="$log_dir/cl_${MODEL_TYPE}_aug_${TIMESTAMP}.log"
    { time python main_cl.py --csv $CL_CSV --model $MODEL_TYPE --epochs $CL_EPOCHS --device $DEVICE --seed $SEED --augment --save_path "$model_dir/aug_cl.pth"; } 2>&1 | tee "$aug_log"
    echo "Augmented CL completed. Log saved to: $aug_log"
    echo "Augmented CL completed. See log: $aug_log" >> "$master_log"
    
    # Test saved models
    echo "🧪 Testing saved CL models ($MODEL_TYPE)..."
    local test_basic_log="$log_dir/cl_${MODEL_TYPE}_test_basic_${TIMESTAMP}.log"
    { python main_cl.py --csv $CL_CSV --model $MODEL_TYPE --device $DEVICE --seed $SEED --test_only --load_path "$model_dir/basic_cl.pth"; } 2>&1 | tee "$test_basic_log"
    echo "Basic model testing completed. Log saved to: $test_basic_log"
    echo "Basic model testing completed. See log: $test_basic_log" >> "$master_log"
    
    local test_aug_log="$log_dir/cl_${MODEL_TYPE}_test_aug_${TIMESTAMP}.log"
    { python main_cl.py --csv $CL_CSV --model $MODEL_TYPE --device $DEVICE --seed $SEED --test_only --load_path "$model_dir/aug_cl.pth"; } 2>&1 | tee "$test_aug_log"
    echo "Augmented model testing completed. Log saved to: $test_aug_log"
    echo "Augmented model testing completed. See log: $test_aug_log" >> "$master_log"
    
    echo "All CL experiments with $MODEL_TYPE completed. Master log: $master_log"
}

# Federated Learning (FL) Experiments
run_fl() {
    local model_dir="models/fl/$MODEL_TYPE"
    local log_dir="logs/fl/$MODEL_TYPE"
    mkdir -p "$model_dir"
    mkdir -p "$log_dir"
    
    echo "=== 🔄 Running Federated Learning (FL) Experiments with $MODEL_TYPE ==="
    
    # Create a master log file for this run
    local master_log="$log_dir/fl_${MODEL_TYPE}_${TIMESTAMP}_master.log"
    echo "=== Federated Learning Experiments with $MODEL_TYPE ($(date)) ===" > "$master_log"
    echo "CSV: $FL_CSV" >> "$master_log"
    echo "Rounds: $FL_ROUNDS" >> "$master_log"
    echo "Device: $DEVICE" >> "$master_log"
    echo "Seed: $SEED" >> "$master_log"
    echo "Target Epsilon: $TARGET_EPSILON" >> "$master_log"
    echo "Delta: $DELTA" >> "$master_log"
    echo "" >> "$master_log"
    
    # Basic FL run
    echo "🏃‍♂️ Running basic FL with $MODEL_TYPE..."
    local basic_log="$log_dir/fl_${MODEL_TYPE}_basic_${TIMESTAMP}.log"
    { time python main_fl.py --csv $FL_CSV --model $MODEL_TYPE --rounds $FL_ROUNDS --device $DEVICE --seed $SEED --save_path "$model_dir/basic_fl.pth"; } 2>&1 | tee "$basic_log"
    echo "Basic FL completed. Log saved to: $basic_log"
    echo "Basic FL completed. See log: $basic_log" >> "$master_log"
    
    # FL with augmentation
    echo "🏃‍♂️ Running FL with augmentation ($MODEL_TYPE)..."
    local aug_log="$log_dir/fl_${MODEL_TYPE}_aug_${TIMESTAMP}.log"
    { time python main_fl.py --csv $FL_CSV --model $MODEL_TYPE --rounds $FL_ROUNDS --device $DEVICE --seed $SEED --augment --save_path "$model_dir/aug_fl.pth"; } 2>&1 | tee "$aug_log"
    echo "Augmented FL completed. Log saved to: $aug_log"
    echo "Augmented FL completed. See log: $aug_log" >> "$master_log"
    
    # FL with DP - Calculate recommended noise for target epsilon
    echo "🔍 Calculating recommended noise multiplier for target epsilon: $TARGET_EPSILON..."
    # GPU 메모리 제약으로 배치 크기 8 유지, 다른 파라미터로 보상
    local batch_size=8  # GPU 메모리 제약으로 8 유지
    local estimated_dataset_size=1000  # 추정 데이터셋 크기
    local sample_rate=$(python -c "print($batch_size / $estimated_dataset_size)")
    # 더 많은 로컬 에폭으로 보상 (2 -> 5)
    local local_epochs=5
    local steps=$(echo "$FL_ROUNDS * $local_epochs / $sample_rate" | bc -l)
    local recommended_noise=$(python -c "from dp_utils import get_recommended_noise; print(get_recommended_noise($TARGET_EPSILON, $DELTA, $sample_rate, $steps))")
    echo "📊 Batch size: $batch_size, Sample rate: $sample_rate, Local epochs: $local_epochs, Steps: $steps"
    echo "📊 Recommended noise multiplier: $recommended_noise for target epsilon: $TARGET_EPSILON"
    echo "📊 Batch size: $batch_size, Sample rate: $sample_rate, Local epochs: $local_epochs, Steps: $steps" >> "$master_log"
    echo "📊 Recommended noise multiplier: $recommended_noise for target epsilon: $TARGET_EPSILON" >> "$master_log"
    
    # FL with DP using recommended noise (improved parameters for small batch)
    echo "🏃‍♂️ Running FL with differential privacy ($MODEL_TYPE) using recommended noise: $recommended_noise..."
    local dp_log="$log_dir/fl_${MODEL_TYPE}_dp_target_${TIMESTAMP}.log"
    { time python main_fl.py --csv $FL_CSV --model $MODEL_TYPE --rounds $FL_ROUNDS --device $DEVICE --seed $SEED --dp --noise "$recommended_noise" --target_epsilon $TARGET_EPSILON --delta $DELTA --local_epochs $local_epochs --lr 1e-3 --batch_size $batch_size --max_grad_norm 10.0 --save_path "$model_dir/dp_target_fl.pth" || echo "⚠️ DP experiment failed but continuing..."; } 2>&1 | tee "$dp_log"
    echo "DP FL (target epsilon) completed. Log saved to: $dp_log"
    echo "DP FL (target epsilon) completed. See log: $dp_log" >> "$master_log"
    
    # 모델 테스트
    if [ -f "$model_dir/dp_target_fl.pth" ]; then
        local test_log="$log_dir/fl_${MODEL_TYPE}_test_dp_target_${TIMESTAMP}.log"
        { python main_fl.py --csv $FL_CSV --model $MODEL_TYPE --device $DEVICE --seed $SEED --dp --test_only --load_path "$model_dir/dp_target_fl.pth" || echo "⚠️ DP 모델 테스트 중 오류가 발생했지만 계속합니다"; } 2>&1 | tee "$test_log"
        echo "DP Target model testing completed. Log saved to: $test_log"
        echo "DP Target model testing completed. See log: $test_log" >> "$master_log"
    else
        echo "⚠️ DP Target model not found, skipping test" | tee -a "$master_log"
    fi
    
    # FL with DP + augmentation using recommended noise
    echo "🏃‍♂️ Running FL with DP + augmentation ($MODEL_TYPE) using recommended noise: $recommended_noise..."
    local dp_aug_log="$log_dir/fl_${MODEL_TYPE}_dp_aug_target_${TIMESTAMP}.log"
    { time python main_fl.py --csv $FL_CSV --model $MODEL_TYPE --rounds $FL_ROUNDS --device $DEVICE --seed $SEED --dp --noise "$recommended_noise" --target_epsilon $TARGET_EPSILON --delta $DELTA --local_epochs $local_epochs --lr 1e-3 --batch_size $batch_size --max_grad_norm 10.0 --augment --save_path "$model_dir/dp_aug_target_fl.pth" || echo "⚠️ DP+Aug experiment failed but continuing..."; } 2>&1 | tee "$dp_aug_log"
    echo "DP + Augmented FL (target epsilon) completed. Log saved to: $dp_aug_log"
    echo "DP + Augmented FL (target epsilon) completed. See log: $dp_aug_log" >> "$master_log"
    
    # 기존 고정 노이즈 값 실험도 수행 (개선된 파라미터 사용)
    # FL with DP
    echo "🏃‍♂️ Running FL with differential privacy ($MODEL_TYPE) using fixed noise: 0.3..."
    local dp_fixed_log="$log_dir/fl_${MODEL_TYPE}_dp_fixed_${TIMESTAMP}.log"
    { time python main_fl.py --csv $FL_CSV --model $MODEL_TYPE --rounds $FL_ROUNDS --device $DEVICE --seed $SEED --dp --noise 0.3 --delta $DELTA --local_epochs $local_epochs --lr 1e-3 --batch_size $batch_size --max_grad_norm 10.0 --save_path "$model_dir/dp_fixed_fl.pth" || echo "⚠️ DP experiment failed but continuing..."; } 2>&1 | tee "$dp_fixed_log"
    echo "DP FL (fixed noise) completed. Log saved to: $dp_fixed_log"
    echo "DP FL (fixed noise) completed. See log: $dp_fixed_log" >> "$master_log"
    
    # FL with DP + augmentation
    echo "🏃‍♂️ Running FL with DP + augmentation ($MODEL_TYPE) using fixed noise: 0.3..."
    local dp_aug_fixed_log="$log_dir/fl_${MODEL_TYPE}_dp_aug_fixed_${TIMESTAMP}.log"
    { time python main_fl.py --csv $FL_CSV --model $MODEL_TYPE --rounds $FL_ROUNDS --device $DEVICE --seed $SEED --dp --noise 0.3 --delta $DELTA --local_epochs $local_epochs --lr 1e-3 --batch_size $batch_size --max_grad_norm 10.0 --augment --save_path "$model_dir/dp_aug_fixed_fl.pth" || echo "⚠️ DP+Aug experiment failed but continuing..."; } 2>&1 | tee "$dp_aug_fixed_log"
    echo "DP + Augmented FL (fixed noise) completed. Log saved to: $dp_aug_fixed_log"
    echo "DP + Augmented FL (fixed noise) completed. See log: $dp_aug_fixed_log" >> "$master_log"
    
    # Test saved models - only if they exist
    echo "🧪 Testing saved FL models ($MODEL_TYPE)..."
    
    # Test all saved models
    local models=("basic_fl.pth" "aug_fl.pth" "dp_target_fl.pth" "dp_aug_target_fl.pth" "dp_fixed_fl.pth" "dp_aug_fixed_fl.pth")
    local model_names=("Basic" "Augmented" "DP (Target ε)" "DP+Aug (Target ε)" "DP (Fixed)" "DP+Aug (Fixed)")
    
    # 결과 수집
    local results=()
    
    for i in "${!models[@]}"; do
        local model_file="${models[$i]}"
        local model_name="${model_names[$i]}"
        local test_log="$log_dir/fl_${MODEL_TYPE}_test_${model_file%.*}_${TIMESTAMP}.log"
        
        if [ -f "$model_dir/$model_file" ]; then
            { python main_fl.py --csv $FL_CSV --model $MODEL_TYPE --device $DEVICE --seed $SEED --test_only --load_path "$model_dir/$model_file"; } 2>&1 | tee "$test_log"
            echo "$model_name model testing completed. Log saved to: $test_log"
            echo "$model_name model testing completed. See log: $test_log" >> "$master_log"
            
            # 결과 추출 (정확도)
            local accuracy=$(grep "전체 정확도:" "$test_log" | tail -1 | awk '{print $3}')
            local epsilon=$(grep -o "ε=[0-9.]*" "$test_log" | tail -1 | cut -d= -f2)
            
            if [ -z "$epsilon" ]; then
                if [[ "$model_name" == *"DP"* ]]; then
                    # DP 모델이지만 epsilon 값이 없는 경우
                    epsilon="unknown"
                else
                    # DP가 아닌 모델
                    epsilon="N/A"
                fi
            fi
            
            results+=("$model_name: Accuracy $accuracy, ε=$epsilon")
        else
            echo "⚠️ $model_name model file not found, skipping test" | tee -a "$master_log"
            results+=("$model_name: Failed (no model)")
        fi
    done
    
    # 결과 요약
    echo "" >> "$master_log"
    echo "=== FL Experiments Results Summary ===" >> "$master_log"
    for result in "${results[@]}"; do
        echo "$result" >> "$master_log"
    done
    
    # 콘솔에도 출력
    echo ""
    echo "=== FL Experiments Results Summary ==="
    for result in "${results[@]}"; do
        echo "$result"
    done
    
    # 결과 비교 분석
    echo "📊 Generating epsilon-accuracy comparison analysis..."
    # 파이썬으로 결과 시각화 호출
    python -c "
import os
import sys
from dp_utils import compare_epsilon_values

try:
    # Extract accuracy and epsilon values safely
    results = {
        'Basic': {'accuracy': None, 'epsilon': None},
        'Augmented': {'accuracy': None, 'epsilon': None},
        'DP (Target ε)': {'accuracy': None, 'epsilon': None},
        'DP+Aug (Target ε)': {'accuracy': None, 'epsilon': None},
        'DP (Fixed)': {'accuracy': None, 'epsilon': None},
        'DP+Aug (Fixed)': {'accuracy': None, 'epsilon': None}
    }
    
    # Parse results from the command line
    for line in \"\"\"${results[@]}\"\"\".split('\\n'):
        if ':' not in line:
            continue
        
        for key in results.keys():
            if line.startswith(key + ':'):
                # Extract accuracy
                acc_match = line.split('Accuracy ')[1].split(',')[0] if 'Accuracy ' in line else None
                if acc_match and acc_match != 'None':
                    try:
                        results[key]['accuracy'] = float(acc_match)
                    except:
                        results[key]['accuracy'] = None
                
                # Extract epsilon
                eps_match = line.split('ε=')[1] if 'ε=' in line else None
                if eps_match and eps_match != 'N/A' and eps_match != 'unknown':
                    try:
                        results[key]['epsilon'] = float(eps_match)
                    except:
                        results[key]['epsilon'] = None
    
    # Call the comparison function
    compare_epsilon_values(results, '$MODEL_TYPE')
except Exception as e:
    print(f'Error during results comparison: {str(e)}')
    import traceback
    traceback.print_exc()
" 2>&1 | tee -a "$master_log"
    
    echo "All FL experiments with $MODEL_TYPE completed. Master log: $master_log"
}

# DP Noise Sweep Experiments
run_dp_noise_sweep() {
    local model_dir="models/fl/$MODEL_TYPE/dp_sweep"
    local log_dir="logs/fl/$MODEL_TYPE/dp_sweep"
    mkdir -p "$model_dir"
    mkdir -p "$log_dir"
    
    echo "=== 🔄 Running DP Noise Sweep Experiments with $MODEL_TYPE ==="
    
    # Create a master log file for this run
    local master_log="$log_dir/dp_sweep_${MODEL_TYPE}_${TIMESTAMP}_master.log"
    echo "=== DP Noise Sweep Experiments with $MODEL_TYPE ($(date)) ===" > "$master_log"
    echo "CSV: $FL_CSV" >> "$master_log"
    echo "Rounds: $FL_ROUNDS" >> "$master_log"
    echo "Device: $DEVICE" >> "$master_log"
    echo "Seed: $SEED" >> "$master_log"
    echo "Target Epsilon: $TARGET_EPSILON" >> "$master_log"
    echo "Delta: $DELTA" >> "$master_log"
    echo "" >> "$master_log"
    
    # 다양한 노이즈 값으로 실험 (reduced for memory efficiency)
    local noise_values=(0.0001 0.001 0.01 0.1 1 10 100)
    
    # 목표 epsilon에 맞는 노이즈 값도 추가 (실제 파라미터 사용)
    local batch_size=8  # GPU 메모리 제약
    local local_epochs=3  # 더 많은 로컬 에폭으로 보상
    local estimated_dataset_size=1500
    local sample_rate=$(python -c "print($batch_size / $estimated_dataset_size)")
    local steps=$(echo "$FL_ROUNDS * $local_epochs / $sample_rate" | bc -l)
    local recommended_noise=$(python -c "from dp_utils import get_recommended_noise; print(get_recommended_noise($TARGET_EPSILON, $DELTA, $sample_rate, $steps))")
    echo "📊 DP Sweep - Batch size: $batch_size, Sample rate: $sample_rate, Local epochs: $local_epochs"
    echo "📊 Recommended noise multiplier: $recommended_noise for target epsilon: $TARGET_EPSILON"
    echo "📊 DP Sweep - Batch size: $batch_size, Sample rate: $sample_rate, Local epochs: $local_epochs" >> "$master_log"
    echo "📊 Recommended noise multiplier: $recommended_noise for target epsilon: $TARGET_EPSILON" >> "$master_log"
    
    # 추천 노이즈 값이 이미 목록에 없는 경우에만 추가
    if ! [[ " ${noise_values[@]} " =~ " ${recommended_noise} " ]]; then
        noise_values+=($recommended_noise)
    fi
    
    local results=()
    
    for noise in "${noise_values[@]}"; do
        echo "🏃‍♂️ Running FL with DP, noise multiplier = $noise ($MODEL_TYPE)..."
        local dp_log="$log_dir/fl_${MODEL_TYPE}_dp_noise${noise}_${TIMESTAMP}.log"
        local model_path="$model_dir/dp_fl_noise${noise}.pth"
        
        # 실험 실행 (with memory cleanup and improved parameters)
        echo "🧹 Clearing GPU cache before experiment..."
        python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
        { time python main_fl.py --csv $FL_CSV --model $MODEL_TYPE --rounds $FL_ROUNDS --device $DEVICE --seed $SEED --dp --noise $noise --delta $DELTA --local_epochs $local_epochs --lr 1e-3 --batch_size $batch_size --max_grad_norm 10.0 --save_path "$model_path" || echo "⚠️ DP experiment with noise $noise failed but continuing..."; } 2>&1 | tee "$dp_log"
        echo "🧹 Clearing GPU cache after experiment..."
        python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
        echo "DP FL (noise=$noise) completed. Log saved to: $dp_log"
        echo "DP FL (noise=$noise) completed. See log: $dp_log" >> "$master_log"
        
        # 테스트 실행 (모델이 저장된 경우에만)
        if [ -f "$model_path" ]; then
            local test_log="$log_dir/fl_${MODEL_TYPE}_test_dp_noise${noise}_${TIMESTAMP}.log"
            { python main_fl.py --csv $FL_CSV --model $MODEL_TYPE --device $DEVICE --seed $SEED --test_only --dp --load_path "$model_path" || echo "⚠️ DP 모델 테스트 중 오류가 발생했지만 계속합니다"; } 2>&1 | tee "$test_log"
            echo "Model testing (noise=$noise) completed. Log saved to: $test_log"
            echo "Model testing (noise=$noise) completed. See log: $test_log" >> "$master_log"
            
            # 결과 추출 (정확도 및 epsilon)
            local accuracy=$(grep "전체 정확도:" "$test_log" | tail -1 | awk '{print $3}')
            local epsilon=$(grep -o "ε=[0-9.]*" "$test_log" | tail -1 | cut -d= -f2)
            
            if [ -z "$epsilon" ]; then
                epsilon="unknown"
            fi
            
            results+=("Noise $noise: Accuracy $accuracy, ε=$epsilon")
        else
            echo "⚠️ Model file for noise=$noise not found, skipping test" | tee -a "$master_log"
            results+=("Noise $noise: Failed (no model)")
        fi
    done
    
    # 결과 요약
    echo "" >> "$master_log"
    echo "=== DP Noise Sweep Results Summary ===" >> "$master_log"
    for result in "${results[@]}"; do
        echo "$result" >> "$master_log"
    done
    
    # 콘솔에도 출력
    echo ""
    echo "=== DP Noise Sweep Results Summary ==="
    for result in "${results[@]}"; do
        echo "$result"
    done
    
    # 결과 시각화
    echo "📊 Generating noise-epsilon-accuracy comparison analysis..."
    python -c "
import os
import sys
import re
from dp_utils import compare_epsilon_values

try:
    results = {}
    for r in \"\"\"${results[@]}\"\"\".split('Noise '):
        if not r.strip(): 
            continue
        
        m = re.match(r'([0-9.]+): Accuracy ([0-9.]+), ε=([0-9.]+|unknown|N/A)', r.strip())
        if m:
            noise, acc, eps = m.groups()
            name = f'Noise {noise}'
            
            try:
                acc_value = float(acc)
            except:
                acc_value = None
                
            if eps in ('unknown', 'N/A'):
                eps_value = None
            else:
                try:
                    eps_value = float(eps)
                except:
                    eps_value = None
                    
            try:
                noise_value = float(noise)
            except:
                noise_value = float(0)
                
            results[name] = {
                'accuracy': acc_value, 
                'epsilon': eps_value, 
                'noise': noise_value
            }
    
    # Only call if we have results
    if results:
        compare_epsilon_values(results, '$MODEL_TYPE', '$log_dir')
    else:
        print('No valid results found for comparison')
except Exception as e:
    print(f'Error during noise sweep results comparison: {str(e)}')
    import traceback
    traceback.print_exc()
" 2>&1 | tee -a "$master_log"
    
    echo "All DP noise sweep experiments with $MODEL_TYPE completed. Master log: $master_log"
}

# Run experiments for all model types
run_all_models() {
    local exp_type=$1
    for model in "${VALID_MODELS[@]}"; do
        MODEL_TYPE=$model
        case "$exp_type" in
            "cl")
                run_cl
                ;;
            "fl")
                run_fl
                ;;
            "dp-sweep")
                run_dp_noise_sweep
                ;;
            "all")
                run_cl
                run_fl
                ;;
        esac
    done
}

# Check if we should run for all models
if [ "$2" == "all_models" ]; then
    run_all_models "$1"
    echo "✅ All experiments with all models completed!"
    exit 0
fi

# Validate the model type
validate_model

# Select which experiments to run
case "$1" in
    "cl")
        run_cl
        ;;
    "fl")
        run_fl
        ;;
    "dp-sweep")
        run_dp_noise_sweep
        ;;
    "all")
        run_cl
        run_fl
        ;;
    *)
        echo "Usage: $0 {cl|fl|dp-sweep|all} [model_type|all_models]"
        echo "  First parameter:"
        echo "    cl        - Run only Centralized Learning experiments"
        echo "    fl        - Run only Federated Learning experiments"
        echo "    dp-sweep  - Run DP experiments with different noise values"
        echo "    all       - Run both CL and FL experiments"
        echo ""
        echo "  Second parameter (optional):"
        echo "    model_type - Specify a model: resnet50, densenet121, efficientnet_b0"
        echo "    all_models - Run experiments with all model types"
        echo ""
        echo "  Examples:"
        echo "    $0 cl resnet50     - Run CL experiments with ResNet50"
        echo "    $0 fl densenet121  - Run FL experiments with DenseNet121"
        echo "    $0 dp-sweep resnet50 - Run DP noise sweep with ResNet50"
        echo "    $0 all all_models  - Run all experiments with all model types"
        exit 1
        ;;
esac

echo "✅ All experiments with $MODEL_TYPE completed!"
echo "📊 Check logs directory for detailed output"