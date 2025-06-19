import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import glob
from matplotlib import font_manager, rc
import datetime

LOG_ROOT = 'logs/cl'
RESULTS_DIR = 'analysis_results/cl'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 모델 타입
MODEL_TYPES = ['resnet50', 'densenet121', 'efficientnet_b0']

# 실험 조건
EXPERIMENT_TYPES = ['basic', 'aug']

# 로그에서 epoch별 loss 추출 패턴
TRAIN_LOSS_RE = re.compile(r'Epoch (\d+)/(\d+), Train Loss: ([0-9.]+)')
VAL_LOSS_RE = re.compile(r'Epoch (\d+), Val Loss: ([0-9.]+), Acc: ([0-9.]+), F1: ([0-9.]+)')
TEST_RESULT_RE = re.compile(r'📊 테스트 세트 성능: 손실=([0-9.]+), 정확도=([0-9.]+), F1=([0-9.]+)')

COLOR_PALETTE = sns.color_palette('tab10', n_colors=10)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def parse_cl_log(log_path):
    """Centralized learning 로그 파일을 파싱하여 epoch별 loss와 accuracy 추출"""
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    test_result = None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Train loss 추출
    for match in TRAIN_LOSS_RE.finditer(content):
        epoch = int(match.group(1))
        total_epochs = int(match.group(2))
        train_loss = float(match.group(3))
        train_losses.append((epoch, train_loss))
    
    # Validation loss, accuracy, F1 추출
    for match in VAL_LOSS_RE.finditer(content):
        epoch = int(match.group(1))
        val_loss = float(match.group(2))
        val_acc = float(match.group(3))
        val_f1 = float(match.group(4))
        val_losses.append((epoch, val_loss))
        val_accuracies.append((epoch, val_acc))
        val_f1_scores.append((epoch, val_f1))
    
    # Test result 추출
    test_match = TEST_RESULT_RE.search(content)
    if test_match:
        test_loss = float(test_match.group(1))
        test_acc = float(test_match.group(2))
        test_f1 = float(test_match.group(3))
        test_result = {'loss': test_loss, 'accuracy': test_acc, 'f1': test_f1}
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores,
        'test_result': test_result
    }

def plot_loss_curves(model_name, save_dir):
    """모델별 loss curve 시각화"""
    model_dir = os.path.join(LOG_ROOT, model_name)
    
    # 실험별 로그 파일 찾기
    experiments_data = {}
    
    for exp_type in EXPERIMENT_TYPES:
        log_pattern = f'cl_{model_name}_{exp_type}_*.log'
        log_files = glob.glob(os.path.join(model_dir, log_pattern))
        
        if not log_files:
            print(f"[{model_name}] {exp_type} 로그 파일을 찾을 수 없습니다.")
            continue
        
        # 가장 최근 파일 선택
        latest_file = max(log_files, key=os.path.getctime)
        print(f"[{model_name}] {exp_type} 분석 중: {os.path.basename(latest_file)}")
        
        data = parse_cl_log(latest_file)
        if data['train_losses']:
            experiments_data[exp_type] = data

    if not experiments_data:
        print(f"[{model_name}] 분석할 로그 데이터가 없습니다.")
        return

    # Loss curve 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Training Progress', fontsize=32, fontweight='bold')
    
    # Train Loss 비교
    ax1 = axes[0, 0]
    for i, (exp_type, data) in enumerate(experiments_data.items()):
        epochs = [x[0] for x in data['train_losses']]
        losses = [x[1] for x in data['train_losses']]
        ax1.plot(epochs, losses, label=f'{exp_type}', linewidth=2, 
                marker='o', markersize=4, color=COLOR_PALETTE[i])
    
    ax1.set_xlabel('Epoch', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Train Loss', fontsize=20, fontweight='bold')
    ax1.set_title('Training Loss', fontsize=22, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Validation Loss 비교
    ax2 = axes[0, 1]
    for i, (exp_type, data) in enumerate(experiments_data.items()):
        epochs = [x[0] for x in data['val_losses']]
        losses = [x[1] for x in data['val_losses']]
        ax2.plot(epochs, losses, label=f'{exp_type}', linewidth=2, 
                marker='s', markersize=4, color=COLOR_PALETTE[i])
    
    ax2.set_xlabel('Epoch', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=20, fontweight='bold')
    ax2.set_title('Validation Loss', fontsize=22, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    # Validation Accuracy 비교
    ax3 = axes[1, 0]
    for i, (exp_type, data) in enumerate(experiments_data.items()):
        epochs = [x[0] for x in data['val_accuracies']]
        accuracies = [x[1] for x in data['val_accuracies']]
        ax3.plot(epochs, accuracies, label=f'{exp_type}', linewidth=2, 
                marker='^', markersize=4, color=COLOR_PALETTE[i])
    
    ax3.set_xlabel('Epoch', fontsize=20, fontweight='bold')
    ax3.set_ylabel('Validation Accuracy', fontsize=20, fontweight='bold')
    ax3.set_title('Validation Accuracy', fontsize=22, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(fontsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    
    # Validation F1 Score 비교
    ax4 = axes[1, 1]
    for i, (exp_type, data) in enumerate(experiments_data.items()):
        epochs = [x[0] for x in data['val_f1_scores']]
        f1_scores = [x[1] for x in data['val_f1_scores']]
        ax4.plot(epochs, f1_scores, label=f'{exp_type}', linewidth=2, 
                marker='D', markersize=4, color=COLOR_PALETTE[i])
    
    ax4.set_xlabel('Epoch', fontsize=20, fontweight='bold')
    ax4.set_ylabel('Validation F1 Score', fontsize=20, fontweight='bold')
    ax4.set_title('Validation F1 Score', fontsize=22, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(fontsize=16)
    ax4.tick_params(axis='both', which='major', labelsize=16)
    
    # 저장
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[{model_name}] 훈련 곡선 그래프 저장: {save_path}")

def plot_cl_train_loss_comparison():
    """CL에서 모든 모델의 train loss 비교"""
    plt.figure(figsize=(14, 8))
    
    # 실험들 (basic, aug)
    experiments = ['basic', 'aug']
    
    # 색상과 마커 설정
    colors = {'resnet50': 'blue', 'densenet121': 'red', 'efficientnet_b0': 'green'}
    markers = {'basic': 'o', 'aug': 's'}
    line_styles = {'basic': '-', 'aug': '--'}
    
    found_data = False
    
    for model_name in MODEL_TYPES:
        model_dir = os.path.join(LOG_ROOT, model_name)
        
        for exp_type in experiments:
            log_pattern = f'cl_{model_name}_{exp_type}_*.log'
            log_files = glob.glob(os.path.join(model_dir, log_pattern))
            
            if not log_files:
                print(f"[{model_name}] {exp_type} 로그 파일을 찾을 수 없습니다.")
                continue
                
            latest_file = max(log_files, key=os.path.getctime)
            print(f"[{model_name}] {exp_type} 분석 중: {os.path.basename(latest_file)}")
            data = parse_cl_log(latest_file)
            
            if data['train_losses']:
                epochs = [x[0] for x in data['train_losses']]
                losses = [x[1] for x in data['train_losses']]
                
                # 라벨 생성
                label = f'{model_name} ({exp_type})'
                
                plt.plot(epochs, losses, 
                        label=label,
                        color=colors[model_name], 
                        linestyle=line_styles[exp_type],
                        marker=markers[exp_type], 
                        linewidth=2.5, 
                        markersize=5,
                        alpha=0.8)
                found_data = True
    
    if found_data:
        plt.xlabel('Epoch', fontsize=24, fontweight='bold')
        plt.ylabel('Train Loss', fontsize=24, fontweight='bold')
        plt.title('CL Train Loss Comparison\n(All Models: Basic vs Aug)', 
                 fontsize=28, fontweight='bold', pad=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 범례를 두 개 컬럼으로 구성
        plt.legend(loc='upper right', ncol=2, frameon=True, 
                  fancybox=True, shadow=True, fontsize=24)
        
        # 축 눈금 크기 설정
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        # Y축 범위 조정
        plt.ylim(bottom=0)
        
        save_path = os.path.join(RESULTS_DIR, 'cl_train_loss_comparison.png')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"CL Train Loss 비교 그래프 저장: {save_path}")
        
        # 추가로 각 실험별로 분리된 그래프도 생성
        plot_cl_by_experiment()
        
    else:
        print("CL Train Loss 데이터를 찾을 수 없습니다.")
        plt.close()

def plot_cl_by_experiment():
    """CL 실험을 basic와 aug로 분리해서 그래프 그리기"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('CL Train Loss by Experiment Type', fontsize=32, fontweight='bold')
    
    experiments = ['basic', 'aug']
    colors = {'resnet50': 'blue', 'densenet121': 'red', 'efficientnet_b0': 'green'}
    
    for idx, exp_type in enumerate(experiments):
        ax = axes[idx]
        found_data = False
        
        for model_name in MODEL_TYPES:
            model_dir = os.path.join(LOG_ROOT, model_name)
            
            log_pattern = f'cl_{model_name}_{exp_type}_*.log'
            log_files = glob.glob(os.path.join(model_dir, log_pattern))
            
            if not log_files:
                continue
                
            latest_file = max(log_files, key=os.path.getctime)
            data = parse_cl_log(latest_file)
            
            if data['train_losses']:
                epochs = [x[0] for x in data['train_losses']]
                losses = [x[1] for x in data['train_losses']]
                
                ax.plot(epochs, losses, 
                       label=model_name,
                       color=colors[model_name], 
                       linewidth=3, 
                       marker='o',
                       markersize=5,
                       alpha=0.8)
                found_data = True
        
        if found_data:
            ax.set_xlabel('Epoch', fontsize=24, fontweight='bold')
            ax.set_ylabel('Train Loss', fontsize=24, fontweight='bold')
            ax.set_title(f'{exp_type.capitalize()} Experiment', fontsize=26, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right', fontsize=20)
            
            # 축 눈금 크기 설정
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            # Y축 범위 조정
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'cl_by_experiment.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"CL 실험별 Train Loss 그래프 저장: {save_path}")

def plot_test_results_comparison(save_dir):
    """모든 모델의 테스트 결과 비교"""
    test_results = {}
    
    for model_name in MODEL_TYPES:
        model_dir = os.path.join(LOG_ROOT, model_name)
        test_results[model_name] = {}
        
        for exp_type in EXPERIMENT_TYPES:
            log_pattern = f'cl_{model_name}_{exp_type}_*.log'
            log_files = glob.glob(os.path.join(model_dir, log_pattern))
            
            if not log_files:
                continue
                
            latest_file = max(log_files, key=os.path.getctime)
            data = parse_cl_log(latest_file)
            
            if data['test_result']:
                test_results[model_name][exp_type] = data['test_result']

    # Test Accuracy 비교 막대 그래프
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Centralized Learning - Test Results Comparison', fontsize=32, fontweight='bold')
    
    metrics = ['accuracy', 'loss', 'f1']
    metric_names = ['Test Accuracy', 'Test Loss', 'Test F1 Score']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        # 모델별, 실험별 결과 정리
        model_names = []
        basic_values = []
        aug_values = []
        
        for model_name in MODEL_TYPES:
            if model_name in test_results:
                model_names.append(model_name)
                basic_val = test_results[model_name].get('basic', {}).get(metric, 0)
                aug_val = test_results[model_name].get('aug', {}).get(metric, 0)
                basic_values.append(basic_val)
                aug_values.append(aug_val)
        
        if model_names:
            x = np.arange(len(model_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, basic_values, width, label='Basic', 
                          color=COLOR_PALETTE[0], alpha=0.8)
            bars2 = ax.bar(x + width/2, aug_values, width, label='Aug', 
                          color=COLOR_PALETTE[1], alpha=0.8)
            
            ax.set_xlabel('Model', fontsize=24, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=24, fontweight='bold')
            ax.set_title(metric_name, fontsize=26, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45)
            ax.legend(fontsize=20)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 축 눈금 크기 설정
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            # 값 표시
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=18)
            
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=18)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_results_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"테스트 결과 비교 그래프 저장: {save_path}")

def parse_execution_time(log_path):
    """로그 파일에서 실행 시간을 파싱 (마지막 부분의 real 시간)"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 파일 끝부분에서 'real' 키워드가 있는 줄 찾기
        for line in reversed(lines[-50:]):  # 마지막 50줄에서 찾기
            if 'real' in line.lower():
                # real    5m32.123s 또는 real	1h5m32s 형태
                time_match = re.search(r'real\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?', line.lower())
                if time_match:
                    hours = int(time_match.group(1) or 0)
                    minutes = int(time_match.group(2) or 0)
                    seconds = float(time_match.group(3) or 0)
                    
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    return total_seconds
        
        return None
    except Exception as e:
        print(f"시간 파싱 오류 ({log_path}): {e}")
        return None

def format_time(seconds):
    """초를 시:분:초 형태로 포맷"""
    if seconds is None:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def analyze_cl_execution_times():
    """CL 실험별 실행 시간 분석"""
    all_experiments = ['basic', 'aug']
    
    execution_times = {}
    
    print("=== CL 실험 실행 시간 분석 ===")
    print()
    
    for model_name in MODEL_TYPES:
        model_dir = os.path.join(LOG_ROOT, model_name)
        execution_times[model_name] = {}
        
        print(f"📊 {model_name.upper()} 모델:")
        
        for exp_type in all_experiments:
            log_pattern = f'cl_{model_name}_{exp_type}_*.log'
            log_files = glob.glob(os.path.join(model_dir, log_pattern))
            
            if not log_files:
                execution_times[model_name][exp_type] = None
                print(f"  ❌ {exp_type}: 로그 파일 없음")
                continue
                
            latest_file = max(log_files, key=os.path.getctime)
            exec_time = parse_execution_time(latest_file)
            execution_times[model_name][exp_type] = exec_time
            
            formatted_time = format_time(exec_time)
            print(f"  ⏱️  {exp_type}: {formatted_time}")
        
        print()
    
    # 실행 시간 요약 테이블 생성
    create_cl_execution_time_table(execution_times)
    
    # 실행 시간 시각화
    plot_cl_execution_times(execution_times)
    
    return execution_times

def create_cl_execution_time_table(execution_times):
    """CL 실행 시간 테이블을 CSV로 저장"""
    all_experiments = ['basic', 'aug']
    
    # DataFrame 생성
    data = []
    for model in MODEL_TYPES:
        row = {'Model': model}
        for exp in all_experiments:
            time_sec = execution_times[model].get(exp)
            row[exp] = format_time(time_sec)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # CSV 저장
    csv_path = os.path.join(RESULTS_DIR, 'cl_execution_times.csv')
    df.to_csv(csv_path, index=False)
    print(f"📄 CL 실행 시간 테이블 저장: {csv_path}")
    
    # 콘솔에 테이블 출력
    print("\n=== CL 실험 실행 시간 요약 ===")
    print(df.to_string(index=False))
    print()

def plot_cl_execution_times(execution_times):
    """CL 실행 시간을 막대 그래프로 시각화"""
    all_experiments = ['basic', 'aug']
    
    # 데이터 준비
    models = []
    experiments = []
    times_minutes = []
    
    for model in MODEL_TYPES:
        for exp in all_experiments:
            time_sec = execution_times[model].get(exp)
            if time_sec is not None:
                models.append(model)
                experiments.append(exp)
                times_minutes.append(time_sec / 60)  # 분 단위로 변환
    
    if not times_minutes:
        print("시각화할 CL 실행 시간 데이터가 없습니다.")
        return
    
    # 모델별 색상 설정
    colors = {'resnet50': 'blue', 'densenet121': 'red', 'efficientnet_b0': 'green'}
    
    # 실험별로 그룹화하여 막대 그래프 그리기
    exp_groups = {}
    for i, exp in enumerate(experiments):
        if exp not in exp_groups:
            exp_groups[exp] = {'models': [], 'times': [], 'colors': []}
        exp_groups[exp]['models'].append(models[i])
        exp_groups[exp]['times'].append(times_minutes[i])
        exp_groups[exp]['colors'].append(colors[models[i]])
    
    # 그래프 생성
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('CL Experiment Execution Times by Model', fontsize=32, fontweight='bold')
    
    for idx, (exp, data) in enumerate(exp_groups.items()):
        ax = axes[idx]
        
        bars = ax.bar(data['models'], data['times'], color=data['colors'], alpha=0.7)
        
        # 막대 위에 시간 표시
        for bar, time_min in zip(bars, data['times']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_min:.1f}m', ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        ax.set_title(f'{exp.capitalize()}', fontsize=26, fontweight='bold')
        ax.set_xlabel('Model', fontsize=24, fontweight='bold')
        ax.set_ylabel('Execution Time (minutes)', fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'cl_execution_times.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"📊 CL 실행 시간 그래프 저장: {save_path}")

def analyze_model(model_name):
    """특정 모델의 centralized learning 로그 분석"""
    save_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss curve 분석 및 시각화
    plot_loss_curves(model_name, save_dir)
    
    print(f"[{model_name}] Centralized Learning 분석 완료!")

def main():
    """모든 모델에 대해 centralized learning 분석 수행"""
    # 분석 결과를 저장할 디렉토리 생성
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # CL 실행 시간 분석
    print("=== CL 실행 시간 분석 시작 ===")
    analyze_cl_execution_times()
    
    # CL Train Loss 비교 그래프 생성
    print("\n=== CL Train Loss 비교 분석 시작 ===")
    plot_cl_train_loss_comparison()
    
    # 각 모델에 대해 분석 수행
    print("\n=== 각 모델별 분석 시작 ===")
    for model in MODEL_TYPES:
        print(f"\n--- {model} 분석 중 ---")
        analyze_model(model)
    
    # 전체 모델 테스트 결과 비교
    print("\n=== 테스트 결과 비교 분석 시작 ===")
    plot_test_results_comparison(RESULTS_DIR)
    
    print("\n=== Centralized Learning 분석 완료 ===")

if __name__ == "__main__":
    main() 