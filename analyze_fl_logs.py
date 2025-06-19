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

LOG_ROOT = 'logs/fl'
RESULTS_DIR = 'analysis_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 실험 조건별 파일명 패턴
EXPERIMENT_PATTERNS = [
    ('basic', 'basic'),
    ('aug', 'aug'),
    ('dp_fixed', 'dp_fixed'),
    ('dp_target', 'dp_target'),
    ('dp_aug_fixed', 'dp_aug_fixed'),
    ('dp_aug_target', 'dp_aug_target'),
]

# efficientnet_b0 제외
MODEL_TYPES = ['resnet50', 'densenet121', 'efficientnet_b0']

# 로그에서 라운드별 epsilon, loss 추출
ROUND_EPSILON_RE = re.compile(r'🔒 라운드 (\d+)의 글로벌 ε 추정값: ([0-9.]+)')
EPOCH_LOSS_RE = re.compile(r'📈 에폭 (\d+)/(\d+), 손실: ([0-9.]+)')

# master log에서 test accuracy, epsilon 추출
SUMMARY_RE = re.compile(r'^(.*): Accuracy ([0-9.]+), ε=([0-9.A-Za-z]+)', re.MULTILINE)

# 실험 조건별 로그 파일명 패턴
LOG_SUFFIXES = {
    'basic': 'basic',
    'aug': 'aug',
    'dp_fixed': 'dp_fixed',
    'dp_target': 'dp_target',
    'dp_aug_fixed': 'dp_aug_fixed',
    'dp_aug_target': 'dp_aug_target',
}

COLOR_PALETTE = sns.color_palette('tab10', n_colors=10)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def find_log_files(model_dir):
    files = os.listdir(model_dir)
    logs = {}
    for key, suffix in LOG_SUFFIXES.items():
        # DP 실험은 dp_*, 나머지는 *_fl
        if 'dp' in key:
            pattern = f'fl_*_{key}_*.log'
            log = [f for f in files if pattern in f and not f.startswith('test_')]
        else:
            pattern = f'fl_*_{suffix}_*.log'
            log = [f for f in files if pattern in f and not f.startswith('test_')]
        if log:
            logs[key] = log[0]
    return logs

def parse_round_metrics(log_path):
    epsilons = []
    losses = []
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    round_eps = None
    round_loss = None
    for i, line in enumerate(lines):
        m = ROUND_EPSILON_RE.search(line)
        if m:
            round_eps = float(m.group(2))
            epsilons.append(round_eps)
        m2 = EPOCH_LOSS_RE.search(line)
        if m2 and m2.group(1) == m2.group(2):  # 마지막 에폭
            round_loss = float(m2.group(3))
            losses.append(round_loss)
    return epsilons, losses

def parse_test_accuracy(master_log_path):
    with open(master_log_path, 'r', encoding='utf-8') as f:
        text = f.read()
    results = {}
    for m in SUMMARY_RE.finditer(text):
        name, acc, eps = m.groups()
        results[name.strip()] = {'accuracy': float(acc), 'epsilon': eps}
    return results

def plot_metric_across_rounds(metric_dict, ylabel, title, save_path):
    plt.figure(figsize=(10,6))
    for i, (cond, values) in enumerate(metric_dict.items()):
        plt.plot(range(1, len(values)+1), values, label=cond, color=COLOR_PALETTE[i])
    plt.xlabel('Round')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_test_accuracy_bar(acc_dict, title, save_path):
    plt.figure(figsize=(8,5))
    names = list(acc_dict.keys())
    accs = [acc_dict[n]['accuracy'] for n in names]
    plt.bar(names, accs, color=COLOR_PALETTE[:len(names)])
    plt.ylabel('Test Accuracy')
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tradeoff(tradeoff_list, title, save_path):
    plt.figure(figsize=(8,6))
    for i, (cond, eps, acc) in enumerate(tradeoff_list):
        if eps is not None:
            plt.scatter(eps, acc, label=cond, s=100, color=COLOR_PALETTE[i])
            plt.text(eps, acc, cond, fontsize=10, ha='right')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Test Accuracy')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_privacy_logs(model_dir, experiment_type):
    """모델 디렉토리에서 privacy 로그 파일들을 로드"""
    privacy_files = glob.glob(os.path.join(model_dir, f"*{experiment_type}*privacy*.json"))
    if not privacy_files:
        return None
    
    # 가장 최근 파일 선택
    latest_file = max(privacy_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        return json.load(f)

def parse_log_file(log_path):
    """로그 파일에서 loss와 accuracy 값을 파싱"""
    train_losses = []  # 각 라운드의 전체 평균 loss
    test_losses = []
    train_accs = []
    test_accs = []
    
    current_round = 0
    current_client_losses = []  # 현재 라운드의 각 클라이언트별 loss
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 라운드 시작 확인
            if '=== 라운드' in line:
                if current_client_losses:
                    # 이전 라운드의 평균 loss 계산
                    round_avg_loss = sum(current_client_losses) / len(current_client_losses)
                    train_losses.append(round_avg_loss)
                    current_client_losses = []
                current_round += 1
            
            # Loss 값 파싱 (에폭별 loss)
            if '📈 에폭' in line and '손실:' in line:
                try:
                    loss = float(re.search(r'손실: ([\d.]+)', line).group(1))
                    current_client_losses.append(loss)
                except:
                    continue
    
    # 마지막 라운드의 평균 loss 계산
    if current_client_losses:
        round_avg_loss = sum(current_client_losses) / len(current_client_losses)
        train_losses.append(round_avg_loss)
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accs,
        'test_accuracies': test_accs
    }

def plot_epsilon_values_comparison(model_name, save_dir):
    """Fixed와 Target의 Global Epsilon 값들을 비교 시각화"""
    model_dir = os.path.join('models', 'fl', model_name)
    
    # Fixed와 Target privacy 데이터 로드
    fixed_data = load_privacy_logs(model_dir, 'dp_fixed')
    target_data = load_privacy_logs(model_dir, 'dp_target')
    
    if not fixed_data and not target_data:
        print(f"[{model_name}] Privacy 데이터를 찾을 수 없습니다.")
        return

    plt.figure(figsize=(16, 8))
    
    # Fixed Global epsilon
    if fixed_data:
        rounds_fixed = [r['round'] for r in fixed_data['rounds']]
        global_epsilons_fixed = [r['global_epsilon'] for r in fixed_data['rounds']]
        plt.plot(rounds_fixed, global_epsilons_fixed, 'b-', 
                label='Fixed DP', linewidth=2.5, marker='o', markersize=6)
    
    # Target Global epsilon  
    if target_data:
        rounds_target = [r['round'] for r in target_data['rounds']]
        global_epsilons_target = [r['global_epsilon'] for r in target_data['rounds']]
        plt.plot(rounds_target, global_epsilons_target, 'r-', 
                label='Target DP', linewidth=2.5, marker='s', markersize=6)

    plt.xlabel('Round', fontsize=24, fontweight='bold')
    plt.ylabel('Global Epsilon Value', fontsize=24, fontweight='bold')
    plt.title(f'{model_name} - Global Epsilon Comparison (Fixed vs Target)', 
             fontsize=28, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=24, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # X축을 5단위로 설정
    if fixed_data or target_data:
        max_rounds = max(
            (max(rounds_fixed) if fixed_data else 0),
            (max(rounds_target) if target_data else 0)
        )
        plt.xticks(range(0, max_rounds + 1, 5), fontsize=20)
    
    # Y축 눈금 크기 설정
    plt.yticks(fontsize=20)
    
    # Save with adjusted layout
    save_path = os.path.join(save_dir, f'{model_name}_global_epsilon_comparison.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[{model_name}] Global Epsilon 비교 그래프 저장: {save_path}")

def plot_loss_values(log_dir, model_name, save_dir):
    """Loss 값들을 시각화"""
    log_files = glob.glob(os.path.join(log_dir, f"fl_{model_name}_*.log"))
    # test_ 파일 제외
    log_files = [f for f in log_files if not os.path.basename(f).startswith('test_')]
    
    if not log_files:
        print(f"[{model_name}] Loss 데이터를 찾을 수 없습니다.")
        return

    # 가장 최근 파일 선택
    latest_file = max(log_files, key=os.path.getctime)
    print(f"[{model_name}] 분석 중인 파일: {os.path.basename(latest_file)}")
    data = parse_log_file(latest_file)
    
    if not data['train_losses']:
        print(f"[{model_name}] Loss 데이터를 찾을 수 없습니다.")
        return

    rounds = list(range(1, len(data['train_losses']) + 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, data['train_losses'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Round')
    plt.ylabel('Loss Value')
    plt.title(f'{model_name} - Loss Values Over Rounds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # X축을 5단위로 설정
    plt.xticks(range(0, len(data['train_losses']) + 1, 5))
    
    save_path = os.path.join(save_dir, f'{model_name}_loss.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[{model_name}] Loss 그래프 저장: {save_path}")

def plot_accuracy_values(log_dir, model_name, save_dir):
    """Accuracy 값들을 시각화"""
    log_files = glob.glob(os.path.join(log_dir, f"fl_{model_name}_*.log"))
    # test_ 파일 제외
    log_files = [f for f in log_files if not os.path.basename(f).startswith('test_')]
    
    if not log_files:
        print(f"[{model_name}] Accuracy 데이터를 찾을 수 없습니다.")
        return

    # 가장 최근 파일 선택
    latest_file = max(log_files, key=os.path.getctime)
    data = parse_log_file(latest_file)
    
    if not data['train_accuracies']:
        print(f"[{model_name}] Accuracy 데이터를 찾을 수 없습니다.")
        return

    rounds = list(range(1, len(data['train_accuracies']) + 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, data['train_accuracies'], 'b-', label='Train Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Accuracy Values Over Rounds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    save_path = os.path.join(save_dir, f'{model_name}_accuracy.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_experiment_metrics(log_dir, model_name, save_dir):
    """각 실험별 loss와 accuracy를 시각화"""
    # 실험 유형별 로그 파일 찾기
    non_dp_experiments = {
        'basic': f'fl_{model_name}_basic_*.log',
        'aug': f'fl_{model_name}_aug_*.log'
    }
    
    dp_experiments = {
        'dp_fixed': f'fl_{model_name}_dp_fixed_*.log',
        'dp_aug_fixed': f'fl_{model_name}_dp_aug_fixed_*.log',
        'dp_target': f'fl_{model_name}_dp_target_*.log',
        'dp_aug_target': f'fl_{model_name}_dp_aug_target_*.log'
    }
    
    # Non-DP 실험 Loss 그래프
    plt.figure(figsize=(12, 6))
    found_data = False
    
    for exp_type, pattern in non_dp_experiments.items():
        log_files = glob.glob(os.path.join(log_dir, pattern))
        # test_ 파일 제외
        log_files = [f for f in log_files if not os.path.basename(f).startswith('test_')]
        
        if not log_files:
            continue
            
        latest_file = max(log_files, key=os.path.getctime)
        print(f"[{model_name}] {exp_type} 분석 중: {os.path.basename(latest_file)}")
        data = parse_log_file(latest_file)
        
        if data['train_losses']:
            rounds = list(range(1, len(data['train_losses']) + 1))
            plt.plot(rounds, data['train_losses'], label=f'{exp_type}', linewidth=2, marker='o', markersize=4)
            found_data = True
    
    if found_data:
        plt.xlabel('Round')
        plt.ylabel('Loss Value')
        plt.title(f'{model_name} - Loss Values (Non-DP Experiments)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        # X축을 5단위로 설정
        max_rounds = max([len(parse_log_file(max(glob.glob(os.path.join(log_dir, pattern)), key=os.path.getctime))['train_losses']) 
                         for pattern in non_dp_experiments.values() 
                         if glob.glob(os.path.join(log_dir, pattern))])
        plt.xticks(range(0, max_rounds + 1, 5))
        
        save_path = os.path.join(save_dir, f'{model_name}_non_dp_loss.png')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[{model_name}] Non-DP Loss 그래프 저장: {save_path}")
    else:
        print(f"[{model_name}] Non-DP Loss 데이터를 찾을 수 없습니다.")
    
    plt.close()
    
    # DP 실험 Loss 그래프
    plt.figure(figsize=(12, 6))
    found_data = False
    
    for exp_type, pattern in dp_experiments.items():
        log_files = glob.glob(os.path.join(log_dir, pattern))
        # test_ 파일 제외
        log_files = [f for f in log_files if not os.path.basename(f).startswith('test_')]
        
        if not log_files:
            continue
            
        latest_file = max(log_files, key=os.path.getctime)
        print(f"[{model_name}] {exp_type} 분석 중: {os.path.basename(latest_file)}")
        data = parse_log_file(latest_file)
        
        if data['train_losses']:
            rounds = list(range(1, len(data['train_losses']) + 1))
            plt.plot(rounds, data['train_losses'], label=f'{exp_type}', linewidth=2, marker='s', markersize=4)
            found_data = True
    
    if found_data:
        plt.xlabel('Round')
        plt.ylabel('Loss Value')
        plt.title(f'{model_name} - Loss Values (DP Experiments)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        # X축을 5단위로 설정
        max_rounds = max([len(parse_log_file(max(glob.glob(os.path.join(log_dir, pattern)), key=os.path.getctime))['train_losses']) 
                         for pattern in dp_experiments.values() 
                         if glob.glob(os.path.join(log_dir, pattern))])
        plt.xticks(range(0, max_rounds + 1, 5))
        
        save_path = os.path.join(save_dir, f'{model_name}_dp_loss.png')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[{model_name}] DP Loss 그래프 저장: {save_path}")
    else:
        print(f"[{model_name}] DP Loss 데이터를 찾을 수 없습니다.")
    
    plt.close()

def analyze_model(model_name):
    """특정 모델의 로그를 분석하고 그래프 생성"""
    model_dir = os.path.join('models', 'fl', model_name)
    log_dir = os.path.join('logs', 'fl', model_name)
    save_dir = os.path.join('analysis_results', model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Global Epsilon 비교 시각화 (Fixed vs Target)
    plot_epsilon_values_comparison(model_name, save_dir)
    
    # 실험별 loss와 accuracy 시각화
    plot_experiment_metrics(log_dir, model_name, save_dir)
    
    # Loss 데이터 분석 및 시각화
    plot_loss_values(log_dir, model_name, save_dir)
    
    # Accuracy 데이터 분석 및 시각화
    plot_accuracy_values(log_dir, model_name, save_dir)
    
    print(f"[{model_name}] 분석 및 그래프 저장 완료!")

def plot_fl_non_dp_train_loss_comparison():
    """FL에서 DP를 적용하지 않은 모든 모델의 train loss 비교"""
    plt.figure(figsize=(18, 10))
    
    # Non-DP 실험들 (basic, aug)
    non_dp_experiments = ['basic', 'aug']
    
    # 색상과 마커 설정
    colors = {'resnet50': 'blue', 'densenet121': 'red', 'efficientnet_b0': 'green'}
    markers = {'basic': 'o', 'aug': 's'}
    line_styles = {'basic': '-', 'aug': '--'}
    
    found_data = False
    
    for model_name in MODEL_TYPES:
        log_dir = os.path.join('logs', 'fl', model_name)
        
        for exp_type in non_dp_experiments:
            if exp_type == 'basic':
                pattern = f'fl_{model_name}_basic_*.log'
            else:  # aug
                pattern = f'fl_{model_name}_aug_*.log'
                
            log_files = glob.glob(os.path.join(log_dir, pattern))
            # test_ 파일 제외
            log_files = [f for f in log_files if not os.path.basename(f).startswith('test_')]
            
            if not log_files:
                print(f"[{model_name}] {exp_type} 로그 파일을 찾을 수 없습니다.")
                continue
                
            latest_file = max(log_files, key=os.path.getctime)
            print(f"[{model_name}] {exp_type} 분석 중: {os.path.basename(latest_file)}")
            data = parse_log_file(latest_file)
            
            if data['train_losses']:
                rounds = list(range(1, len(data['train_losses']) + 1))
                
                # 라벨 생성
                label = f'{model_name} ({exp_type})'
                
                plt.plot(rounds, data['train_losses'], 
                        label=label,
                        color=colors[model_name], 
                        linestyle=line_styles[exp_type],
                        marker=markers[exp_type], 
                        linewidth=2.5, 
                        markersize=5,
                        alpha=0.8)
                found_data = True
    
    if found_data:
        plt.xlabel('Round', fontsize=24, fontweight='bold')
        plt.ylabel('Train Loss', fontsize=24, fontweight='bold')
        plt.title('FL Non-DP Train Loss Comparison\n(All Models: Basic vs Aug)', 
                 fontsize=28, fontweight='bold', pad=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 범례를 그래프 바깥으로 구성
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, frameon=True, 
                  fancybox=True, shadow=True, fontsize=24)
        
        # X축을 5단위로 설정
        if rounds:
            max_rounds = max([len(parse_log_file(max(glob.glob(os.path.join('logs', 'fl', model, f'fl_{model}_basic_*.log')), key=os.path.getctime))['train_losses']) 
                             for model in MODEL_TYPES 
                             if glob.glob(os.path.join('logs', 'fl', model, f'fl_{model}_basic_*.log'))])
            plt.xticks(range(0, max_rounds + 1, 5), fontsize=20)
        
        # Y축 눈금 크기 설정
        plt.yticks(fontsize=20)
        
        # Y축 범위 조정
        plt.ylim(bottom=0)
        
        save_path = os.path.join('analysis_results', 'fl_non_dp_train_loss_comparison.png')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"FL Non-DP Train Loss 비교 그래프 저장: {save_path}")
        
        # 추가로 각 실험별로 분리된 그래프도 생성
        plot_fl_non_dp_by_experiment()
        
    else:
        print("FL Non-DP Train Loss 데이터를 찾을 수 없습니다.")
        plt.close()

def plot_fl_non_dp_by_experiment():
    """FL Non-DP 실험을 basic와 aug로 분리해서 그래프 그리기"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('FL Non-DP Train Loss by Experiment Type', fontsize=32, fontweight='bold')
    
    experiments = ['basic', 'aug']
    colors = {'resnet50': 'blue', 'densenet121': 'red', 'efficientnet_b0': 'green'}
    
    for idx, exp_type in enumerate(experiments):
        ax = axes[idx]
        found_data = False
        
        for model_name in MODEL_TYPES:
            log_dir = os.path.join('logs', 'fl', model_name)
            
            if exp_type == 'basic':
                pattern = f'fl_{model_name}_basic_*.log'
            else:  # aug
                pattern = f'fl_{model_name}_aug_*.log'
                
            log_files = glob.glob(os.path.join(log_dir, pattern))
            # test_ 파일 제외
            log_files = [f for f in log_files if not os.path.basename(f).startswith('test_')]
            
            if not log_files:
                continue
                
            latest_file = max(log_files, key=os.path.getctime)
            data = parse_log_file(latest_file)
            
            if data['train_losses']:
                rounds = list(range(1, len(data['train_losses']) + 1))
                
                ax.plot(rounds, data['train_losses'], 
                       label=model_name,
                       color=colors[model_name], 
                       linewidth=3, 
                       marker='o',
                       markersize=5,
                       alpha=0.8)
                found_data = True
        
        if found_data:
            ax.set_xlabel('Round', fontsize=24, fontweight='bold')
            ax.set_ylabel('Train Loss', fontsize=24, fontweight='bold')
            ax.set_title(f'{exp_type.capitalize()} Experiment', fontsize=26, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
            
            # X축을 5단위로 설정
            if rounds:
                ax.set_xticks(range(0, len(rounds) + 1, 5))
            
            # 축 눈금 크기 설정
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            # Y축 범위 조정
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    save_path = os.path.join('analysis_results', 'fl_non_dp_by_experiment.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"FL Non-DP 실험별 Train Loss 그래프 저장: {save_path}")

def plot_fl_dp_train_loss_comparison():
    """FL에서 DP를 적용한 모든 모델의 train loss 비교"""
    plt.figure(figsize=(20, 12))
    
    # DP 실험들
    dp_experiments = ['dp_fixed', 'dp_target', 'dp_aug_fixed', 'dp_aug_target']
    
    # 색상과 마커 설정
    colors = {'resnet50': 'blue', 'densenet121': 'red', 'efficientnet_b0': 'green'}
    markers = {'dp_fixed': 'o', 'dp_target': 's', 'dp_aug_fixed': '^', 'dp_aug_target': 'D'}
    line_styles = {'dp_fixed': '-', 'dp_target': '--', 'dp_aug_fixed': '-.', 'dp_aug_target': ':'}
    
    found_data = False
    
    for model_name in MODEL_TYPES:
        log_dir = os.path.join('logs', 'fl', model_name)
        
        for exp_type in dp_experiments:
            pattern = f'fl_{model_name}_{exp_type}_*.log'
            log_files = glob.glob(os.path.join(log_dir, pattern))
            # test_ 파일 제외
            log_files = [f for f in log_files if not os.path.basename(f).startswith('test_')]
            
            if not log_files:
                print(f"[{model_name}] {exp_type} 로그 파일을 찾을 수 없습니다.")
                continue
                
            latest_file = max(log_files, key=os.path.getctime)
            print(f"[{model_name}] {exp_type} 분석 중: {os.path.basename(latest_file)}")
            data = parse_log_file(latest_file)
            
            if data['train_losses']:
                rounds = list(range(1, len(data['train_losses']) + 1))
                
                # 라벨 생성
                label = f'{model_name} ({exp_type})'
                
                plt.plot(rounds, data['train_losses'], 
                        label=label,
                        color=colors[model_name], 
                        linestyle=line_styles[exp_type],
                        marker=markers[exp_type], 
                        linewidth=2.5, 
                        markersize=5,
                        alpha=0.8)
                found_data = True
    
    if found_data:
        plt.xlabel('Round', fontsize=24, fontweight='bold')
        plt.ylabel('Train Loss', fontsize=24, fontweight='bold')
        plt.title('FL DP Train Loss Comparison\n(All Models: Fixed, Target, Aug+Fixed, Aug+Target)', 
                 fontsize=28, fontweight='bold', pad=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 범례를 그래프 바깥으로 구성
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, frameon=True, 
                  fancybox=True, shadow=True, fontsize=20)
        
        # 축 눈금 크기 설정
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        # Y축 범위 조정
        plt.ylim(bottom=0)
        
        save_path = os.path.join('analysis_results', 'fl_dp_train_loss_comparison.png')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"FL DP Train Loss 비교 그래프 저장: {save_path}")
        
    else:
        print("FL DP Train Loss 데이터를 찾을 수 없습니다.")
        plt.close()

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

def analyze_fl_execution_times():
    """FL 실험별 실행 시간 분석"""
    all_experiments = ['basic', 'aug', 'dp_fixed', 'dp_target', 'dp_aug_fixed', 'dp_aug_target']
    
    execution_times = {}
    
    print("=== FL 실험 실행 시간 분석 ===")
    print()
    
    for model_name in MODEL_TYPES:
        log_dir = os.path.join('logs', 'fl', model_name)
        execution_times[model_name] = {}
        
        print(f"📊 {model_name.upper()} 모델:")
        
        for exp_type in all_experiments:
            pattern = f'fl_{model_name}_{exp_type}_*.log'
            log_files = glob.glob(os.path.join(log_dir, pattern))
            # test_ 파일 제외
            log_files = [f for f in log_files if not os.path.basename(f).startswith('test_')]
            
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
    create_execution_time_table(execution_times)
    
    # 실행 시간 시각화
    plot_execution_times(execution_times)
    
    return execution_times

def create_execution_time_table(execution_times):
    """실행 시간 테이블을 CSV로 저장"""
    all_experiments = ['basic', 'aug', 'dp_fixed', 'dp_target', 'dp_aug_fixed', 'dp_aug_target']
    
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
    csv_path = os.path.join('analysis_results', 'fl_execution_times.csv')
    df.to_csv(csv_path, index=False)
    print(f"📄 실행 시간 테이블 저장: {csv_path}")
    
    # 콘솔에 테이블 출력
    print("\n=== FL 실험 실행 시간 요약 ===")
    print(df.to_string(index=False))
    print()

def plot_execution_times(execution_times):
    """실행 시간을 막대 그래프로 시각화"""
    all_experiments = ['basic', 'aug', 'dp_fixed', 'dp_target', 'dp_aug_fixed', 'dp_aug_target']
    
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
        print("시각화할 실행 시간 데이터가 없습니다.")
        return
    
    # 그래프 생성
    plt.figure(figsize=(20, 10))
    
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
    
    # 서브플롯 배치
    n_experiments = len(exp_groups)
    n_cols = 3
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    if n_rows == 1:
        axes = [axes] if n_experiments == 1 else axes
    else:
        axes = axes.flatten()
    
    fig.suptitle('FL Experiment Execution Times by Model', fontsize=32, fontweight='bold')
    
    for idx, (exp, data) in enumerate(exp_groups.items()):
        ax = axes[idx] if n_experiments > 1 else axes[0]
        
        bars = ax.bar(data['models'], data['times'], color=data['colors'], alpha=0.7)
        
        # 막대 위에 시간 표시
        for bar, time_min in zip(bars, data['times']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_min:.1f}m', ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        ax.set_title(f'{exp}', fontsize=24, fontweight='bold')
        ax.set_xlabel('Model', fontsize=20, fontweight='bold')
        ax.set_ylabel('Execution Time (minutes)', fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # 빈 서브플롯 숨기기
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join('analysis_results', 'fl_execution_times.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"📊 실행 시간 그래프 저장: {save_path}")

def main():
    """모든 모델에 대해 분석 수행"""
    # 분석 결과를 저장할 디렉토리 생성
    os.makedirs('analysis_results', exist_ok=True)
    
    # FL 실행 시간 분석
    print("=== FL 실행 시간 분석 시작 ===")
    analyze_fl_execution_times()
    
    # FL Non-DP Train Loss 비교 그래프 생성
    print("\n=== FL Non-DP Train Loss 비교 분석 시작 ===")
    plot_fl_non_dp_train_loss_comparison()
    
    # FL DP Train Loss 비교 그래프 생성
    print("\n=== FL DP Train Loss 비교 분석 시작 ===")
    plot_fl_dp_train_loss_comparison()
    
    # 각 모델에 대해 분석 수행
    print("\n=== 각 모델별 분석 시작 ===")
    models = ['resnet50', 'densenet121', 'efficientnet_b0']
    for model in models:
        print(f"\n--- {model} 분석 중 ---")
        analyze_model(model)

if __name__ == "__main__":
    main() 