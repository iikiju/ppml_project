import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def parse_time_to_minutes(time_str):
    """시간 문자열을 분 단위로 변환"""
    if time_str == "N/A" or pd.isna(time_str):
        return 0
    
    total_minutes = 0
    
    # 시간 파싱 (예: "1h 26m 47s" -> 86.78분)
    if 'h' in time_str:
        hours = int(time_str.split('h')[0])
        total_minutes += hours * 60
        time_str = time_str.split('h')[1].strip()
    
    if 'm' in time_str:
        minutes = int(time_str.split('m')[0])
        total_minutes += minutes
        time_str = time_str.split('m')[1].strip()
    
    if 's' in time_str and time_str.replace('s', '').strip():
        seconds = int(time_str.replace('s', ''))
        total_minutes += seconds / 60
    
    return total_minutes

def load_execution_times():
    """FL과 CL의 실행 시간 데이터를 로드"""
    # FL 데이터 로드
    fl_df = pd.read_csv('analysis_results/fl_execution_times.csv')
    
    # CL 데이터 로드
    cl_df = pd.read_csv('analysis_results/cl/cl_execution_times.csv')
    
    return fl_df, cl_df

def plot_fl_cl_execution_comparison():
    """FL과 CL의 실행 시간을 비교하는 통합 그래프"""
    fl_df, cl_df = load_execution_times()
    
    # 데이터 전처리
    models = ['resnet50', 'densenet121', 'efficientnet_b0']
    colors = {'resnet50': 'blue', 'densenet121': 'red', 'efficientnet_b0': 'green'}
    
    # 공통 실험 (basic, aug) 비교
    fig, axes = plt.subplots(2, 3, figsize=(26, 18))
    fig.suptitle('FL vs CL Execution Time Comparison', fontsize=36, fontweight='bold', y=0.98)
    
    # 1. Basic 실험 비교
    ax1 = axes[0, 0]
    fl_basic_times = []
    cl_basic_times = []
    model_names = []
    
    for model in models:
        fl_time = parse_time_to_minutes(fl_df[fl_df['Model'] == model]['basic'].iloc[0])
        cl_time = parse_time_to_minutes(cl_df[cl_df['Model'] == model]['basic'].iloc[0])
        
        fl_basic_times.append(fl_time)
        cl_basic_times.append(cl_time)
        model_names.append(model)
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, fl_basic_times, width, label='FL', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, cl_basic_times, width, label='CL', alpha=0.8, color='lightcoral')
    
    # 막대 위에 시간 표시
    for bar, time_min in zip(bars1, fl_basic_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
               f'{time_min:.0f}m', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    for bar, time_min in zip(bars2, cl_basic_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
               f'{time_min:.0f}m', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax1.set_title('Basic Experiment', fontsize=26, fontweight='bold', pad=15)
    ax1.set_xlabel('Model', fontsize=24, fontweight='bold')
    ax1.set_ylabel('Execution Time (minutes)', fontsize=24, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(fontsize=20)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    
    # 2. Aug 실험 비교
    ax2 = axes[0, 1]
    fl_aug_times = []
    cl_aug_times = []
    
    for model in models:
        fl_time = parse_time_to_minutes(fl_df[fl_df['Model'] == model]['aug'].iloc[0])
        cl_time = parse_time_to_minutes(cl_df[cl_df['Model'] == model]['aug'].iloc[0])
        
        fl_aug_times.append(fl_time)
        cl_aug_times.append(cl_time)
    
    bars1 = ax2.bar(x - width/2, fl_aug_times, width, label='FL', alpha=0.8, color='skyblue')
    bars2 = ax2.bar(x + width/2, cl_aug_times, width, label='CL', alpha=0.8, color='lightcoral')
    
    # 막대 위에 시간 표시
    for bar, time_min in zip(bars1, fl_aug_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{time_min:.0f}m', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    for bar, time_min in zip(bars2, cl_aug_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{time_min:.0f}m', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax2.set_title('Augmentation Experiment', fontsize=26, fontweight='bold', pad=15)
    ax2.set_xlabel('Model', fontsize=24, fontweight='bold')
    ax2.set_ylabel('Execution Time (minutes)', fontsize=24, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(fontsize=20)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    
    # 3. FL vs CL 속도 비교 (배수)
    ax3 = axes[0, 2]
    basic_ratio = [fl/cl for fl, cl in zip(fl_basic_times, cl_basic_times)]
    aug_ratio = [fl/cl for fl, cl in zip(fl_aug_times, cl_aug_times)]
    
    bars1 = ax3.bar(x - width/2, basic_ratio, width, label='Basic', alpha=0.8, color='gold')
    bars2 = ax3.bar(x + width/2, aug_ratio, width, label='Aug', alpha=0.8, color='orange')
    
    # 막대 위에 배수 표시
    for bar, ratio in zip(bars1, basic_ratio):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
               f'{ratio:.1f}x', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    for bar, ratio in zip(bars2, aug_ratio):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
               f'{ratio:.1f}x', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax3.set_title('FL/CL Speed Ratio', fontsize=26, fontweight='bold', pad=15)
    ax3.set_xlabel('Model', fontsize=24, fontweight='bold')
    ax3.set_ylabel('FL Time / CL Time (ratio)', fontsize=24, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend(fontsize=20)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=20)
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)  # 1배 기준선
    
    # 4-6. FL DP 실험들 (시간 단위로 표시)
    dp_experiments = ['dp_fixed', 'dp_target', 'dp_aug_fixed', 'dp_aug_target']
    dp_titles = ['DP Fixed', 'DP Target', 'DP Aug Fixed', 'DP Aug Target']
    
    for idx, (exp, title) in enumerate(zip(dp_experiments, dp_titles)):
        ax = axes[1, idx] if idx < 3 else axes[1, 2]  # 마지막 두 개는 겹침
        
        dp_times = []
        for model in models:
            time_min = parse_time_to_minutes(fl_df[fl_df['Model'] == model][exp].iloc[0])
            dp_times.append(time_min / 60)  # 시간 단위로 변환
        
        bars = ax.bar(models, dp_times, color=[colors[model] for model in models], alpha=0.7)
        
        # 막대 위에 시간 표시
        for bar, time_hour in zip(bars, dp_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_hour:.1f}h', ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        if idx < 3:
            ax.set_title(title, fontsize=26, fontweight='bold', pad=15)
            ax.set_xlabel('Model', fontsize=24, fontweight='bold')
            ax.set_ylabel('Execution Time (hours)', fontsize=24, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=20)
    
    # DP Aug Target은 세 번째 서브플롯에 오버레이
    if len(dp_experiments) == 4:
        ax = axes[1, 2]
        dp_times_target = []
        for model in models:
            time_min = parse_time_to_minutes(fl_df[fl_df['Model'] == model]['dp_aug_target'].iloc[0])
            dp_times_target.append(time_min / 60)
        
        # 두 번째 세트로 표시
        x_pos = np.arange(len(models))
        bars1 = ax.bar(x_pos - 0.2, [parse_time_to_minutes(fl_df[fl_df['Model'] == model]['dp_aug_fixed'].iloc[0])/60 for model in models], 
                      width=0.4, label='DP Aug Fixed', color='lightblue', alpha=0.7)
        bars2 = ax.bar(x_pos + 0.2, dp_times_target, width=0.4, label='DP Aug Target', color='lightgreen', alpha=0.7)
        
        ax.set_title('DP Augmentation Experiments', fontsize=26, fontweight='bold', pad=15)
        ax.set_xlabel('Model', fontsize=24, fontweight='bold')
        ax.set_ylabel('Execution Time (hours)', fontsize=24, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.legend(fontsize=18)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 상단 제목을 위한 여백 확보
    plt.subplots_adjust(hspace=0.35, wspace=0.25)  # 서브플롯 간격 조정
    save_path = 'analysis_results/fl_cl_execution_time_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"📊 FL vs CL 실행 시간 비교 그래프 저장: {save_path}")

def create_summary_table():
    """FL과 CL 실행 시간 요약 테이블 생성"""
    fl_df, cl_df = load_execution_times()
    
    summary_data = []
    models = ['resnet50', 'densenet121', 'efficientnet_b0']
    
    for model in models:
        fl_basic = parse_time_to_minutes(fl_df[fl_df['Model'] == model]['basic'].iloc[0])
        cl_basic = parse_time_to_minutes(cl_df[cl_df['Model'] == model]['basic'].iloc[0])
        fl_aug = parse_time_to_minutes(fl_df[fl_df['Model'] == model]['aug'].iloc[0])
        cl_aug = parse_time_to_minutes(cl_df[cl_df['Model'] == model]['aug'].iloc[0])
        
        summary_data.append({
            'Model': model,
            'FL_Basic_min': f"{fl_basic:.0f}",
            'CL_Basic_min': f"{cl_basic:.0f}",
            'Basic_Ratio': f"{fl_basic/cl_basic:.1f}x",
            'FL_Aug_min': f"{fl_aug:.0f}",
            'CL_Aug_min': f"{cl_aug:.0f}",
            'Aug_Ratio': f"{fl_aug/cl_aug:.1f}x"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = 'analysis_results/fl_cl_execution_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print("\n=== FL vs CL 실행 시간 비교 요약 ===")
    print(summary_df.to_string(index=False))
    print(f"\n📄 요약 테이블 저장: {summary_path}")

def main():
    """FL과 CL 실행 시간 비교 분석 실행"""
    print("=== FL vs CL 실행 시간 비교 분석 ===")
    
    # 통합 비교 그래프 생성
    plot_fl_cl_execution_comparison()
    
    # 요약 테이블 생성
    create_summary_table()
    
    print("\n=== 분석 완료 ===")

if __name__ == "__main__":
    main() 