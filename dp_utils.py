from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import torch.optim as optim
import numpy as np
import warnings
import math
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from opacus.utils.batch_memory_manager import BatchMemoryManager

# Epsilon 범위 설정 (추가)
EPSILON_RANGES = {
    'very_strong': (0.0, 1.0),    # 매우 강한 프라이버시 보호
    'strong': (1.0, 3.0),         # 강한 프라이버시 보호
    'moderate': (3.0, 8.0),       # 중간 수준의 프라이버시 보호
    'weak': (8.0, 16.0),          # 약한 프라이버시 보호
    'very_weak': (16.0, float('inf'))  # 매우 약한 프라이버시 보호
}

def get_privacy_level(epsilon):
    """
    입력된 epsilon 값에 따라 프라이버시 보호 수준을 반환합니다.
    
    Args:
        epsilon (float): 계산된 epsilon 값
    
    Returns:
        str: 프라이버시 보호 수준 ('very_strong', 'strong', 'moderate', 'weak', 'very_weak')
    """
    if epsilon is None:
        return "unknown"
        
    for level, (min_val, max_val) in EPSILON_RANGES.items():
        if min_val <= epsilon < max_val:
            return level
    
    return "very_weak"  # 기본값

def get_recommended_noise(target_epsilon, delta=1e-5, sample_rate=0.04, steps=50):
    """
    목표 epsilon 값을 달성하기 위한 노이즈 멀티플라이어 값을 추정합니다.
    이진 검색을 사용하여 노이즈 값을 찾습니다.
    
    Args:
        target_epsilon (float): 목표 epsilon 값
        delta (float): 목표 delta 값
        sample_rate (float): 샘플링 비율 (배치 크기 / 데이터셋 크기)
        steps (int): 훈련 스텝 수
    
    Returns:
        float: 추정된 노이즈 멀티플라이어 값
    """
    try:
        # 노이즈 멀티플라이어의 탐색 범위
        min_noise = 0.1
        max_noise = 30.0
        tolerance = 0.01  # 허용 오차
        
        # 이진 검색
        while max_noise - min_noise > tolerance:
            mid_noise = (min_noise + max_noise) / 2
            estimated_epsilon = compute_rdp_to_dp(mid_noise, sample_rate, steps, delta)
            
            if estimated_epsilon is None:
                # 계산 실패 시
                max_noise = mid_noise
                continue
                
            if abs(estimated_epsilon - target_epsilon) < tolerance:
                # 충분히 가까운 값을 찾은 경우
                return mid_noise
            elif estimated_epsilon > target_epsilon:
                # 현재 노이즈가 너무 작음 (epsilon이 큼)
                min_noise = mid_noise
            else:
                # 현재 노이즈가 너무 큼 (epsilon이 작음)
                max_noise = mid_noise
        
        # 최종 노이즈 값 반환
        return (min_noise + max_noise) / 2
    except Exception as e:
        print(f"⚠️ 노이즈 추천 계산 오류: {e}")
        # 실패 시 합리적인 기본값 제공
        return 1.0

def fix_model_for_dp(model):
    """
    모델을 DP 적용 가능하도록 수정합니다.
    BatchNorm 등의 레이어를 GroupNorm으로 대체합니다.
    """
    # 모델 구조 변경 전 원본 state_dict 저장
    original_state_dict = model.state_dict()
    original_state_dict_keys = set(original_state_dict.keys())
    
    # 모델 수정 가능성 확인
    is_valid = ModuleValidator.is_valid(model)
    
    if not is_valid:
        print("⚠️ 모델에 DP 호환되지 않는 레이어가 있습니다.")
        try:
            # 엄격하지 않은 모드로 변환 시도
            model = ModuleValidator.fix(model, strict=False)
            print("경고: 모델이 아직 완전히 호환되지 않을 수 있으나, 비엄격 모드로 진행합니다.")
            
            # 변환 후 state_dict 키 변경 확인
            new_state_dict = model.state_dict()
            new_state_dict_keys = set(new_state_dict.keys())
            added_keys = new_state_dict_keys - original_state_dict_keys
            removed_keys = original_state_dict_keys - new_state_dict_keys
            
            if added_keys:
                print(f"🔄 DP 변환으로 추가된 키: {len(added_keys)}개")
            if removed_keys:
                print(f"🔄 DP 변환으로 제거된 키: {len(removed_keys)}개")
            
            # 공통 키에 대해 원본 가중치 복원 시도
            common_keys = original_state_dict_keys & new_state_dict_keys
            if common_keys:
                print(f"🔄 공통 키 {len(common_keys)}개에 대해 가중치 복원 중...")
                for key in common_keys:
                    try:
                        # 텐서 크기와 타입이 호환되는지 확인
                        original_tensor = original_state_dict[key]
                        new_tensor = new_state_dict[key]
                        
                        if original_tensor.shape == new_tensor.shape:
                            # 데이터 타입 호환성 확인 후 복사
                            if original_tensor.dtype == new_tensor.dtype:
                                new_state_dict[key].copy_(original_tensor)
                            else:
                                # 타입이 다른 경우 안전하게 변환
                                new_state_dict[key].copy_(original_tensor.to(new_tensor.dtype))
                        else:
                            print(f"⚠️ 키 {key}의 텐서 크기가 변경됨: {original_tensor.shape} -> {new_tensor.shape}")
                    except Exception as e:
                        print(f"⚠️ 키 {key} 복원 실패: {e}")
                        continue
                
                # 복원된 state_dict로 모델 업데이트
                try:
                    model.load_state_dict(new_state_dict, strict=False)
                except Exception as e:
                    print(f"⚠️ 복원된 가중치 로딩 실패: {e}")
            
        except Exception as e:
            print(f"❌ 모델 변환 실패: {e}")
            return None
    
    return model

def attach_dp(model, optimizer, data_loader, noise_multiplier=1.0, max_grad_norm=1.0, device='cuda'):
    """
    모델, 옵티마이저, 데이터 로더에 DP를 적용합니다.
    """
    # DP 적용 전 모델 상태 기록
    pre_dp_state_dict_keys = set(model.state_dict().keys())
    
    # 모델이 올바른 디바이스에 있는지 확인
    model = model.to(device)
    
    # 시드 고정을 위한 secure_mode 끄기
    try:
        privacy_engine = PrivacyEngine(secure_mode=False)
        
        # DP 적용
        try:
            # 배치 메모리 관리자 비활성화로 디바이스 충돌 방지
            # poisson_sampling=False로 설정하여 더 안정적인 샘플링 사용
            model, optimizer, data_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                poisson_sampling=False,  # 포아송 샘플링 비활성화
            )
            
            # DP 적용 후 모델을 다시 올바른 디바이스로 이동
            model = model.to(device)
            
            # 옵티마이저의 상태도 올바른 디바이스로 이동
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        try:
                            state[k] = v.to(device)
                        except Exception as e:
                            print(f"⚠️ 옵티마이저 상태 {k} 디바이스 이동 실패: {e}")
                            continue
            
            # 모든 모델 파라미터가 올바른 디바이스에 있는지 확인
            for param in model.parameters():
                try:
                    if param.device != torch.device(device):
                        param.data = param.data.to(device)
                        if param.grad is not None:
                            param.grad = param.grad.to(device)
                except Exception as e:
                    print(f"⚠️ 모델 파라미터 디바이스 이동 실패: {e}")
                    continue
            
            # DP 적용 후 모델 상태 확인
            post_dp_state_dict_keys = set(model.state_dict().keys())
            added_keys = post_dp_state_dict_keys - pre_dp_state_dict_keys
            removed_keys = pre_dp_state_dict_keys - post_dp_state_dict_keys
            
            if added_keys or removed_keys:
                print(f"🔄 DP 엔진 적용으로 모델 상태 변경: +{len(added_keys)}개, -{len(removed_keys)}개 키")
            
            return model, optimizer, data_loader, privacy_engine
            
        except Exception as e:
            print(f"⚠️ DP 엔진 적용 실패: {e}")
            print("⚠️ 기본 DP 설정으로 재시도합니다...")
            
            # 기본 설정으로 재시도
            try:
                model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=data_loader,
                    target_epsilon=10.0,  # 완화된 epsilon 목표
                    target_delta=1e-5,
                    epochs=1,
                    max_grad_norm=max_grad_norm,
                    poisson_sampling=False,  # 포아송 샘플링 비활성화
                )
                
                # 재시도 후에도 디바이스 확인
                model = model.to(device)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
                
                # 모든 모델 파라미터가 올바른 디바이스에 있는지 확인
                for param in model.parameters():
                    if param.device != torch.device(device):
                        param.data = param.data.to(device)
                        if param.grad is not None:
                            param.grad = param.grad.to(device)
                
                return model, optimizer, data_loader, privacy_engine
            except Exception as e2:
                print(f"❌ 기본 DP 설정으로도 실패: {e2}")
                raise e2
    
    except Exception as e:
        print(f"❌ DP 적용 실패: {e}")
        raise e

def calculate_epsilon(privacy_engine, sample_rate, epochs, delta=1e-5):
    """
    DP에서의 epsilon 값을 계산합니다.
    """
    if privacy_engine is None:
        print("⚠️ Privacy Engine이 없습니다.")
        return None
    
    try:
        # 방법 1: 내장 accountant 사용
        try:
            epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
            print(f"🔒 계산된 ε (accountant 방식): {epsilon:.4f}")
            return epsilon
        except Exception as e:
            print(f"⚠️ Accountant 방식으로 epsilon 계산 실패: {e}")
        
        # 방법 2: get_epsilon 직접 호출 시도
        try:
            steps = epochs / sample_rate
            epsilon = privacy_engine.get_epsilon(delta=delta)
            print(f"🔒 계산된 ε (get_epsilon 방식): {epsilon:.4f}")
            return epsilon
        except Exception as e:
            print(f"⚠️ get_epsilon 방식으로 epsilon 계산 실패: {e}")
        
        # 방법 3: noise multiplier 추출 시도
        try:
            # Privacy Engine에서 노이즈 멀티플라이어 추출 시도
            noise_multiplier = privacy_engine.noise_multiplier
            if noise_multiplier is not None:
                print(f"🔒 추출된 노이즈 멀티플라이어: {noise_multiplier}")
                steps = epochs / sample_rate
                epsilon = compute_rdp_to_dp(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                    steps=steps,
                    delta=delta
                )
                print(f"🔒 계산된 ε (노이즈 멀티플라이어 추출 방식): {epsilon:.4f}")
                return epsilon
            else:
                print("⚠️ 노이즈 멀티플라이어 값을 찾을 수 없습니다.")
        except Exception as e:
            print(f"⚠️ 노이즈 멀티플라이어 추출 실패: {e}")
        
        return None
    
    except Exception as e:
        print(f"⚠️ Epsilon 계산 중 예외 발생: {e}")
        return None

def compute_rdp_to_dp(noise_multiplier, sample_rate, steps, delta=1e-5):
    """
    RDP에서 DP로 변환하여 epsilon 값을 계산합니다.
    (수동 계산용)
    """
    try:
        # Opacus 버전에 따라 다른 방식으로 계산
        try:
            # 최신 Opacus 버전 방식 시도 (0.15+)
            from opacus.accountants import RDPAccountant
            
            accountant = RDPAccountant()
            # 최신 버전에서는 step을 한 번에 여러 번 호출하는 대신, 한 번 호출에 step 수를 곱함
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
            # 실제 단계 수에 해당하는 프라이버시 비용을 계산
            for _ in range(int(steps) - 1):  # 이미 한 번 호출했으므로 -1
                accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
            
            epsilon = accountant.get_epsilon(delta=delta)
            return epsilon
        except (ImportError, AttributeError, TypeError) as e1:
            print(f"최신 Opacus 방식 실패: {e1}")
            
            try:
                # 이전 Opacus 버전 방식 시도
                from opacus.accountants.utils import get_noise_multiplier, get_epsilon
                
                # 스텝 수와 샘플링 비율 기반으로 epsilon 계산
                epsilon = get_epsilon(
                    target_delta=delta,
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                    steps=steps,
                    alphas=[2, 4, 8, 16, 32, 64]
                )
                return epsilon
            except ImportError as e2:
                print(f"이전 Opacus 방식도 실패: {e2}")
                
                # 두 방식 모두 실패한 경우 기본값 반환
                print("⚠️ Opacus epsilon 계산 실패, 기본값 사용")
                return 1.0 / noise_multiplier * sample_rate * steps  # 매우 간단한 근사치
        
    except Exception as e:
        print(f"⚠️ RDP에서 DP로 변환 중 오류 발생: {e}")
        return 3.0  # 기본값 반환

def estimate_global_epsilon(local_epsilons, weights):
    """
    각 로컬 모델의 epsilon 값을 기반으로 글로벌 모델의 epsilon 값을 추정합니다.
    가중 평균 기반으로 계산합니다.
    
    Args:
        local_epsilons (list): 로컬 모델들의 epsilon 값 리스트
        weights (list): 각 로컬 모델의 가중치 (데이터 크기 비율)
    
    Returns:
        float: 추정된 글로벌 epsilon 값
    """
    if not local_epsilons or None in local_epsilons:
        print("⚠️ 일부 로컬 epsilon 값이 없어 글로벌 epsilon을 계산할 수 없습니다.")
        return None
    
    try:
        # 가중 평균 계산
        global_epsilon = sum(e * w for e, w in zip(local_epsilons, weights))
        return global_epsilon
    except Exception as e:
        print(f"⚠️ 글로벌 epsilon 계산 중 오류 발생: {e}")
        return None

def compare_epsilon_values(results, model_type, output_dir='logs/epsilon_comparison'):
    """
    다양한 모델 및 설정의 epsilon 값을 비교하는 그래프와 테이블을 생성합니다.
    
    Args:
        results (dict): 각 모델 및 설정의 epsilon 값과 정확도 등을 포함하는 사전
        model_type (str): 모델 유형 (resnet50, densenet121, efficientnet_b0)
        output_dir (str): 결과를 저장할 디렉토리
        
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 결과를 그래프로 시각화
    plt.figure(figsize=(12, 8))
    
    # 색상 지정
    colors = {
        'very_strong': 'darkgreen',
        'strong': 'green',
        'moderate': 'orange',
        'weak': 'red',
        'very_weak': 'darkred',
        'unknown': 'gray'
    }
    
    # 데이터 포인트 플롯
    x_values = []
    y_values = []
    colors_list = []
    labels = []
    
    for name, data in results.items():
        if 'epsilon' in data and 'accuracy' in data:
            epsilon = data['epsilon']
            accuracy = data['accuracy']
            
            if epsilon is not None:
                x_values.append(epsilon)
                y_values.append(accuracy)
                
                privacy_level = get_privacy_level(epsilon)
                colors_list.append(colors[privacy_level])
                labels.append(f"{name} ({privacy_level})")
    
    # 산점도 그리기
    plt.scatter(x_values, y_values, c=colors_list, s=100)
    
    # 데이터 포인트에 레이블 추가
    for i, txt in enumerate(labels):
        plt.annotate(txt, (x_values[i], y_values[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # 그래프 설정
    plt.title(f'정확도 vs. ε (Epsilon) - {model_type}', fontsize=14)
    plt.xlabel('ε (Epsilon)', fontsize=12)
    plt.ylabel('정확도', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 프라이버시 수준 구분선 추가
    for level, (min_val, max_val) in EPSILON_RANGES.items():
        if level != 'very_weak' and min_val > 0:  # very_weak는 상한이 inf이므로 제외
            plt.axvline(x=min_val, color=colors[level], linestyle='--', alpha=0.5)
            plt.text(min_val, plt.ylim()[1]*0.95, level, 
                    rotation=90, verticalalignment='top', color=colors[level])
    
    # 파일로 저장
    graph_path = f"{output_dir}/{model_type}_epsilon_accuracy_{timestamp}.png"
    plt.savefig(graph_path)
    print(f"📊 비교 그래프 저장됨: {graph_path}")
    
    # 테이블 형태로 JSON 저장
    table_data = {
        'model_type': model_type,
        'timestamp': timestamp,
        'results': {}
    }
    
    for name, data in results.items():
        epsilon = data.get('epsilon')
        accuracy = data.get('accuracy')
        
        if epsilon is not None:
            privacy_level = get_privacy_level(epsilon)
        else:
            privacy_level = 'unknown'
            
        table_data['results'][name] = {
            'epsilon': epsilon,
            'accuracy': accuracy,
            'privacy_level': privacy_level,
            'other_metrics': {k: v for k, v in data.items() 
                             if k not in ['epsilon', 'accuracy']}
        }
    
    # JSON으로 저장
    json_path = f"{output_dir}/{model_type}_epsilon_comparison_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(table_data, f, indent=2)
    
    print(f"📄 비교 데이터 저장됨: {json_path}")
    
    return graph_path, json_path