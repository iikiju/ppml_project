import torch
from model import get_model
from data_loader import load_data
from dp_utils import attach_dp, fix_model_for_dp, calculate_epsilon, estimate_global_epsilon, compute_rdp_to_dp, get_privacy_level
from train_utils import train_one_epoch, evaluate
import torch.nn as nn
import torch.optim as optim
import copy
import os
from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import json
from datetime import datetime
import random

def set_seed(seed):
    """재현성을 위해 랜덤 시드를 설정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🔒 랜덤 시드 고정됨: {seed}")

def average_weights(w):
    """
    Returns the average of the weights from all clients.
    Handles different data types safely to prevent casting errors.
    """
    if not w:
        raise ValueError("가중치 리스트가 비어있습니다")
    
    w_avg = copy.deepcopy(w[0])
    
    for key in w_avg.keys():
        # 참조 텐서의 속성 확인
        reference_tensor = w[0][key]
        tensor_dtype = reference_tensor.dtype
        tensor_device = reference_tensor.device
        
        # 정수형 타입인지 확인
        is_integer_type = tensor_dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
        
        # 평균 계산을 위한 초기화
        if is_integer_type:
            # 정수형의 경우 float로 계산
            w_avg[key] = w[0][key].float()
        else:
            w_avg[key] = w[0][key].clone()
        
        # 나머지 가중치들을 더함
        for i in range(1, len(w)):
            try:
                current_tensor = w[i][key].to(device=tensor_device)
                if is_integer_type:
                    w_avg[key] += current_tensor.float()
                else:
                    w_avg[key] += current_tensor.to(dtype=tensor_dtype)
            except Exception as e:
                print(f"⚠️ 키 {key}에서 가중치 합산 오류: {e}")
                continue
        
        # 평균 계산
        try:
            w_avg[key] = torch.div(w_avg[key], len(w))
            
            # 정수형의 경우 원래 타입으로 변환
            if is_integer_type:
                w_avg[key] = w_avg[key].round().to(dtype=tensor_dtype)
            else:
                w_avg[key] = w_avg[key].to(dtype=tensor_dtype)
                
        except Exception as e:
            print(f"⚠️ 키 {key}에서 평균 계산 오류: {e}")
            # 오류 발생 시 첫 번째 클라이언트의 가중치 사용
            w_avg[key] = w[0][key].clone()
    
    return w_avg

def train_federated(args):
    # 랜덤 시드 설정
    if args.seed is not None:
        set_seed(args.seed)
    
    # CSV 파일에서 고유 클라이언트 ID 추출
    df = pd.read_csv(args.csv)
    
    # 유효한 클라이언트 ID만 추출 (nan 제외, 문자열로 변환)
    train_df = df[df['split'] == 'train']
    if 'client_id' not in train_df.columns:
        raise ValueError("client_id 열이 CSV 파일에 없습니다.")
    
    # NaN 값을 제외하고 유효한 client ID만 추출
    client_ids = [str(cid) for cid in train_df['client_id'].unique() if str(cid) != 'nan' and pd.notna(cid)]
    
    if not client_ids:
        raise ValueError("유효한 client_id가 없습니다. CSV 파일을 확인하세요.")
    
    print(f"🔍 발견된 클라이언트 ID: {', '.join(client_ids)}")
    
    # 모델 생성
    global_model = get_model(args.model).to(args.device)
    
    # DP를 사용하는 경우 모델 수정
    if args.dp:
        print("🔒 DP 모델로 변환 중...")
        global_model = fix_model_for_dp(global_model)
    
    # 프라이버시 예산 추적을 위한 변수들
    privacy_logs = {
        'rounds': [],
        'noise_multiplier': args.noise,
        'max_grad_norm': args.max_grad_norm,
        'delta': args.delta,
        'target_epsilon': args.target_epsilon
    }
    
    # 학습 시작
    for round_idx in range(args.rounds):
        print(f"\n=== 라운드 {round_idx+1}/{args.rounds} ===")
        
        # 클라이언트 모델 가중치 저장을 위한 딕셔너리
        local_weights = []
        local_sizes = []
        local_epsilons = []
        
        round_clients = {}
        
        # 각 클라이언트에서 학습
        for client_id in client_ids:
            print(f"\n👤 클라이언트 {client_id} 학습 중...")
            
            # 데이터 로드
            train_dataset = load_data(args.csv, 'train', client_id=client_id, augment=args.augment)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            print(f"📊 클라이언트 {client_id}의 데이터 크기: {len(train_dataset)}")
            
            # 클라이언트 모델 초기화 (글로벌 모델 복사)
            local_model = get_model(args.model).to(args.device)
            
            # DP를 사용하는 경우 모델 수정
            if args.dp:
                local_model = fix_model_for_dp(local_model)
                # 글로벌 모델도 DP 호환 버전으로 변환 (첫 라운드인 경우)
                if round_idx == 0:
                    global_model = fix_model_for_dp(global_model)
            
            # 글로벌 모델에서 상태 복사 (DP 변환 후)
            local_model.load_state_dict(global_model.state_dict())
            
            # 옵티마이저 설정
            optimizer = optim.Adam(local_model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()
            
            # 클라이언트 정보 초기화
            client_info = {
                'data_size': len(train_dataset),
                'epsilon': None,
                'privacy_engine': None
            }
            
            # 샘플링 비율 계산
            sampling_rate = args.batch_size / len(train_dataset)
            
            # DP를 사용하는 경우 프라이버시 엔진 설정
            if args.dp:
                try:
                    local_model, optimizer, train_loader, privacy_engine = attach_dp(
                        local_model, 
                        optimizer, 
                        train_loader,
                        noise_multiplier=args.noise,
                        max_grad_norm=args.max_grad_norm,
                        device=args.device
                    )
                    client_info['privacy_engine'] = privacy_engine
                    print(f"🔒 DP 적용됨 (noise={args.noise}, max_grad_norm={args.max_grad_norm}, 샘플링 비율={sampling_rate:.4f})")
                except Exception as e:
                    print(f"❌ DP 적용 실패: {e}")
                    print("⚠️ DP 없이 계속합니다")
            
            # 로컬 모델 학습
            for epoch in range(args.local_epochs):
                train_loss = train_one_epoch(local_model, train_loader, optimizer, criterion, args.device)
                print(f"📈 에폭 {epoch+1}/{args.local_epochs}, 손실: {train_loss:.4f}")
            
            # 에폭 완료 후 epsilon 계산
            if args.dp and client_info['privacy_engine'] is not None:
                # Opacus로 epsilon 계산 시도
                client_epsilon = calculate_epsilon(
                    client_info['privacy_engine'], 
                    sample_rate=sampling_rate,
                    epochs=args.local_epochs,
                    delta=args.delta
                )
                
                # 계산 실패 시 수동 계산
                if client_epsilon is None:
                    print("🔄 수동으로 epsilon 계산 시도 중...")
                    steps = args.local_epochs / sampling_rate
                    client_epsilon = compute_rdp_to_dp(
                        noise_multiplier=args.noise,
                        sample_rate=sampling_rate,
                        steps=steps,
                        delta=args.delta
                    )
                    print(f"📊 수동 계산된 DP 매개변수: (ε={client_epsilon:.4f}, δ={args.delta})")
                
                client_info['epsilon'] = client_epsilon
                
                if client_epsilon is not None:
                    print(f"🔒 클라이언트 {client_id}의 ε 값: {client_epsilon:.4f}")
                    # epsilon 값의 프라이버시 수준 표시
                    privacy_level = get_privacy_level(client_epsilon)
                    print(f"🔒 프라이버시 보호 수준: {privacy_level}")
                    
                    # 목표 epsilon과 비교
                    if args.target_epsilon and client_epsilon > args.target_epsilon:
                        print(f"⚠️ 주의: 현재 ε 값 ({client_epsilon:.4f})이 목표 ε 값 ({args.target_epsilon:.4f})보다 큽니다.")
                    
                    local_epsilons.append(client_epsilon)
                else:
                    print(f"⚠️ 클라이언트 {client_id}의 ε 값을 계산할 수 없습니다")
                    local_epsilons.append(None)
            
            # 클라이언트 가중치 수집
            local_weights.append(local_model.state_dict())
            local_sizes.append(len(train_dataset))
            round_clients[client_id] = client_info
        
        # 글로벌 모델 업데이트 (가중치 평균)
        with torch.no_grad():
            # 데이터 크기에 비례하는 가중치 계산
            total_size = sum(local_sizes)
            weights = [size / total_size for size in local_sizes]
            
            # 글로벌 모델의 각 파라미터 업데이트
            try:
                # 모든 로컬 모델이 일관된 키를 가지는지 확인
                keys_set = [set(w.keys()) for w in local_weights]
                common_keys = set.intersection(*keys_set)
                
                if not common_keys:
                    raise ValueError("로컬 모델들 간에 공통된 파라미터 키가 없습니다")
                
                # 모든 모델이 DP로 변환되면서 일부 키가 변경될 수 있음
                # 일관된 키 집합을 사용하여 가중 평균 계산
                global_weights = {}
                for key in common_keys:
                    # 각 텐서의 원래 데이터 타입과 디바이스 보존
                    reference_tensor = local_weights[0][key]
                    tensor_dtype = reference_tensor.dtype
                    tensor_device = reference_tensor.device
                    
                    # 정수형 타입인지 확인 (예: Long, Int 등)
                    is_integer_type = tensor_dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
                    
                    # 가중 평균 계산을 위한 초기화
                    if is_integer_type:
                        # 정수형 파라미터의 경우 float로 계산 후 다시 정수형으로 변환
                        global_weights[key] = torch.zeros_like(reference_tensor, dtype=torch.float32, device=tensor_device)
                    else:
                        global_weights[key] = torch.zeros_like(reference_tensor, dtype=tensor_dtype, device=tensor_device)
                    
                    # 가중 평균 계산
                    for i in range(len(local_weights)):
                        local_tensor = local_weights[i][key].to(device=tensor_device)
                        if is_integer_type:
                            # 정수형의 경우 float로 변환하여 계산
                            global_weights[key] += weights[i] * local_tensor.float()
                        else:
                            global_weights[key] += weights[i] * local_tensor.to(dtype=tensor_dtype)
                    
                    # 최종 가중치를 원래 데이터 타입으로 변환
                    if is_integer_type:
                        # 정수형의 경우 반올림 후 원래 타입으로 변환
                        global_weights[key] = global_weights[key].round().to(dtype=tensor_dtype)
                    else:
                        # 부동소수점형의 경우 원래 타입 유지
                        global_weights[key] = global_weights[key].to(dtype=tensor_dtype)
                
                # 모델 업데이트
                missing_keys, unexpected_keys = global_model.load_state_dict(global_weights, strict=False)
                
                if missing_keys:
                    print(f"⚠️ 글로벌 모델 업데이트 중 누락된 키: {missing_keys}")
                if unexpected_keys:
                    print(f"⚠️ 글로벌 모델 업데이트 중 예상치 않은 키: {unexpected_keys}")
            
            except Exception as e:
                print(f"❌ 글로벌 모델 업데이트 중 오류 발생: {str(e)}")
                print("⚠️ 첫 번째 클라이언트의 가중치로 폴백합니다")
                try:
                    # 폴백 시에도 안전한 로딩
                    global_model.load_state_dict(local_weights[0], strict=False)
                except Exception as fallback_error:
                    print(f"❌ 폴백 로딩도 실패: {str(fallback_error)}")
                    # 더 안전한 폴백: 키별로 개별 처리
                    for key, value in local_weights[0].items():
                        try:
                            if key in global_model.state_dict():
                                global_model.state_dict()[key].copy_(value)
                        except Exception as key_error:
                            print(f"⚠️ 키 {key} 복사 실패: {str(key_error)}")
                            continue
        
        # 글로벌 epsilon 계산 (가중치 적용)
        global_epsilon = None
        if args.dp and local_epsilons:
            global_epsilon = estimate_global_epsilon(local_epsilons, weights)
            if global_epsilon is not None:
                print(f"🔒 라운드 {round_idx+1}의 글로벌 ε 추정값: {global_epsilon:.4f}")
                # 글로벌 epsilon 값의 프라이버시 수준 표시
                privacy_level = get_privacy_level(global_epsilon)
                print(f"🔒 글로벌 프라이버시 보호 수준: {privacy_level}")
                
                # 목표 epsilon과 비교
                if args.target_epsilon and global_epsilon > args.target_epsilon:
                    print(f"⚠️ 주의: 글로벌 ε 값 ({global_epsilon:.4f})이 목표 ε 값 ({args.target_epsilon:.4f})보다 큽니다.")
            else:
                print(f"⚠️ 라운드 {round_idx+1}의 글로벌 ε 값을 추정할 수 없습니다")
        
        # 라운드 정보 저장
        round_info = {
            'round': round_idx + 1,
            'clients': {
                client_id: {
                    'data_size': info['data_size'],
                    'epsilon': info['epsilon']
                } for client_id, info in round_clients.items()
            },
            'global_epsilon': global_epsilon if args.dp else None,
            'weights': weights
        }
        privacy_logs['rounds'].append(round_info)
        
        print(f"✅ 라운드 {round_idx+1} 완료: 글로벌 모델 업데이트됨")
    
    # 모델 저장
    if args.save_path:
        torch.save(global_model.state_dict(), args.save_path)
        print(f"💾 모델 저장됨: {args.save_path}")
        
        # 프라이버시 로그 저장
        if args.dp:
            log_dir = os.path.dirname(args.save_path)
            if not log_dir:
                log_dir = '.'
            model_name = os.path.splitext(os.path.basename(args.save_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            privacy_log_path = f"{log_dir}/{model_name}_privacy_{timestamp}.json"
            
            with open(privacy_log_path, 'w') as f:
                json.dump(privacy_logs, f, indent=2, default=lambda x: None if not isinstance(x, (int, float, str, bool, list, dict)) else str(x))
            print(f"📊 프라이버시 로그 저장됨: {privacy_log_path}")
    
    return global_model

def test_federated(args, model=None):
    # 랜덤 시드 설정
    if args.seed is not None:
        set_seed(args.seed)
    
    # 모델 로드 또는 사용
    if model is None:
        model = get_model(args.model).to(args.device)
        
        # 만약 DP 모델을 로드하는 경우, 먼저 모델을 DP 호환되게 변환
        if args.dp or ".pth" in args.load_path and any(dp_term in args.load_path for dp_term in ["dp", "DP"]):
            print("🔒 DP 호환 모델로 변환하여 로드합니다...")
            model = fix_model_for_dp(model)
            
        # 모델 가중치 로드 (strict=False로 설정하여 키 불일치 허용)
        state_dict = torch.load(args.load_path, map_location=args.device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ 모델 로드 중 누락된 키: {len(missing_keys)}개")
        if unexpected_keys:
            print(f"⚠️ 모델 로드 중 예상치 않은 키: {len(unexpected_keys)}개")
            
        print(f"📂 모델 로드됨: {args.load_path}")
    
    # 테스트 데이터 로드 - client_id 없이 전체 테스트 데이터 사용
    test_dataset = load_data(args.csv, 'test', client_id=None, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\n📊 테스트 데이터 크기: {len(test_dataset)} 샘플")
    
    # 전체 테스트 수행
    test_loss, preds, targets = evaluate(model, test_loader, nn.CrossEntropyLoss(), args.device)
    overall_accuracy = accuracy_score(targets, preds)
    overall_f1 = f1_score(targets, preds, average='weighted')
    
    print("\n=== 전체 테스트 결과 ===")
    print(f"전체 정확도: {overall_accuracy:.4f}")
    print(f"전체 F1 점수: {overall_f1:.4f}")
    
    # DP 적용 모델인 경우 epsilon 정보 표시
    if args.dp and args.target_epsilon:
        print(f"🔒 목표 ε 값: {args.target_epsilon} (δ={args.delta})")
    
    # 각 클라이언트별 테스트 (클라이언트 ID가 있는 경우에만)
    df = pd.read_csv(args.csv)
    client_metrics = {}
    
    # 클라이언트 ID가 있는지 확인
    if 'client_id' in df.columns:
        # 테스트 데이터에 클라이언트 ID가 있는지 확인
        test_df = df[df['split'] == 'test']
        has_client_ids = 'client_id' in test_df.columns and not test_df['client_id'].isna().all()
        
        if has_client_ids:
            # 유효한 클라이언트 ID만 추출
            client_ids = [str(cid) for cid in test_df['client_id'].unique() if str(cid) != 'nan' and pd.notna(cid)]
            
            if client_ids:
                print("\n=== 클라이언트별 테스트 결과 ===")
                
                for client_id in client_ids:
                    # 클라이언트별 테스트 데이터 로드
                    client_test_dataset = load_data(args.csv, 'test', client_id=client_id, augment=False)
                    
                    if len(client_test_dataset) > 0:
                        client_test_loader = DataLoader(client_test_dataset, batch_size=args.batch_size, shuffle=False)
                        
                        # 평가
                        test_loss, preds, targets = evaluate(model, client_test_loader, nn.CrossEntropyLoss(), args.device)
                        accuracy = accuracy_score(targets, preds)
                        f1 = f1_score(targets, preds, average='weighted')
                        
                        print(f"👤 클라이언트 {client_id} 결과:")
                        print(f"   손실: {test_loss:.4f}, 정확도: {accuracy:.4f}, F1 점수: {f1:.4f}")
                        
                        client_metrics[client_id] = {
                            'loss': test_loss,
                            'accuracy': accuracy,
                            'f1': f1,
                            'samples': len(client_test_dataset)
                        }
    
    return overall_accuracy, overall_f1, client_metrics

def main():
    parser = argparse.ArgumentParser(description='Federated Learning for Medical Imaging')
    parser.add_argument('--csv', type=str, default='data/meta_info_with_manufacturer_use_federated_new.csv',
                        help='CSV file with dataset information')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'densenet121', 'efficientnet_b0'],
                       help='Model architecture')
    parser.add_argument('--rounds', type=int, default=5, help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=2, help='Number of local epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save model')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load model for testing')
    parser.add_argument('--test_only', action='store_true', help='Only test the model')
    parser.add_argument('--dp', action='store_true', help='Use differential privacy')
    parser.add_argument('--noise', type=float, default=1.0, help='Noise multiplier for DP')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for DP')
    parser.add_argument('--delta', type=float, default=1e-5, help='Delta parameter for DP privacy guarantees')
    parser.add_argument('--target_epsilon', type=float, default=None, help='Target epsilon value for DP')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # 모델 학습 또는 테스트
    if args.test_only:
        if args.load_path is None:
            print("❌ 테스트 전용 모드에서는 --load_path가 필요합니다")
            return
        test_federated(args)
    else:
        # 모델 학습
        model = train_federated(args)
        # 학습된 모델 테스트
        test_federated(args, model)

if __name__ == "__main__":
    main()