import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import copy

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """한 에폭 동안 모델 학습
    
    Args:
        model: 학습할 모델
        dataloader: 데이터 로더
        optimizer: 옵티마이저
        criterion: 손실 함수
        device: 학습 장치 (cuda/cpu)
        
    Returns:
        평균 학습 손실
    """
    model.train()
    total_loss = 0
    
    # 모델이 올바른 디바이스에 있는지 확인
    model = model.to(device)
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # DP 사용 시 gradient clipping에서 디바이스 불일치가 발생할 수 있음
        # 이를 방지하기 위해 try-catch로 감싸고 필요시 재시도
        try:
            optimizer.step()
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                print(f"⚠️ 디바이스 불일치 오류 감지: {e}")
                # 옵티마이저 상태를 올바른 디바이스로 이동
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
                # 재시도
                try:
                    optimizer.step()
                except Exception as retry_e:
                    print(f"❌ 재시도 후에도 실패: {retry_e}")
                    raise retry_e
            else:
                raise e
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, criterion, device):
    """모델 평가
    
    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        device: 학습 장치 (cuda/cpu)
        
    Returns:
        (평균 손실, 예측값 리스트, 실제 라벨 리스트)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation", leave=False):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_targets

def train_and_evaluate(model, optimizer, criterion, train_dataset, val_dataset, test_dataset, args):
    # 데이터로더 생성
    train_loader = None
    if train_dataset:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 학습 전 평가
    if args.test_only:
        with torch.no_grad():
            test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, args.device)
            test_acc = accuracy_score(test_targets, test_preds)
            test_f1 = f1_score(test_targets, test_preds, average='weighted')
            print(f"📊 테스트 세트 성능: 손실={test_loss:.4f}, 정확도={test_acc:.4f}, F1={test_f1:.4f}")
            return test_acc, model
    
    # 학습 루프
    best_val_acc = 0
    best_model = None
    
    for epoch in range(args.epochs):
        # 학습
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")
        
        # 검증
        if val_loader:
            val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, args.device)
            val_acc = accuracy_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds, average='weighted')
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # 최고 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model.state_dict()
                print(f"✅ New best model saved with validation accuracy: {val_acc:.4f}")
    
    # 학습 완료 후 테스트
    if best_model:
        model.load_state_dict(best_model)
    
    with torch.no_grad():
        test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, args.device)
        test_acc = accuracy_score(test_targets, test_preds)
        test_f1 = f1_score(test_targets, test_preds, average='weighted')
        print(f"📊 테스트 세트 성능: 손실={test_loss:.4f}, 정확도={test_acc:.4f}, F1={test_f1:.4f}")
    
    return test_acc, best_model