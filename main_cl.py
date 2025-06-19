import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import get_model
from data_loader import load_data
from train_utils import train_and_evaluate, evaluate
import random
import numpy as np
from torch.utils.data import DataLoader

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

def main(args):
    # 랜덤 시드 설정
    if args.seed is not None:
        set_seed(args.seed)
    
    # 데이터셋 로드
    train_dataset = load_data(args.csv, 'train', augment=args.augment)
    val_dataset = load_data(args.csv, 'val', augment=False)
    test_dataset = load_data(args.csv, 'test')

    # 모델 선택
    model = get_model(args.model).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 테스트 모드: 저장된 모델 불러와서 테스트만 수행
    if args.test_only and args.load_path:
        print(f"⏩ Testing only mode, loading model from: {args.load_path}")
        model.load_state_dict(torch.load(args.load_path, map_location=args.device))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, args.device)
        test_acc = np.mean(np.array(test_preds) == np.array(test_targets))
        print(f"🧪 Test Accuracy: {test_acc:.4f}")
        return test_acc

    # 학습 및 평가
    best_acc, best_model_state = train_and_evaluate(model, optimizer, criterion, train_dataset, val_dataset, test_dataset, args)
    
    # 모델 저장
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(best_model_state, args.save_path)
        print(f"✅ Model saved to: {args.save_path}")
    
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='resnet50', help='resnet50 | densenet121 | efficientnet_b0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='', help='Path to save the trained model')
    parser.add_argument('--load_path', type=str, default='', help='Path to load a pretrained model')
    parser.add_argument('--test_only', action='store_true', help='Only test, no training')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args)