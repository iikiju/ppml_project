import os
import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from model import get_model
from data_loader import load_data
import torch.nn.functional as F
import pandas as pd

# Grad-CAM 라이브러리: pytorch-grad-cam 사용
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    raise ImportError("pytorch-grad-cam 라이브러리를 설치하세요: pip install grad-cam")

def get_last_conv_layer(model, model_type):
    if model_type == 'resnet50':
        return model.layer4[-1]  # 마지막 Bottleneck 블록
    elif model_type == 'densenet121':
        return model.features[-2]  # 마지막 DenseBlock (features[-1]은 BatchNorm)
    elif model_type == 'efficientnet_b0':
        return model._conv_head  # 마지막 conv layer
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_experiment_configs():
    configs = [
        # (실험명, 모델 저장 경로, csv, augment 여부)
        ("cl_basic", "models/cl/{model_type}/basic_cl.pth", "data/meta_info_with_manufacturer_use_split_cl_new.csv", False),
        ("cl_aug",   "models/cl/{model_type}/aug_cl.pth",   "data/meta_info_with_manufacturer_use_split_cl_new.csv", True),
        ("fl_basic", "models/fl/{model_type}/basic_fl.pth", "data/meta_info_with_manufacturer_use_federated_new.csv", False),
        ("fl_aug",   "models/fl/{model_type}/aug_fl.pth",   "data/meta_info_with_manufacturer_use_federated_new.csv", True),
    ]
    return configs

def custom_overlay(image, cam, threshold=0.5, alpha=0.4):
    """
    커스텀 overlay 함수 - 임계값 적용하여 강한 활성화 영역만 표시
    image: [H, W, 3] RGB 이미지 (0~1)
    cam: [H, W] heatmap (0~1)
    threshold: heatmap 임계값
    alpha: overlay 투명도
    """
    # 임계값 적용 - 임계값 이상인 영역만 표시
    cam_binary = (cam > threshold).astype(np.float32)
    
    # 임계값을 통과한 영역의 원본 cam 값 유지
    cam_thresholded = cam * cam_binary
    
    # heatmap을 컬러로 변환
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_thresholded), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0
    
    # 원본 이미지를 0~255 범위로 변환
    image_255 = (image * 255).astype(np.uint8).astype(np.float32)
    
    # overlay - 임계값을 통과한 영역에만 적용
    overlay = image_255 * (1 - alpha * cam_binary[..., None]) + \
             heatmap * 255 * alpha * cam_binary[..., None]
    
    return overlay.astype(np.uint8)

def generate_gradcam_image(image_tensor, cam, pred, label, pred_prob, model_name):
    """
    Grad-CAM 이미지 생성 (저장하지 않고 numpy array 반환)
    """
    # image_tensor: torch.Tensor [1, H, W] or [H, W]
    # cam: np.ndarray [H, W], 0~1
    image = image_tensor.squeeze().cpu().numpy()  # [H, W]
    image_rgb = np.stack([image]*3, axis=-1)  # [H, W, 3]
    
    # 커스텀 overlay 적용 (임계값 0.5)
    cam_img = custom_overlay(image_rgb, cam, threshold=0.5, alpha=0.4)
    
    # 모델명과 예측/정답 정보 overlay
    text1 = f"{model_name}"
    text2 = f"Pred: {pred} ({pred_prob:.3f})"
    text3 = f"Label: {label}"
    
    cv2.putText(cam_img, text1, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(cam_img, text2, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(cam_img, text3, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    
    return cam_img

def concat_model_results(images_dict, exp_name, dicom_path):
    """
    여러 모델의 결과를 수평으로 concat
    images_dict: {model_name: image_array}
    """
    model_names = ['resnet50', 'densenet121', 'efficientnet_b0']
    concat_images = []
    
    for model_name in model_names:
        if model_name in images_dict:
            concat_images.append(images_dict[model_name])
        else:
            # 모델 결과가 없으면 빈 이미지 생성
            h, w = 224, 224  # 기본 크기
            if concat_images:
                h, w = concat_images[0].shape[:2]
            empty_img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(empty_img, f"{model_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,128), 2)
            cv2.putText(empty_img, "Not Available", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1)
            concat_images.append(empty_img)
    
    # 수평으로 concat
    result = np.hstack(concat_images)
    
    # 실험명 추가
    cv2.putText(result, f"Experiment: {exp_name.upper()}", (10, result.shape[0]-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='gradcam_results')
    args = parser.parse_args()

    model_types = ['resnet50', 'densenet121', 'efficientnet_b0']
    configs = get_experiment_configs()
    
    # 각 실험별로 처리
    for exp_name, model_path_tpl, csv_path, augment in configs:
        print(f"\n🔥 Processing experiment: {exp_name}")
        
        # CSV 파일 로드하여 test set의 dicom_img_path 가져오기
        df = pd.read_csv(csv_path)
        test_df = df[df['split'] == 'test']
        test_paths = test_df['dicom_img_path'].tolist()
        
        # test set 로드 (한 번만)
        test_dataset = load_data(csv_path, 'test', augment=False)
        
        # 결과 저장 폴더
        concat_out_dir = os.path.join(args.output_dir, 'concat', exp_name)
        os.makedirs(concat_out_dir, exist_ok=True)
        
        # 각 모델별 결과 저장용 딕셔너리
        all_results = {}  # {idx: {model_name: gradcam_image}}
        
        # 모든 모델에 대해 처리
        for model_type in model_types:
            model_path = model_path_tpl.format(model_type=model_type)
            if not os.path.exists(model_path):
                print(f"  [Skip] {model_type}: model not found at {model_path}")
                continue
            
            print(f"  [Process] {model_type}: {model_path}")
            
            # 모델 로드
            model = get_model(model_type)
            model.load_state_dict(torch.load(model_path, map_location=args.device))
            model.eval()
            model.to(args.device)
            
            # Grad-CAM 준비
            target_layer = get_last_conv_layer(model, model_type)
            cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(args.device=='cuda'))
            
            # test set 순회
            for idx in tqdm(range(len(test_dataset)), desc=f"  {model_type}"):
                image, label = test_dataset[idx]  # image: [1, H, W]
                input_tensor = image.unsqueeze(0).to(args.device)  # [1, 1, H, W]
                
                # 모델 예측
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    pred_prob = probs[0, pred].item()
                
                # Grad-CAM 생성
                targets_pred = [ClassifierOutputTarget(pred)]
                grayscale_cam_pred = cam(input_tensor=input_tensor, targets=targets_pred)[0]  # [H, W], 0~1
                
                # Grad-CAM 이미지 생성
                gradcam_img = generate_gradcam_image(image, grayscale_cam_pred, pred, label, pred_prob, model_type.upper())
                
                # 결과 저장
                if idx not in all_results:
                    all_results[idx] = {}
                all_results[idx][model_type] = gradcam_img
        
        # 각 이미지별로 모델 결과 concat하여 저장
        print(f"  [Concat] Creating concatenated images...")
        for idx in tqdm(range(len(test_dataset)), desc=f"  Concat {exp_name}"):
            if idx in all_results and all_results[idx]:  # 최소 하나의 모델 결과가 있는 경우
                dicom_path = test_paths[idx].replace('/', '_').replace('\\', '_')
                
                # 모델별 결과 concat
                concat_img = concat_model_results(all_results[idx], exp_name, dicom_path)
                
                # 최종 이미지 저장
                out_path = os.path.join(concat_out_dir, f"{dicom_path}.png")
                cv2.imwrite(out_path, concat_img)
        
        print(f"  ✅ {exp_name} completed!")
            
    print("\n🎉 모든 Grad-CAM concat 결과 저장 완료!")

if __name__ == "__main__":
    main() 