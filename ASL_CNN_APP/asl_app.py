import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# import torchvision.io as tv_io
from PIL import Image

# 모델 블록 정의
class MyConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.model(x)

# ASL 모델 아키텍처
IMG_CHS = 1
IMG_WIDTH, IMG_HEIGHT = 28, 28
N_CLASSES = 24  

class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            MyConvBlock(IMG_CHS, 25, 0.0),
            MyConvBlock(25, 50, 0.2),
            MyConvBlock(50, 75, 0.0),
            nn.Flatten(),
            nn.Linear(75 * 3 * 3, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, N_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

# 레이블 매핑 (예시)
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
    20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}
# TODO: 필요에 따라 J, Z를 포함하거나 순서를 변경하세요

# 장치 설정 및 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ASLModel().to(device)
model_path = './asl_cnn_model.pth'
model = torch.load(model_path, map_location=device).to(device)
model.eval()

# 전처리 정의
preprocess_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.Grayscale()  # From Color to Gray
])

# Streamlit UI
st.title("ASL Alphabet Prediction")
st.write("이미지를 업로드하거나 카메라로 촬영하여 ASL 알파벳을 예측합니다.")

uploaded_file = st.file_uploader("Upload an ASL image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # PIL 이미지로 읽기
    image = Image.open(uploaded_file).convert("RGB")
    
    # Tensor로 변환
    image_tensor = preprocess_trans(image)

    # 확인
    st.image(image, caption="업로드된 이미지")
    st.write("Tensor shape:", image_tensor.shape)


if uploaded_file is not None:
    st.image(uploaded_file, caption="Input Image", use_container_width=True)
    if st.button("Predict"):
        image = Image.open(uploaded_file).convert("RGB")
        img_tensor = preprocess_trans(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            pred_label = label_map[int(pred_idx)]
            st.write(f"Predicted: **{pred_label}** ({float(confidence)*100:.2f}%)")

