import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image

# ------------------------------------------
# 🔧 모델 블록 정의
# ------------------------------------------
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

# ------------------------------------------
# 🔧 ASL 모델 아키텍처
# ------------------------------------------
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

# ------------------------------------------
# 🔠 라벨 매핑 (J, Z 제외)
# ------------------------------------------
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
    20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# ------------------------------------------
# ✅ 모델 로드
# ------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './asl_cnn_model.pth'
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# ------------------------------------------
# 🔧 이미지 전처리 함수
# ------------------------------------------
def preprocess_image(image_pil):
    # 📐 Step 1: 중앙 정사각형 crop
    width, height = image_pil.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    image_cropped = image_pil.crop((left, top, left + min_dim, top + min_dim))

    # Step 2: 전처리 (Grayscale + Resize + ToTensor)
    transform = transforms.Compose([
        transforms.Grayscale(),                    
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor()
    ])
    return transform(image_cropped)

# ------------------------------------------
# 🖼️ Streamlit UI
# ------------------------------------------
st.title("🤟 ASL 알파벳 예측기")
st.write("이미지를 업로드하거나 카메라로 촬영하여 손 모양을 예측해보세요.")

tab1, tab2 = st.tabs(["📁 이미지 업로드", "📸 카메라 촬영"])

# 이미지 업로드
with tab1:
    uploaded_file = st.file_uploader("ASL 이미지 업로드", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="업로드한 이미지", use_container_width=True)

# 카메라 입력
with tab2:
    camera_image = st.camera_input("카메라로 손 모양을 촬영하세요")
    if camera_image:
        camera_image_pil = Image.open(camera_image).convert("RGB")
        st.image(camera_image_pil, caption="카메라 이미지", use_container_width=True)
    
# 예측 버튼
if st.button("Predict"):
    if uploaded_file:
        input_image = uploaded_image
    elif camera_image:
        input_image = camera_image_pil
    else:
        st.warning("이미지를 업로드하거나 카메라로 촬영해주세요.")
        st.stop()

    # 전처리 및 예측
    input_tensor = preprocess_image(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        pred_label = label_map[int(pred_idx)]

    st.success(f"예측 결과: **{pred_label}** ({float(confidence)*100:.2f}%)")
