import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ---------------- 모델 정의 ----------------
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

# ---------------- 설정 및 모델 로드 ----------------
label_map = {i: chr(ord('A') + i) for i in range(24) if i != 9}  # J, Z 제외
label_map = dict(enumerate([c for c in label_map.values()]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './asl_cnn_model.pth'
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# ---------------- 전처리 함수 ----------------
def center_square_crop(image_pil):
    width, height = image_pil.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    return image_pil.crop((left, top, left + min_dim, top + min_dim))

def preprocess_image(image_pil):
    image_cropped = center_square_crop(image_pil)
    # 정적인 transform 만 적용함
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor()
    ])
    return transform(image_cropped), image_cropped

# ---------------- Streamlit UI ----------------
st.title("🤟 ASL 알파벳 예측기")
st.write("이미지를 업로드하거나 카메라로 촬영하여 손 모양을 예측해보세요.")

# 세션 상태 초기화
st.session_state.setdefault("prediction_result", None)

# 탭 구분
tab1, tab2 = st.tabs(["📁 이미지 업로드", "📸 카메라 촬영"])

input_image = None

# 업로드 탭
with tab1:
    uploaded_file = st.file_uploader("ASL 이미지 업로드", type=['png', 'jpg', 'jpeg'], key="upload_input")
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
    else:
        st.session_state.prediction_result = None

# 카메라 탭
with tab2:
    camera_image = st.camera_input("카메라로 손 모양을 촬영하세요", key="camera_input")
    if camera_image is not None:
        input_image = Image.open(camera_image).convert("RGB")
    else:
        st.session_state.prediction_result = None

# 이미지 표시 및 예측
if input_image is not None:
    cropped_image = center_square_crop(input_image)
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_image, caption="📸 원본 이미지", use_container_width=True)
        st.write(f"원본 크기: {input_image.size[0]}×{input_image.size[1]}")
    with col2:
        st.image(cropped_image, caption="✂ 중앙 Crop 이미지", use_container_width=True)
        st.write(f"Crop 크기: {cropped_image.size[0]}×{cropped_image.size[1]}")

    if st.button("Predict"):
        img_tensor, _ = preprocess_image(input_image)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            topk_probs, topk_indices = torch.topk(probs, 3)

        result_lines = []
        for i in range(3):
            idx = int(topk_indices[0][i])
            if idx in label_map:
                label = label_map[idx]
                prob = float(topk_probs[0][i]) * 100
                result_lines.append(f"{i+1}. **{label}** ({prob:.2f}%)")

        st.session_state.prediction_result = result_lines

# 예측 결과 출력
if st.session_state.prediction_result:
    for line in st.session_state.prediction_result:
        st.success(line)