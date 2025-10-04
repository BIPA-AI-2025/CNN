import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from PIL import Image

# =====================
# 설정 (필요시 아래 경로만 바꾸세요)
# =====================
# MODEL_PATH = "model/asl_nvidia_model.pth"  # torch.save(model, MODEL_PATH) 로 저장된 '완전 모델' 전용
MODEL_PATH = "model/asl_model_state.pth"   # torch.save(model.state_dict(),

IMG_SIZE = (28, 28)           # 첨부 노트북 기준

# Sign-MNIST 기준(J/Z 제외) 24클래스 — 필요없다면 결과는 인덱스로만 확인하세요.
CLASS_NAMES = [
    "A","B","C","D","E","F","G","H","I",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y"
]

st.set_page_config(page_title="Simple ASL Prediction App ", page_icon="🤟", layout="centered")
st.title("🤟 Simple ASL Prediction App")

class MyConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        kernel_size = 3
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.model(x)

flattened_img_size = 75 * 3 * 3
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1
N_CLASSES = 24

# Input 1 x 28 x 28
base_model = nn.Sequential(
    MyConvBlock(IMG_CHS, 25, 0), # 25 x 14 x 14
    MyConvBlock(25, 50, 0.2), # 50 x 7 x 7
    MyConvBlock(50, 75, 0),  # 75 x 3 x 3
    # Flatten to Dense Layers
    nn.Flatten(),
    nn.Linear(flattened_img_size, 512),
    nn.Dropout(.3),
    nn.ReLU(),
    nn.Linear(512, N_CLASSES)
)

# GPU 처리
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 (한 번만)
@st.cache_resource(show_spinner=False)  # 리소스(모델, DB 연결, 세션 객체 등) 를 재사용
def load_model(path: str, device_str: str):
    # model = torch.load(path, map_location="cpu")
    # state_dict = torch.load(path, map_location="cpu")   
    state_dict = torch.load(path, map_location=torch.device(device_str))  
    model = base_model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model = load_model(MODEL_PATH, str(device))
    st.caption(f"모델 로드 완료: {MODEL_PATH}")
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

# 전처리: PIL -> Grayscale -> Resize -> Tensor(float, [0,1])
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMG_SIZE),
    transforms.PILToTensor(),                     # -> CxHxW (uint8)
    transforms.ToDtype(torch.float32, scale=True) # [0,255] -> [0,1]
])

# 테스트 이미지 업로드
file = st.file_uploader("테스트 이미지 업로드 (png/jpg 등)", type=["png","jpg","jpeg","bmp","webp"])

# --------------------
# 입력 & 예측
# --------------------
if st.button("예측하기", disabled=(file is None)):
    with st.spinner("예측 중..."):
        pil = Image.open(file).convert("RGB")
        
        x = preprocess(pil).unsqueeze(0).to(device)  # <-- 입력도 GPU로
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # [24]
            top_probs, top_idxs = torch.topk(probs, k=3)
            top_probs = top_probs.tolist()
            top_idxs = top_idxs.tolist()

        st.image(pil, caption="업로드한 이미지", use_container_width=True)
        
        for rank, (idx, p) in enumerate(zip(top_idxs, top_probs), start=1):
            label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f"Class-{idx}"
            st.success(f"Top-{rank}: {label} ({p*100:.2f}%)")  # 확률을 %로 표시