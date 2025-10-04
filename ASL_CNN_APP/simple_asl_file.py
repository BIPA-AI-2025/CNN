import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io
import torchvision.transforms.functional as F
from PIL import Image
import io

st.set_page_config(page_title="ASL Predictor (Simple)", page_icon="🤟", layout="centered")

st.title("🤟 Simple ASL Prediction App")
st.caption("업로드한 체크포인트(.pth/.pt)와 이미지를 이용해 예측합니다 — 첨부 노트북의 전처리 흐름(그레이스케일, 28×28 리사이즈)을 따릅니다.")

# Constants (from notebook)
IMG_CHS = 1
IMG_WIDTH = 28
IMG_HEIGHT = 28

# ASL class names (24 classes: A-I, K-Y — J/Z 제외)
CLASS_NAMES = [
    "A","B","C","D","E","F","G","H","I",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y"
]

@st.cache_resource(show_spinner=False)
def load_model_from_bytes(model_bytes: bytes):
    # Try to load a pickled full model first (torch.save(model, ...))
    buffer = io.BytesIO(model_bytes)
    try:
        model = torch.load(buffer, map_location="cpu")
        model.eval()
        return model, "loaded_full_model"
    except Exception as e_full:
        # Fallback: assume it's a state_dict and try a light CNN head that matches the notebook's hint.
        class MySimpleASLNet(nn.Module):
            def __init__(self, num_classes=24):
                super().__init__()
                kernel_size = 3
                self.features = nn.Sequential(
                    nn.Conv2d(1, 25, kernel_size, stride=1, padding=1),
                    nn.BatchNorm2d(25),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),      # 14x14

                    nn.Conv2d(25, 50, kernel_size, stride=1, padding=1),
                    nn.BatchNorm2d(50),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.MaxPool2d(2, 2),      # 7x7

                    nn.Conv2d(50, 75, kernel_size, stride=1, padding=1),
                    nn.BatchNorm2d(75),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),      # 3x3
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(75 * 3 * 3, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                )
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        model = MySimpleASLNet(num_classes=len(CLASS_NAMES))
        buffer.seek(0)
        try:
            state_dict = torch.load(buffer, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            return model, "loaded_state_dict"
        except Exception as e_sd:
            raise RuntimeError(
                f"모델 로드 실패: full model 오류={type(e_full).__name__}: {e_full} | "
                f"state_dict 오류={type(e_sd).__name__}: {e_sd}"
            )

# Preprocess (from notebook): float32 scaling, resize to (28,28), grayscale
preprocess_trans = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.Grayscale()
])

def load_image_as_tensor(file) -> torch.Tensor:
    pil = Image.open(file).convert("RGB")  # robust, we'll grayscale later
    img_bytes = io.BytesIO()
    pil.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    img = tv_io.read_image(img_bytes, tv_io.ImageReadMode.RGB)  # CxHxW uint8
    img = preprocess_trans(img)  # float32 [0,1], resized, grayscale -> 1x28x28
    return img

with st.sidebar:
    st.header("1) 모델 업로드")
    model_file = st.file_uploader("ASL 모델 체크포인트 (.pth/.pt)", type=["pth", "pt"])
    st.header("2) 이미지 업로드")
    image_file = st.file_uploader("예측할 이미지 파일", type=["png", "jpg", "jpeg", "bmp", "webp"])

model = None
load_status = None
if model_file is not None:
    try:
        model, load_status = load_model_from_bytes(model_file.read())
        st.sidebar.success(f"모델 로드 성공: {load_status}")
    except Exception as e:
        st.sidebar.error(str(e))

col1, col2 = st.columns(2)
with col1:
    st.subheader("입력 이미지")
    if image_file is not None:
        st.image(image_file, caption="업로드한 이미지", use_container_width=True)
    else:
        st.info("왼쪽 사이드바에서 이미지를 업로드하세요.")

with col2:
    st.subheader("예측 결과")
    run = st.button("🔮 Predict", use_container_width=True, disabled=(model is None or image_file is None))
    if run:
        with st.spinner("예측 중..."):
            x = load_image_as_tensor(image_file)  # 1x28x28
            x = x.unsqueeze(0)  # BxCxHxW
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                topk = torch.topk(probs, k=min(3, probs.numel()))
                top_probs = topk.values.tolist()
                top_idxs = topk.indices.tolist()
                top_labels = [CLASS_NAMES[i] for i in top_idxs]

            st.success(f"Top-1: {top_labels[0]} (p={top_probs[0]:.4f})")
            if len(top_labels) > 1:
                st.write(f"Top-2: {top_labels[1]} (p={top_probs[1]:.4f})")
            if len(top_labels) > 2:
                st.write(f"Top-3: {top_labels[2]} (p={top_probs[2]:.4f})")

            st.caption(f"클래스 개수: {len(CLASS_NAMES)} | 입력 크기: 1×{IMG_WIDTH}×{IMG_HEIGHT}")
