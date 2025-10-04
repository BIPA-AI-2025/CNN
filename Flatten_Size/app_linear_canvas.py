import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist_cnn import MNIST_CNN
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Streamlit 기본 설정
# -------------------------
st.set_page_config(page_title="✍️ 손글씨 숫자 인식", layout="centered")
st.title("Handwritten Digit Recognition using nn.Linear ✍️ ")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의
model =  nn.Sequential(
            nn.Flatten(),       # (B, 28, 28) → (B, 784)
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
).to(device)

# -------------------------
# 모델 로드
# -------------------------
model.load_state_dict(torch.load("./model/mnist_linear_model.pth", map_location=device))
model.eval()

# -------------------------
# 캔버스 생성
# -------------------------
st.markdown("## 캔버스에 숫자를 그려보세요 (0~9)")
canvas_result = st_canvas(
    fill_color="white",          # 그릴 때 채우는 색
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# 입력 이미지 처리 함수
def preprocess_image(image):
    image = ImageOps.grayscale(image)     # 흑백 변환
    image = image.resize((28, 28))        # MNIST 크기
    transform = transforms.ToTensor()     # [0, 1], (1, 28, 28)
    image = transform(image).unsqueeze(0) # 배치 추가 → (1, 1, 28, 28)
    return image.to(device)

# -------------------------
# 예측 버튼 클릭 시 실행
# -------------------------
if st.button("예측하기") and canvas_result.image_data is not None:
    
    # 1) 캔버스 RGBA → 흑백 반전 이미지(PIL)
    #    - 캔버스는 배경 흰색(255), 글씨 검정(0)인데 모델 입력도 "검정글씨/흰배경"을 기대하므로
    #      혹시 색 뒤집힌 경우를 대비해 R 채널을 반전하여 정규화
    rgba = canvas_result.image_data.astype(np.uint8)
    # R 채널만 사용(흑백 근사), 글씨/배경 반전
    img_pil = Image.fromarray((255 - rgba[:, :, 0]).astype(np.uint8))

    # 2) 전처리 함수로 텐서화
    input_tensor = preprocess_image(img_pil)  # (1, 1, 28, 28)

    # 3. 모델 추론 (Top-3)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze()       # (10,)
        topk_probs, topk_indices = torch.topk(probs, k=3)    # Top-3

    # 결과 출력
    st.image(img_pil, caption="28x28 전처리 이미지", width=150)
    st.markdown("### Top-3 예측 결과")
    
    # Top-3 막대 그래프 시각화
    labels = [str(topk_indices[i].item()) for i in range(3)]
    scores = [topk_probs[i].item() * 100 for i in range(3)]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, scores, color=['orange', 'gray', 'gray'])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Predicted Digit")
    ax.set_title("Top-3 Prediction Probabilities")

    # 각 막대 위에 확률 수치 표시
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2, f"{height:.2f}%", ha='center')

    st.pyplot(fig)

else:
    st.info("캔버스에 숫자를 그리고 '예측하기' 버튼을 눌러보세요.")

st.markdown("---")
st.caption("PyTorch + CNN + Streamlit + drawable-canvas")
