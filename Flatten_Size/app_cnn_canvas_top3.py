import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from mnist_cnn import MNIST_CNN  # <- 별도 파일에서 모델 정의
import matplotlib.pyplot as plt


# -------------------------
# 1. Streamlit 기본 설정
# -------------------------
st.set_page_config(page_title="✍️ 손글씨 숫자 인식", layout="centered")
st.title("Handwritten Digit Recognition usig PyTorch CNN ✍️")
st.markdown("---")

# -------------------------
# 2. 모델 로드
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN().to(device)
model.load_state_dict(torch.load("./model/mnist_cnn_model.pth", map_location=device))
model.eval()

# -------------------------
# 3. 캔버스 생성
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

# -------------------------
# 4. 예측 버튼 클릭 시 실행
# -------------------------
if st.button("예측하기") and canvas_result.image_data is not None:
    # 캔버스 이미지 가져오기
    img = canvas_result.image_data

    # 1. numpy → PIL 이미지 (흑백 & 색 반전), [:, :, 0]은 Red 채널만 추출 -> (H, W, 3) → (H, W)
    img_pil = Image.fromarray((255 - img[:, :, 0]).astype(np.uint8))  # 배경 흰색, 글씨 검정으로 반전
    img_pil = img_pil.resize((28, 28))  # MNIST 사이즈로 축소
    img_pil = ImageOps.grayscale(img_pil)

    # 2. 전처리 → Tensor
    transform = transforms.ToTensor()
    input_tensor = transform(img_pil).unsqueeze(0).to(device)  # (1, 1, 28, 28)

    # 3. 모델 추론 (Top-3)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze()       # (10,)
        topk_probs, topk_indices = torch.topk(probs, k=3)    # Top-3

    # 4. 결과 출력
    st.image(img_pil, caption="28x28 전처리 이미지", width=150)
    st.markdown("### Top-3 예측 결과")
    
    # Top-3 막대 그래프 시각화
    labels = [str(topk_indices[i].item()) for i in range(3)]
    scores = [topk_probs[i].item() * 100 for i in range(3)]
    
    for i in range(3):
        st.markdown(f"**{i+1}. Class {labels[i]}** — {scores[i]:.2f}% confidence")
    
    # fig, ax = plt.subplots()
    # ax.bar(labels, scores)
    # ax.set_ylabel("Confidence (%)")
    # ax.set_xlabel("Class")
    # ax.set_title("Top-3 Predictions")
    # st.pyplot(fig)

else:
    st.info("캔버스에 숫자를 그리고 '예측하기' 버튼을 눌러보세요.")

st.markdown("---")
st.caption("PyTorch + CNN + Streamlit + drawable-canvas")
