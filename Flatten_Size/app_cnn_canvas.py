# app_canvas.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from mnist_cnn import MNIST_CNN  # <- MNIST_CNN 정의는 이전과 동일하게 유지

# 페이지 설정
st.set_page_config(page_title="🖌️ 캔버스 숫자 분류기", layout="centered")
st.title("✍️ Handwritten Digit Classifier (MNIST CNN)")

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN().to(device)
model.load_state_dict(torch.load("./model/mnist_cnn_model.pth", map_location=device))
model.eval()

# 캔버스 설정
st.markdown("## ✏️ 숫자를 그려보세요 (0~9)")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 예측 버튼
if st.button("예측하기") and canvas_result.image_data is not None:
    img = canvas_result.image_data

    # 1. numpy image → PIL → grayscale
    img_pil = Image.fromarray((255 - img[:, :, 0]).astype(np.uint8))  # Invert black/white
    img_pil = img_pil.resize((28, 28))
    img_pil = ImageOps.grayscale(img_pil)

    # 2. 전처리 (Tensor 변환)
    transform = transforms.ToTensor()
    input_tensor = transform(img_pil).unsqueeze(0).to(device)  # (1, 1, 28, 28)

    # 3. 모델 추론
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).squeeze()[pred].item()

    # 4. 결과 출력
    st.image(img_pil, caption="28x28 전처리 이미지", width=150)
    st.markdown(f"### 예측 숫자: **{pred}**")
    st.markdown(f"정확도: `{confidence * 100:.2f}%`")

else:
    st.info("왼쪽 캔버스에 숫자를 그리고 예측 버튼을 눌러보세요.")

st.markdown("---")
st.caption("by ChatGPT · Streamlit + PyTorch + drawable-canvas")
