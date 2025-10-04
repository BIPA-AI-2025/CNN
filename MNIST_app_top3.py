import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# -------------------------------
# 디바이스 설정
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 저장된 모델 구조 정의
# -------------------------------
@st.cache_resource
def load_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).to(device)

    model.load_state_dict(torch.load("./model/my_MNIST_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# -------------------------------
# 학습 시 사용한 전처리 정의
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

# -------------------------------
# 숫자를 중심 정렬하는 함수
# -------------------------------
def center_image(pil_img, size=(28, 28)):
    """
    - 색 반전된 이미지에서 숫자 영역을 crop
    - 비율을 유지한 채 resize
    - 부족한 영역은 최소한으로 padding
    """
    img = pil_img.convert("L")
    img = img.point(lambda x: 0 if x < 128 else 255, '1')
    img = ImageOps.invert(img)

    bbox = img.getbbox()
    if not bbox:
        return img.resize(size, Image.Resampling.LANCZOS)

    img = img.crop(bbox)

    # 비율 유지하며 리사이즈 (긴 쪽을 20~26 정도로 설정 → 여백 약간 포함)
    img.thumbnail((22, 22), Image.Resampling.LANCZOS)

    # 중심 정렬용 흑백 배경 생성
    canvas = Image.new("L", size, 0)
    paste_x = (size[0] - img.size[0]) // 2
    paste_y = (size[1] - img.size[1]) // 2
    canvas.paste(img, (paste_x, paste_y))

    return canvas

# -------------------------------
# Streamlit UI
# -------------------------------
st.subheader("✍️ MNIST 손글씨 숫자 예측기")
st.write("⬇️ 아래에 숫자를 흰색으로 그리고 [예측하기]를 눌러보세요.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=18,                 
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=392,                      # 28 * 14
    width=392,                       
    drawing_mode="freedraw",
    key="canvas"
)

# -------------------------------
# 예측 버튼
# -------------------------------
if st.button("예측하기"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, :3].astype(np.uint8)
        pil_img = Image.fromarray(img).convert("RGB")

        # 색상 반전 및 중앙 정렬
        inverted = TF.invert(pil_img)
        centered = center_image(inverted)

        # 전처리 → 텐서
        input_tensor = transform(centered).unsqueeze(0).to(device)  # [1, 1, 28, 28]

        # 입력 이미지 시각화
        st.image(input_tensor[0][0].cpu().numpy(), caption="입력 이미지 (28x28)", width=100, clamp=True)

        # 예측 수행
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
            pred = int(np.argmax(probabilities))

        st.success(f"예측된 숫자: **{pred}**")

        # 🔹 Top-3 클래스 출력
        top3 = np.argsort(probabilities)[::-1][:3]
        st.subheader("Top-3 예측 결과")
        for rank, idx in enumerate(top3, start=1):
            st.write(f"{rank}. 숫자 {idx} → 확률: {probabilities[idx]:.4f}")

        # 🔸 확률 막대그래프 시각화
        st.subheader("클래스별 확률")
        fig, ax = plt.subplots()
        ax.bar(range(10), probabilities, tick_label=[str(i) for i in range(10)], color='orange')
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.0)
        st.pyplot(fig)
    else:
        st.warning("먼저 숫자를 캔버스에 그려주세요!")
