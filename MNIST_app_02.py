import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas

# -------------------------------
# 디바이스 설정
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 모델 로딩
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
# 전처리 정의 (학습 시 동일하게)
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

# -------------------------------
# 중심 정렬 함수 (비율 유지 + 최소 패딩)
# -------------------------------
def center_image(pil_img, size=(28, 28)):
    img = pil_img.convert("L")
    img = img.point(lambda x: 0 if x < 128 else 255, '1')
    img = ImageOps.invert(img)

    bbox = img.getbbox()
    if not bbox:
        return img.resize(size, Image.Resampling.LANCZOS)

    img = img.crop(bbox)
    img.thumbnail((22, 22), Image.Resampling.LANCZOS)

    canvas = Image.new("L", size, 0)
    paste_x = (size[0] - img.size[0]) // 2
    paste_y = (size[1] - img.size[1]) // 2
    canvas.paste(img, (paste_x, paste_y))
    return canvas

# -------------------------------
# Streamlit UI
# -------------------------------
# st.title("MNIST Handwritten Digit Recognizer")
st.subheader("✍️ MNIST Handwritten Digit Recognizer")
st.write("숫자를 흰색으로 그리고 [예측하기] 버튼을 눌러보세요.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=196,  # 28x7
    width=196,
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

        # 전처리 후 텐서화
        input_tensor = transform(centered).unsqueeze(0).to(device)

        # 디버깅: 입력 이미지 확인
        st.image(input_tensor[0][0].cpu().numpy(), caption="입력 이미지 (28x28)", width=100, clamp=True)

        # 예측
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
            pred = int(np.argmax(probabilities))

        # 결과 출력
        st.success(f"예측된 숫자: **{pred}**")

        st.subheader("클래스별 확률 (%)")
        for i, prob in enumerate(probabilities):
            st.write(f"{i} : **{prob * 100:.2f}%**")
    else:
        st.warning("먼저 숫자를 그려주세요!")
