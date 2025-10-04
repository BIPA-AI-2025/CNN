import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

# Streamlit 설정
st.set_page_config(page_title="Handwritten Digit Classifier", layout="centered")
st.title("Handwritten Digit Classifier (0~9)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의
model =  nn.Sequential(
            nn.Flatten(),       # (B, 28, 28) → (B, 784)
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
).to(device)


# 모델 로드
model.load_state_dict(torch.load("./model/mnist_linear_model.pth", map_location=device))
model.eval()

# 입력 이미지 처리 함수
def preprocess_image(image):
    image = ImageOps.grayscale(image)     # 흑백 변환
    image = image.resize((28, 28))        # MNIST 크기
    transform = transforms.ToTensor()     # [0, 1], (1, 28, 28)
    image = transform(image).unsqueeze(0) # 배치 추가 → (1, 1, 28, 28)
    return image.to(device)

# 이미지 업로드
uploaded_file = st.file_uploader("28x28 손글씨 이미지(.png/.jpg) 업로드", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", width=150)

    # 버튼 생성
    if st.button("예측하기"):
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).squeeze()[pred].item()

        st.markdown(f"예측 숫자: **{pred}**")
        st.markdown(f"정확도: `{confidence * 100:.2f}%`")
else:
    st.info("왼쪽에 손글씨 숫자 이미지를 업로드하고 버튼을 클릭하세요.")


st.markdown("---")
st.caption("Powered by ChatGPT-5")