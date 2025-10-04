import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ConvertImageDtype

import torch.nn.functional as F  
import torchvision.transforms.functional as F_tx
import torchvision.io as tv_io
import tempfile

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

# ASL Sign Language MNIST 데이터셋 클래스 레이블 (29개가 아닌, J,Z 제외된 24개)
class_names = [
    "A","B","C","D","E","F","G","H","I",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y"
]

# 1) 모델 로드 (CPU/GPU 자동 선택)
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.load('asl_model_no_state.pth', map_location=device)
    model = torch.load('asl_model_no_state.pth').to(device)
    # model = torch.load('asl_nvidia_model.pth').to(device)
    model.eval()
    return model, device

model, device = load_model()

# 2) 전처리 파이프라인 정의
IMG_WIDTH, IMG_HEIGHT = 28, 28
preprocess = transforms.Compose([
    # uint8 [0,255] → float32 [0,1]
    ConvertImageDtype(torch.float32), 
    # transforms.ToDtype(torch.float32, scale=True),
    # 크기 조정
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    # 흑백 처리 (채널 1개)
    transforms.Grayscale(),
])

# 3) Streamlit UI
st.title("ASL Gesture Recognition")
st.write("ASL(수화) 이미지를 업로드하면 숫자 클래스를 예측해 드립니다.")

uploaded_file = st.file_uploader("이미지 파일 업로드 (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    # 업로드된 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
        print(tmp_path)

    # 이미지 파일을 읽어와 Tensor로 변환(GRAY 모드 → shape [1,H,W], dtype=uint8)
    image_tensor = tv_io.read_image(tmp_path, tv_io.ImageReadMode.GRAY) 
    image_tensor_rgb = tv_io.read_image(tmp_path, tv_io.ImageReadMode.RGB)
    
    st.image(
        F_tx.to_pil_image(image_tensor_rgb),
        caption="Input Image",
        use_container_width=True
    )
    
     # 전처리
    processed = preprocess(image_tensor)               # [1,28,28], float32, [0,1]
        

    # 배치 차원 추가 및 GPU 전송
    input_tensor = processed.unsqueeze(0).to(device)   # [1,1,28,28]
    if st.button("Predict"):
    
        # 추론
        with torch.no_grad():
            logits = model(input_tensor)
            probs  = F.softmax(logits, dim=1)[0]
        
        # 상위 3개 클래스와 확률
        topk = torch.topk(probs, k=3)
        top_probs = topk.values.tolist()           # [prob1, prob2, prob3]
        top_idxs  = topk.indices.tolist()          # [idx1, idx2, idx3]
        
        # 인덱스를 레이블로 변환
        top_labels = [class_names[i] for i in top_idxs]
                
        # 결과 출력
        st.success(f"예측 결과: 클래스 {top_labels[0]} (확률 {top_probs[0]:.4f})")
        st.write("— Top-3 클래스별 확률 —")
        for label, prob in zip(top_labels, top_probs):
            st.write(f"클래스 {label}: {prob:.4f}")