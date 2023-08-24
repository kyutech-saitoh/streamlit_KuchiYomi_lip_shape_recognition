import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import japanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm

st.title("Streamlit App: Mouth shape recognition")
st.write("Kyutech, Saitoh-lab")
st.markdown("---")

target_person_id = st.selectbox("Target person", ("P001", "P002", "P003"))
st.write("You selected:", target_person_id)
st.markdown("---")

def preprocess(image_path, transform):
    #########################################
    # Plese write extracting lipROI process #
    #########################################
    
    image = Image.open(image_path)
    image = transform(image)  # PIL
    C, H, W = image.shape
    image = image.reshape(1, C, H, W)
    
    return image

def make_graph(values):
    Y = np.arange(6)
    #Y = ['N', 'a', 'i', 'u', 'e', 'o']
    X = values
    sm = matplotlib.cm.ScalarMappable(cmap='jet')
    fig, ax = plt.subplots()
    # 横棒グラフ
    ax.barh(Y, X, color=sm.to_rgba(X))
    ax.set_xlim(0, 1)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(['閉', 'あ', 'い', 'う', 'え', 'お'] , size=14)

    st.pyplot(fig)
    
def test(model, crop_image):

    # モデルを評価モードにする
    model.eval()

    with torch.no_grad():
        # 予測
        outputs = model(crop_image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        #st.write(probabilities)

        # obtain first six classes
        probabilities6 = probabilities[0:6]
        st.write(probabilities6)

        make_graph(probabilities6)
        
        # 予測結果をクラス番号に変換
        _, predicted = torch.max(outputs, 1)

    return predicted


def main():
    # data transform
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])

    # vowel dict
    idxtovowel = {0: '閉口', 1: 'あ', 2: 'い', 3: 'う', 4: 'え', 5: 'お'}
    # training model path
    model_path = 'model/model_mobilenetv2.pth'
    # load model
    model = torch.load(model_path)
    # load device : cpu
    device = torch.device("cpu")
    model.to(device)

    image_data = st.file_uploader("Upload file", ['jpg','png'])

    input_image = None
    
    if image_data is not None:
        image = Image.open(image_data)
        img_array = np.array(image)
        st.image(img_array, caption = 'uploaded image', use_column_width = None)
        #input_image = pil2cv(image) 

        # preprocess
        crop_image = preprocess(image_data, transform)
        # predict
        predict = test(model, crop_image)

        st.write(predict)
        st.write(predict.item())
        st.write(idxtovowel[predict.item()])

if __name__ == '__main__':
    main()
