import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
import pandas as pd
from model import CNN

st.set_option('deprecation.showfileUploaderEncoding', False)



st.markdown('# Diagnosing Lung X-Rays 🩺')
st.markdown("This model uses a convolutional neural network to classify images of lung x-rays.")
st.markdown("It has been **trained on a Tesla V100-SXM2 GPU using ~4500 different lung x-rays**.")
st.markdown("The model achieved an accuracy of ~91% on a testset of 600 lung x-rays")
st.markdown("*Disclaimer: For educational porpuses only*")

st.sidebar.write('''
                # Data Acknowledgements
                [Open Source Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)
                ## Institutions
                University of California San Diego
                ## License
                [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
                ''')


def img_to_torch(pil_image):
	img = pil_image.convert('L')
	x = torchvision.transforms.functional.to_tensor(img)
	x = torchvision.transforms.functional.resize(x, [150, 150])
	x.unsqueeze_(0)
	return x

def predict(image, model):
	x = img_to_torch(image)
	pred = model(x)
	pred = pred.detach().numpy()

	df = pd.DataFrame(data=pred[0], index=['Bacterial', 'Normal', 'Viral'], columns=['confidence'])

	st.write(f'''### 🧫 Confidence - Bacterial:  **{np.round(pred[0][0]*100, 3)}%**''')
	st.write(f'''### 🦠 Confidence - Viral: **{np.round(pred[0][2]*100, 3)}%**''')
	st.write(f'''### 👍 Confidence - Normal: **{np.round(pred[0][1]*100, 3)}%**''')
	st.write('')
	st.bar_chart(df)

PATH_TO_MODEL = 'model.py'
model = torch.load(PATH_TO_MODEL)
model.eval()

uploaded_file = st.file_uploader('Upload image...', type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption='This x-ray will be diagnosed...', use_column_width=True)

	if st.button('Predict 🧠'):
		predict(image, model)
