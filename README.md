# Pneumonia Detection
Pneumonia Detection using machine learning.

I used a modified version of [this](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from kaggle. Instead of NORMAL and PNEUMONIA I split the PNEUMONIA dataset to BACTERIAL PNUEMONIA and VIRAL PNEUMONIA. This way the data is more evenly distributed and I can distinguish between viral and bacterial pneumonia. I also combined the validation dataset with the test dataset because the validation dataset only had 8 images per class.

This is the resulting distribution:

![data distribution](https://github.com/wilhelm/pneumonia_detection/blob/main/resources/labels.png?raw=true)

