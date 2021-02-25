# Pneumonia Detection
Pneumonia Detection using machine learning.

**Training was done in colab:**

[![Training In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uJn7TpIHWkjiDGX6r8rlgm32jhzYhcLi?usp=sharing)

<br>

**DEMO:**

![gif](https://github.com/wilhelmberghammer/pneumonia_detection/blob/main/resources/web.gif?raw=true)

**Result (Confusion Matrix):**

![confusion matrix](https://github.com/wilhelmberghammer/pneumonia_detection/blob/main/resources/confusion_matrix.png?raw=true)


## Data
[I uploaded my dataset to kaggle](https://www.kaggle.com/wilhelmberghammer/chest-xrays-pneumonia-detection)
I used a modified version of [this](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from kaggle. Instead of NORMAL and PNEUMONIA I split the PNEUMONIA dataset to BACTERIAL PNUEMONIA and VIRAL PNEUMONIA. This way the data is more evenly distributed and I can distinguish between viral and bacterial pneumonia. I also combined the validation dataset with the test dataset because the validation dataset only had 8 images per class.

This is the resulting distribution:

![data distribution](https://github.com/wilhelmberghammer/pneumonia_detection/blob/main/resources/labels.png?raw=true)


### **Processing and Augmentation**
I resized the images to 150x150 and because some images already were grayscale I also transformed all the images to grayscale.

Additionaly I applied the following transformations/augmentations on the **training data**:
```python
transforms.Resize((150, 150)),
transforms.Grayscale(),
transforms.ToTensor(),
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
transforms.RandomRotation(45)
```

and those transformations on the **test data**:
```python
transforms.Resize((150, 150)),
transforms.Grayscale(),
transforms.ToTensor(),
```


This is the resulting data:

![sample images](https://github.com/wilhelmberghammer/pneumonia_detection/blob/main/resources/x_rays.png?raw=true)


**I also used one-hot encoding for the labels!**


<br>

---
## Model

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 16, 148, 148]             160
              ReLU-2         [-1, 16, 148, 148]               0
       BatchNorm2d-3         [-1, 16, 148, 148]              32
            Conv2d-4         [-1, 16, 146, 146]           2,320
              ReLU-5         [-1, 16, 146, 146]               0
       BatchNorm2d-6         [-1, 16, 146, 146]              32
         MaxPool2d-7           [-1, 16, 73, 73]               0
            Conv2d-8           [-1, 32, 71, 71]           4,640
              ReLU-9           [-1, 32, 71, 71]               0
      BatchNorm2d-10           [-1, 32, 71, 71]              64
           Conv2d-11           [-1, 32, 69, 69]           9,248
             ReLU-12           [-1, 32, 69, 69]               0
      BatchNorm2d-13           [-1, 32, 69, 69]              64
        MaxPool2d-14           [-1, 32, 34, 34]               0
           Conv2d-15           [-1, 64, 32, 32]          18,496
             ReLU-16           [-1, 64, 32, 32]               0
      BatchNorm2d-17           [-1, 64, 32, 32]             128
           Conv2d-18           [-1, 64, 30, 30]          36,928
             ReLU-19           [-1, 64, 30, 30]               0
      BatchNorm2d-20           [-1, 64, 30, 30]             128
        MaxPool2d-21           [-1, 64, 15, 15]               0
           Conv2d-22          [-1, 128, 13, 13]          73,856
             ReLU-23          [-1, 128, 13, 13]               0
      BatchNorm2d-24          [-1, 128, 13, 13]             256
           Conv2d-25          [-1, 128, 11, 11]         147,584
             ReLU-26          [-1, 128, 11, 11]               0
      BatchNorm2d-27          [-1, 128, 11, 11]             256
        MaxPool2d-28            [-1, 128, 5, 5]               0
          Flatten-29                 [-1, 3200]               0
           Linear-30                 [-1, 4096]      13,111,296
             ReLU-31                 [-1, 4096]               0
          Dropout-32                 [-1, 4096]               0
           Linear-33                 [-1, 4096]      16,781,312
             ReLU-34                 [-1, 4096]               0
          Dropout-35                 [-1, 4096]               0
           Linear-36                    [-1, 3]          12,291
          Softmax-37                    [-1, 3]               0
================================================================
Total params: 30,199,091
Trainable params: 30,199,091
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.09
Forward/backward pass size (MB): 27.95
Params size (MB): 115.20
Estimated Total Size (MB): 143.24
----------------------------------------------------------------
```


---
## Visualization using [Streamlit](https://www.streamlit.io/)
The webapp is not hosted because the model is too large. I'd have to host it on a server. This is just to visualize.