import streamlit as st
import cv2 as cv

placeholder = st.empty()

# Initial manipulation parameters
filterStrength = 10
alpha = 0
beta = 0

inputImage = st.sidebar.file_uploader("Upload image")

if inputImage is not None:
    
    # writing uploaded image temporarily
    inputImage = inputImage.read()
    with open("input.jpeg", 'wb') as f: 
        f.write(inputImage)
    
    enhancementType = st.sidebar.radio(
        "Choose the enhancement you want to apply to the uploaded image?",
        ('None', 'Histogram Equalization', 'Others'))


    filterStrength = st.sidebar.slider('Denoising Filter Strength', min_value=0, max_value=10, value=0)
    alpha = st.sidebar.slider('Contrast', min_value=1.0, max_value=3.0, value=1.0)
    beta = st.sidebar.slider('Brightness', min_value=0, max_value=100, value=0)
        
    inputImage = cv.imread('input.jpeg')
    with placeholder:
       
        st.image(cv.resize(cv.cvtColor(inputImage, cv.COLOR_BGR2RGB), (780, 540), interpolation=cv.INTER_AREA), caption='Input')

    outputImage = inputImage    
    
    if enhancementType != 'None':
        
        outputImage = cv.resize(inputImage, (780, 540), interpolation=cv.INTER_AREA)
        outputImage = cv.cvtColor(outputImage, cv.COLOR_BGR2RGB)
        
        if enhancementType == 'Histogram Equalization':
            grayScaledImage = cv.cvtColor(outputImage, cv.COLOR_RGB2GRAY)
            outputImage = cv.equalizeHist(grayScaledImage)

        elif enhancementType == 'Others':
            outputImage = cv.fastNlMeansDenoisingColored(outputImage,None,filterStrength,10,7,21)
            outputImage = cv.convertScaleAbs(outputImage, alpha=alpha, beta=beta)
        
        with placeholder:
            st.image(outputImage, caption='Output')