import streamlit as st
import cv2 as cv

placeholder = st.empty()

inputImage = st.file_uploader("Choose an image")

if inputImage is not None:
    enhancementType = st.radio(
        "Choose the enhancement you want to apply to the uploaded image?",
        ('Histogram Equalization', 'Color Denoising', 'Improve Contrast', 'Increase Brightness'))

    with placeholder:
        st.image(inputImage, caption='Input')
    
    inputImage = inputImage.read()
    with open("input.jpeg", 'wb') as f: 
        f.write(inputImage)
    
    inputImage = cv.imread('input.jpeg')
    
    resizedImage = cv.resize(inputImage, (780, 540), interpolation=cv.INTER_AREA)
    
    if enhancementType == 'Histogram Equalization':
        grayScaledImage = cv.cvtColor(resizedImage, cv.COLOR_BGR2GRAY)
        outputImage = cv.equalizeHist(grayScaledImage)

    elif enhancementType == 'Color Denoising':
        resizedImage = cv.cvtColor(resizedImage, cv.COLOR_BGR2RGB)
        
        outputImage = cv.fastNlMeansDenoisingColored(resizedImage,None,10,10,7,21)
        
    elif enhancementType == 'Improve Contrast':
        outputImage = cv.convertScaleAbs(resizedImage, alpha=2.0, beta=0)
    
    elif enhancementType == 'Increase Brightness':
        outputImage = cv.convertScaleAbs(resizedImage, alpha=1.0, beta=70)

    with placeholder:
        st.image(outputImage, caption='Output')