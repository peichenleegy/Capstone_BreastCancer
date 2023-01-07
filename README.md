# Enhance spectrally encoded confocal microscopy(SECM) on breast cancer diagnosis through deep learning models created by 124 images from 23 patients 

### Background

1. Breast Cancer
2. The Main Step in Breast Cancer Diagnosis Process
3. Spectrally encoded confocal microscopy (SECM) images
4. Models for Classifying SECM images

### Significance

Adoption of new technology to a diagnosis of breast margin assessment can decrease morbidity and mortality. 
Neural network models can observe details in SECM images to aid pathology diagnosis via reducing time on getting pathology
With computer learning algorithms to classify the images from SECM, patients would get high-quality medical reports. 

### Innovation

Many pre-trained models with many parameters can be applied to classify histopathology images via less than 124 training images. 
For the data augmentation, we newly import H&E images while retraining the models
This is the first-time attempt for building a neural network model to classify SECM images to enhance the workflow of diagnosing breast margin pathology tissues. 

### Question

Any methods of doing image pre-processing can be applied to SECM images?
Retrain models by using H&E stain images would benefit the model's ability to differentiate SECM images?
Based on analysis results of the classification systems and evaluated metrics, what model can help clinical applications? 

### Data

Schlachter SC, K.D., Gora MJ, et al, Spectrally encoded confocal microscopy of esophageal tissues at 100 kHz line rate. Biomedical optics express, 2013. 4,9: p. 1636-1645.



### Research Procedures

I tried some image pre-processing techniques to let the images become more clear.
Next, I select images of certain patients having adipose, fibrous, and malignant tissue type to be validation datasets
After that, the models will be test to differciate images from benign/malignant or reviewed by human/not reviewed needs. 
Considering the data size and previous findings, I modified the workflow and tried to apply leave-one-out cross validation to figure out the best performance of the model that we could provide

### Results

For each iteration I got a metrix like this, and then I did calculations to get average prediction and average recall based on the sum of references, predictions and correct labels. 
Finally, I have got the best performance at the fifth epochs are the average precision is 0.85, the average recall is 0.998. Specifically, the model showed the 85 corrected predictions on 100 predictions, and there are 85 other tissues selected from overall 86 images of the other tissue type. 

### Findings

1. Images become clear for human eyes are not always good for deep learning training 
2. Importing H&E images for build models to recognize histopathology images  
3. ResNet152V2 differentiating adipose, fibrous, and other tissues could approaching pathologist’s ability.
4. ResNet152V2 had 0.85 of the average precision and 0.998 of the average recall for distinguishing images without human-reviewed needed.

