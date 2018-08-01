# Proscia Tech challenge
---
## Minimum Software Requirements
* #### Python 3
* #### OpenCV-3.1.0
* #### sklearn-0.19.1
* #### numpy-1.14.0
* #### matplotlib-2.0.2
* #### OS-Windows 10


## Files submitted
---
#### 1.pathology_A : Contains all pathology a type images
#### 2.pathology_B : Contains all pathology b type images
#### 3.tech_challenge:-->identify.py : The main file used for classification task
#### 4. writeup: The short report of observations
#### 5.pathology_A_HSV_analysis: HSV figures of pathology A generated during the exploratory process
#### 6.pathology_B_HSVanalysis: HSV figures of pathology B generated during he exploratory process

# Files generated
---
#### 1.svm_model_poly_trained.sav : The training model generated during training if selected

## Installation and Execution
---
### To install and run the program:

 ##### 1. Unzip the contents
 ##### 2. cd to proscia/tech_challenge
 ##### 3. Run python idenify_slide.py [followed by the path of the image you want to test] using command line for example : python identify_slide.py ../pathology_A/10.png
 ##### 4. The command prompt will ask user to either enter 1 for classifying using ML or enter 0 for classifying using computer vision methods5. Output will be either A or B
 


