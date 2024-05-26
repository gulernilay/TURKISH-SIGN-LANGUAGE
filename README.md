# Turkish Sign Language Recognition App

# About the Project
This project is an innovative initiative designed to bridge the communication gap for the hearing-impaired community through technology. By converting sign language gestures and audio signals into text, this mobile application stands as a beacon of accessibility and inclusion. Utilizing state-of-the-art convolutional neural networks, specifically MobileNetv2, the app interprets complex sign language gestures in real-time, rendering them into easily understandable text. Our application is not only a tool but a companion that enhances daily interactions for those challenged by hearing impairments, enabling them to engage more fully with the world around them.

# Motivation
The inspiration for this project is deeply rooted in our commitment to social responsibility and technological innovation. With over 466 million people worldwide experiencing significant hearing impairments, the need for effective communication tools is undeniable. These individuals often encounter substantial barriers in both personal interactions and public engagements, limiting their access to services and opportunities that many take for granted. By integrating cutting-edge AI and mobile technology, we aim to dismantle these barriers, offering a transformative tool that empowers the hearing-impaired community. This project goes beyond mere technological advancement; it's about creating a more inclusive society where communication barriers are obliterated and where everyone has the chance to thrive.

# Datasets
We utilized three main datasets for training our models:
 - Kaggle's Turkish Sign Language Dataset : https://www.kaggle.com/datasets/berkaykocaoglu/tr-sign-language 
 - A manually photographed dataset of sign language letters created by our team.
 - Combined Dataset - A fusion of Kaggle's original dataset and our self-created dataset :  https://www.kaggle.com/datasets/nilaygler/signdet 

# Technologies Used
MobileNet: For efficient, real-time sign language gesture recognition.

Kotlin: Used for developing the mobile application.

Deep Learning Libraries: PyTorch and TensorFlow for training convolutional neural networks.

# Features
-Sign Language Classification: Converts gestures into corresponding text.

-Speech-to-Text Conversion: Allows for verbal communication interpretation.

-Educational Tools: Includes quizzes and educational photos to aid learning.

-Real-Time Recognition: Offers immediate gesture-to-text conversion.

-Quiz Screen : To educate users in order to learn sign language.

#  Model Training
We trained our models using various augmentation techniques such as rotation, scaling, and resizing to improve data quality. The models, including AlexNet, ResNet-18, ResNet-50, MobileNetv1, and MobileNetv2, underwent rigorous training to ensure high accuracy and performance.

#  Application Workflow
Camera Module : 

    Hand Detection: Utilizes a pre-trained model to detect hands in video frames.
    
    Gesture Recognition: Once detected, the hand region bounded with a box  is cropped and analyzed by the MobileNetv2 model for gesture recognition.
    
    Output Generation: Converts recognized gestures into text displayed within the app.
    
Quiz Module :    

    25 questions are waiting to be answered by users.
    
Speech Module : 
    Google API is used for converting speech into text 

