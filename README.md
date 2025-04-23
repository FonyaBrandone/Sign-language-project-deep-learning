# Sign-language-project-deep-learning
Real-Time Sign Language Recognition and Speech Transcription using Deep Learning

## Introduction
### Course: (17-644) Applied Deep Learning | Carnegie Mellon University
This project addresses the challenges faced by visually impaired communicating with their environment by designing and implementing a real-time assistive system that leverages deep learning techniques to translate static sign language gestures into both textual and auditory outputs. Specifically, the system employs a Convolutional Neural Network (CNN) trained on the Sign Language MNIST dataset to classify American Sign Language (ASL) hand signs from live video input. The classified gestures are then converted into spoken words using a text-to-speech engine, enabling real-time audio feedback.

The system was developed using Python and integrates several key libraries and frameworks. OpenCV facilitates video capture from a webcam, MediaPipe is used for hand detection and landmark tracking, and pyttsx3 provides offline speech synthesis capabilities. Together, these components form a pipeline capable of capturing, classifying, and vocalizing hand gestures in real time using only a standard computing device and webcam.

## Dataset
Download the <a href='https://www.kaggle.com/datasets/datamunge/sign-language-mnist'>Sign Language MNIST</a> dataset on kaggle, create a folder named _"data"_ in the project folder and place the _"sign_mnist_train.csv"_ and _"sign_mnist_test"_ datasets within.

## Building and running
Run the cells in the notebook _sign_language_notebook.ipynb_ to build and train the model then run the program _sign_language_program.py_

## Disclaimer
_Using any part or entirety of this work as a requirement for any course or project requirement is considered cheating._
