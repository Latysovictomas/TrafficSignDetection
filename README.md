# TrafficSignDetection
The main goal of this project was to develop traffic sign detection system on Android device which would inform drivers about traffic signs ahead. The German Traffic Sign Detection Benchmark datasest was chosen to train SSD architecture-based mobileNet v2 model 300x300. 

## Structure
The project consists of two parts: training the neural network, and making an Android application.

Python directory has two main scripts:
1. data_augmentation.ipynb to split the dataset and make tf.record files necessary for neural network input.
2. Object_detection_mobile.ipynb is used in Google Colab to train the detector.

Android has 4 files:
1. Classifier.java to describe what was recognized.
2. FPSCounter.java to count the fps during inference.
3. MainActivity.java to run all process on camera frame and allows to change settings.
4. TFLiteObjectDetectionAPIModel.java to output prediction results.

## Usage
1. Open a project on Android studio
2. Connect your device
3. Run the project

## Results
The application achieves 0.68 mAP and 10 FPS on Samsung A7 (2018).

## Additional notes
All data files has been removed to save space in the repository. Dataset has to be downloaded, everything else can be generated.
