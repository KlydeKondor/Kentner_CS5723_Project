# Kyle Kentner
# Dr. Doug Heisterkamp
# CS-5723: Artificial Intelligence
# 2 May 2022
# Image Sharpening via Reinforcement Learning

# REQUIREMENTS
This project requires the Numpy, OpenCV, OpenAI Gym, Tensorflow, and Keras libraries.
Numpy and OpenCV are required for array/image operations.
Tensorflow, OpenAI Gym, and Keras are required for the reinforcement-learning implemenation.
Each of these can be installed in the following manner:

	pip install --user numpy

	pip install --user opencv-python

	pip install --user tensorflow==2.9.0rc1

	pip install --user gym

	pip install --user keras

	pip install --user keras-rl2

NOTE: The '--user' option is included because on my machine, some of these libraries would not
install without it. Further, for clarity, the Tensorflow installation is 'r-c-ONE', whereas the second
Keras installation is 'r-L-2'.

The Python version used in the project was 3.10.

# RUNNING
To train the agent, the following command may be entered via a command window in the same
directory as the project:

	python main.py blurryImage.png sharpImage.png

The 'blurryImage' file is the intentionally blurred image, and the 'sharpImage' file is the target image
for the training. Neither image name should contain spaces. Full file paths are required if the images
only exist outside of the local directory.

To test the agent, the following command may be entered:

	python main.py blurryImage.png

This will perform the sharpening procedure without a target image.

NOTE: Attempts to suppress the Tensorflow deprecation warnings were unsuccessful, but the program will
run as intended regardless.

# OUTPUT
The training procedure will yield a file named 'IMG_Train.png', and the testing will yield 'IMG_Test.png'.

# OTHER NOTES
Input images for training should be kept small, as the training process takes an extremely long time to terminate.