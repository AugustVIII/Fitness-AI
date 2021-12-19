# Fizruk - fitness application for yoga and workout

### Fizruk was created in order to practice yoga and workout in real time using computer vision
- Fizruk supports 2 modes: yoga and workout
- There are 3 exercises in each program

    Yoga: T-Pose, Warrior Pose, Tree Pose
    
    Workout: Arms, Legs, Gym
    
- Fizruk was created using [MediaPipe](https://github.com/google/mediapipe), [OpenCV](https://github.com/opencv/opencv), [PyInstaller](https://github.com/pyinstaller/pyinstaller).


### Example of application operation
![process](static/WarriorPose.gif)

### Vitrual hands control
![virtual_control](static/hands_control.gif)

### The process of pose recognition consists of following steps:
- Recieve video stream using OpenCV. 
- Video frames are passed to MediaPipe Pose model that detects pose, adds landmarks (33 landmarks per body) and records their coordinates.
- Landmarks coordinates are extracted, organized and passed to the model for prediction.
- The visualization of model prediction is implemented in the top middle corner of the application screen (see pictures above). 
- Detected poses and displayed on top of the screen (see pictures above). 


#### This project was completed in 10 days by:
- https://github.com/AugustVIII
- https://github.com/raulgad
- https://github.com/samot-samoe
