import cv2
import mediapipe as mp
import math
import json
import os
import pickle
import pandas as pd
import numpy as np

window_width = 960 #1024
window_height = 540 #768

dirname = os.path.dirname(__file__)

class poseDetector():
 
    def __init__(self, static_image_mode=False, model_complexity=0, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
 
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, 
                                    self.smooth_landmarks, self.enable_segmentation, 
                                    self.smooth_segmentation, self.min_detection_confidence, 
                                    self.min_tracking_confidence)
        
        # # Init poses dictionary from json file
        # with open(os.path.join(dirname, 'poses.json'), 'r') as fp:
        #     self.poses = json.load(fp)
        with open(os.path.join(dirname,'yoga_angles.pkl'), 'rb') as f:
            self.model = pickle.load(f)

    def drawCircle(img, x, y, clr = (0,0,255)):
        cv2.circle(img, (x, y), 10, clr, cv2.FILLED)
        cv2.circle(img, (x, y), 15, clr, 2)

    def drawPoints(self, img, landmarks, points=[]):
        white_clr = (0,0,0)
        points.sort()
        for p_idx, point in enumerate(points):
            # Draw point
            x1, y1, _ = landmarks[point]
            self.drawCircle(img, x1, y1)
            # Draw line and next point
            next_p_idx = p_idx + 1
            if next_p_idx < len(points):
                next_point = points[next_p_idx]
                x2, y2, _ = landmarks[next_point]
                cv2.line(img, (x1, y1), (x2, y2), white_clr, 3)
                self.drawCircle(img, x2, y2)

    # def classify_yoga(self, rel_lmks, show_correct=False):
    #     label = 'Unknown Pose'
    #     # Set gap between current and correct pose for softer recognition
    #     gap = 0.1 
    #     # Check if current points is equal to one of the pose from poses.
    #     if self.poses:
    #         for pose_name, pose in self.poses.items():
    #             # Init correct poses's points
    #             start_p_idx = -1
    #             points = []

    #             for key, val in pose.items():
    #                 # Set index of point relative to which we calculated the distances to other points
    #                 if key == 'START_POINT':
    #                     start_p_idx = val['index']
    #                 else:
    #                     # Add relative points of the pose
    #                     points.append([val['x'], val['y']])
                
    #             # # Reinit current points to be same format as correct points of the pose
    #             # curr_points = []
    #             # abs_curr_points = []
                
    #             for p_idx, point in enumerate(points):
    #                 corr_x = point[0]
    #                 corr_y = point[1]
    #                 # Change values only for points that was used for recognition (that not zeros in pose's points)
    #                 if abs(corr_x) > 0 or abs(corr_y) > 0:
    #                     # Calculate current point that relative to start point of the pose.
                        
    #                     # ДОЛЖНА БЫТЬ КОРРЕКЦИЯ ПО Z (ТАК КАК МЕНЯЕТСЯ ДЛИНА МЕЖДУ START_POINT И ТОЧКОЙ, КОГДА ИДЕМ В ГЛУБИНУ)
    #                     # ДОЛЖНА БЫТЬ КОРРЕКЦИЯ ПО ПРОПОРЦИЯМ ТЕЛА (ТАК КАК, НАПРИМЕР, РАССТОЯНИЕ МЕЖДУ ТОЧКАМИ МОЖЕТ ОТЛИЧАТЬСЯ ИЗЗА РАЗНИЦЫ В ДЛИНЕ РУК)
                        
    #                     curr_x = rel_lmks[start_p_idx].x - rel_lmks[p_idx].x
    #                     curr_y = rel_lmks[start_p_idx].y - rel_lmks[p_idx].y
                        
    #                     # curr_points.append([lmks[start_p].x - point[0], lmks[start_p].y - point[1]])

    #                     # Check if current point between min and max values of pose's point
    #                     diff_x = curr_x - corr_x
    #                     diff_y = curr_y - corr_y
    #                     # Set correct pose's name to label if we detect it

    #                     # ДОЛЖНЫ ВСЕ ТОЧКИ СОВПАСТЬ, А НЕ ОДНА

    #                     if abs(diff_x) <= gap and abs(diff_y) <= gap:


    #                         label = pose_name.capitalize() + ' Pose'
    #                     else:
    #                         # Draw correction if in current points we have at least one point that not in correct point's range.
    #                         if show_correct:
    #                             # Calculate absolute correct point's values
    #                             abs_corr_x = (rel_lmks[p_idx].x - diff_x) * window_width
    #                             abs_corr_y = (rel_lmks[p_idx].y - diff_y) * window_height

    #             #     else:
    #             #         curr_points.append(0.0, 0.0)
                        


    #             # curr_x_rel = rel_lmks[start_p].x - results.pose_landmarks.landmark[i.value].x
    #             # curr_y_rel = rel_lmks[start_p].y - results.pose_landmarks.landmark[i.value].y

    #             # # For better perfomance we will use numpy to calculate distance between point's vectors
                
    #             # # curr_points = np.array(lmks)
    #             # corr_points = np.array(points)

    #             # # Check if current points between min and max values of pose's points
    #             # curr_in_corr = np.absolute(curr_points - corr_points) <= gap
    #             # # Draw correct points if in current points we have at least one point that not in correct point's range.
    #             # if np.any(curr_in_corr == False):
    #             #     if show_correct:
    #             #         continue
    #             # else:
    #             #     # Set label to correct pose's name
    #             #     label = pose_name.capitalize() + ' Pose'

    #     else:
    #         print('self.poses wasnt read -- in PoseModule')

    #     return label

    def classifyPose(self):
            

            results = self.results.pose_landmarks

            left_elbow_angle = self.findAngle(self.mpPose.PoseLandmark.LEFT_SHOULDER.value, 
                                            self.mpPose.PoseLandmark.LEFT_ELBOW.value, 
                                            self.mpPose.PoseLandmark.LEFT_WRIST.value, 
                                            )

            # Get the angle between the right shoulder, elbow and wrist points. 
            right_elbow_angle = self.findAngle(self.mpPose.PoseLandmark.RIGHT_SHOULDER.value,
                                            self.mpPose.PoseLandmark.RIGHT_ELBOW.value,
                                            self.mpPose.PoseLandmark.RIGHT_WRIST.value, 
                                            )   
            
            # Get the angle between the left elbow, shoulder and hip points. 
            left_shoulder_angle = self.findAngle(self.mpPose.PoseLandmark.LEFT_ELBOW.value,
                                                self.mpPose.PoseLandmark.LEFT_SHOULDER.value,
                                                self.mpPose.PoseLandmark.LEFT_HIP.value, 
                                            )

            # Get the angle between the right hip, shoulder and elbow points. 
            right_shoulder_angle = self.findAngle(self.mpPose.PoseLandmark.RIGHT_HIP.value,
                                                self.mpPose.PoseLandmark.RIGHT_SHOULDER.value,
                                                self.mpPose.PoseLandmark.RIGHT_ELBOW.value, 
                                            )

            # Get the angle between the left hip, knee and ankle points. 
            left_knee_angle = self.findAngle(self.mpPose.PoseLandmark.LEFT_HIP.value,
                                            self.mpPose.PoseLandmark.LEFT_KNEE.value,
                                            self.mpPose.PoseLandmark.LEFT_ANKLE.value, 
                                            )

            right_knee_angle = self.findAngle(self.mpPose.PoseLandmark.RIGHT_HIP.value,
                                            self.mpPose.PoseLandmark.RIGHT_KNEE.value,
                                            self.mpPose.PoseLandmark.RIGHT_ANKLE.value, 
                                            )

                            # Get the angele between the left shoulder ,left hip and left knee 
            left_hip_angle = self.findAngle(self.mpPose.PoseLandmark.LEFT_SHOULDER.value,
                                            self.mpPose.PoseLandmark.LEFT_HIP.value,
                                            self.mpPose.PoseLandmark.LEFT_KNEE.value)

                    # Get the angle between the right shoulder, right hip and right knee
            right_hip_angle = self.findAngle(self.mpPose.PoseLandmark.RIGHT_SHOULDER.value,
                                            self.mpPose.PoseLandmark.RIGHT_HIP.value,
                                            self.mpPose.PoseLandmark.RIGHT_KNEE.value)
                    
                

            
            if results:

                # frame = output_image
                # pose = results.pose_landmarks.landmark
                angle_row = [left_elbow_angle,right_elbow_angle,left_shoulder_angle,right_shoulder_angle\
                        ,left_knee_angle,right_knee_angle,left_hip_angle,right_hip_angle]


                # pose_row.insert(0,class_name)

                # with open('coords.csv', mode='a', newline='') as f:
                #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                #     csv_writer.writerow(pose_row)
                label = 'Unknown Pose'
                color = (0,0,255)

                X = pd.DataFrame([angle_row])
                yoga_angles_class = self.model.predict(X)[0]
                yoga_angles_prob = self.model.predict_proba(X)[0] 

                if yoga_angles_prob[np.argmax(yoga_angles_prob)] > 0.90:
                    label = yoga_angles_class
                    if yoga_angles_class != 'Unknown class':
                        color = (0, 255, 0) 
            

                # elif yoga_angles_prob[np.argmax(yoga_angles_prob)] > -0.3:

                #     cv2.putText(frame, yoga_poses_class,(15, 75) , 
                #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 2)
                
                #     cv2.putText(frame, 'PROB'
                #                 , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                #     cv2.putText(frame, str(round(yoga_poses_prob[np.argmax(yoga_poses_prob)],2))
                #                 , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                

                # cv2.putText(output_image, label, (350, 75),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            

            return label 

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img, self.results
 
    def findPosition(self, img):
        self.lmList = []
        try:
            if self.results.pose_landmarks:
                for lm in self.results.pose_landmarks.landmark:
                    h, w, _ = img.shape
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    self.lmList.append([cx, cy, cz])
        except:
            pass
        return self.lmList
 
    def findAngle(self, p1, p2, p3):
        # Get the landmarks
        x1, y1, _ = self.lmList[p1]
        x2, y2, _ = self.lmList[p2]
        x3, y3, _ = self.lmList[p3]
 
        # Calculate the Angle
        angle = abs(math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2)))

        if angle > 180:
            angle = 360 - angle

        return angle
    
    # def findAngle(self, p1, p2, p3):
    #     # Get the landmarks
    #     x1, y1, _ = self.lmList[p1]
    #     x2, y2, _ = self.lmList[p2]
    #     x3, y3, _ = self.lmList[p3]
 
    #     # Calculate the Angle
    #     angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
    #                          math.atan2(y1 - y2, x1 - x2))
    #     if angle < 0:
    #         angle += 360
 
    #     return angle

    def classifyPose(self):
        '''
        This function classifies yoga poses depending upon the angles of various body joints.
        Args:
            landmarks: A list of detected landmarks of the person whose pose needs to be classified.
            output_image: A image of the person with the detected pose landmarks drawn.
            display: A boolean value that is if set to true the function displays the resultant image with the pose label 
            written on it and returns nothing.
        Returns:
            output_image: The image with the detected pose landmarks drawn and pose label written.
            label: The classified pose label of the person in the output_image.
        '''
        
        # Initialize the label of the pose. It is not known at this stage.
        label = 'Unknown Pose'

        # Calculate the required angles.
        #----------------------------------------------------------------------------------------------------------------
        
        # Get the angle between the left shoulder, elbow and wrist points.
        left_elbow_angle = self.findAngle(self.mpPose.PoseLandmark.LEFT_SHOULDER.value, 
                                        self.mpPose.PoseLandmark.LEFT_ELBOW.value, 
                                        self.mpPose.PoseLandmark.LEFT_WRIST.value)

        # Get the angle between the right shoulder, elbow and wrist points. 
        right_elbow_angle = self.findAngle(self.mpPose.PoseLandmark.RIGHT_SHOULDER.value,
                                        self.mpPose.PoseLandmark.RIGHT_ELBOW.value,
                                        self.mpPose.PoseLandmark.RIGHT_WRIST.value)   
        
        # Get the angle between the left elbow, shoulder and hip points. 
        left_shoulder_angle = self.findAngle(self.mpPose.PoseLandmark.LEFT_ELBOW.value,
                                            self.mpPose.PoseLandmark.LEFT_SHOULDER.value,
                                            self.mpPose.PoseLandmark.LEFT_HIP.value)

        # Get the angle between the right hip, shoulder and elbow points. 
        right_shoulder_angle = self.findAngle(self.mpPose.PoseLandmark.RIGHT_HIP.value,
                                            self.mpPose.PoseLandmark.RIGHT_SHOULDER.value,
                                            self.mpPose.PoseLandmark.RIGHT_ELBOW.value)

        # Get the angle between the left hip, knee and ankle points. 
        left_knee_angle = self.findAngle(self.mpPose.PoseLandmark.LEFT_HIP.value,
                                        self.mpPose.PoseLandmark.LEFT_KNEE.value,
                                        self.mpPose.PoseLandmark.LEFT_ANKLE.value)

        # Get the angle between the right hip, knee and ankle points 
        right_knee_angle = self.findAngle(self.mpPose.PoseLandmark.RIGHT_HIP.value,
                                        self.mpPose.PoseLandmark.RIGHT_KNEE.value,
                                        self.mpPose.PoseLandmark.RIGHT_ANKLE.value)
        
        #----------------------------------------------------------------------------------------------------------------
        
        # Check if it is the warrior II pose or the T pose.
        # As for both of them, both arms should be straight and shoulders should be at the specific angle.
        #----------------------------------------------------------------------------------------------------------------
        
        # Check if the both arms are straight.
        if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

            # Check if shoulders are at the required angle.
            if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

        # Check if it is the warrior II pose.
        #----------------------------------------------------------------------------------------------------------------

                # Check if one leg is straight.
                if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                    # Check if the other leg is bended at the required angle.
                    if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                        # Specify the label of the pose that is Warrior II pose.
                        label = 'Warrior Pose' 
                            
        #----------------------------------------------------------------------------------------------------------------
        
        # Check if it is the T pose.
        #----------------------------------------------------------------------------------------------------------------
        
                # Check if both legs are straight
                if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

                    # Specify the label of the pose that is tree pose.
                    label = 'T Pose'

        #----------------------------------------------------------------------------------------------------------------
        
        # Check if it is the tree pose.
        #----------------------------------------------------------------------------------------------------------------
        
        # Check if one leg is straight
        if left_knee_angle > 165 and left_knee_angle < 230 or right_knee_angle > 165 and right_knee_angle < 230:

            # Check if the other leg is bended at the required angle.
            if left_knee_angle > 315 and left_knee_angle < 360 or right_knee_angle > 15 and right_knee_angle < 75:
                
                # Specify the label of the pose that is tree pose.
                label = 'Tree Pose'
                    
        #----------------------------------------------------------------------------------------------------------------

        return label
        