import cv2
import numpy as np
import time
import PoseModule as pm
import os

dirname = os.path.dirname(__file__)

cap = cv2.VideoCapture(0)
#-----------------------------
window_width = 1280 #960 #1024
window_height= 720 #540 #768
#-------------------------------
cap.set(3,window_width)
cap.set(4,window_height)

enable_segmentation=True
# #-------------------------------------------
# # Background video for Yoga
# videoBack_yoga = 'T_pose_flip.mp4'
# bg_video_name_yoga = os.path.join(dirname, videoBack_yoga)
# capBackground_yoga = cv2.VideoCapture(bg_video_name_yoga)
# capBackground_yoga.set(3,window_height)
# capBackground_yoga.set(4,window_width)

# #-------------------------------------------
# # Background video fot Workout
# videoBack_workout = 'Arms.mp4'
# bg_video_name_workout = os.path.join(dirname, videoBack_workout)
# capBackground_workout = cv2.VideoCapture(bg_video_name_workout)
# capBackground_workout.set(3,window_height)
# capBackground_workout.set(4,window_width)

detector = pm.poseDetector(enable_segmentation=enable_segmentation)
count = 0
#-----------------------
curr_count_bar_scale = 0
#------------------------
dir = 0
pTime = 0

black_clr = (0, 0, 0)
white_clr = (255, 255, 255)
green_clr = (0, 250, 0)
red_clr = (0, 0, 255)
blue_clr = (255, 0, 0)
#--------------------------
grey_clr = (192, 192, 192)
#-------------------------

black_matrix = np.zeros((window_width, window_height), dtype=np.uint8)

tng_duration = 5 * 60 # mins, secs
pose_duration = 3 # secs

label = 'Unknown Pose'
label_next_pose = 'T Pose'


wtimer_left = tng_duration
wtimer_text_size = 2
wtimer_curr_rec_width = 0
winit_time = time.time()
prev_wcurr_time = 0
wmins, wsecs = divmod(wtimer_left, 60)
wtimer = '{:02d}:{:02d}'.format(wmins, wsecs)

ptimer_left = pose_duration
ptimer_curr_rec_width = 0
pinit_time = time.time()
prev_pcurr_time = 0

wtimer_tapped = False
wtimer_active = True
wtimer_show = True
ptimer_active = True
ptimer_show = True

exit_butt_active = False
cont_butt_active = False

yoga_butt_active = True
workout_butt_active = True

yoga_active = False
workout_active = False

winit_time = time.time()
wcurr_time = 0

pinit_time = time.time()
pcurr_time = 0

hand_in_wtimer_inittime = time.time()
hand_in_wtimer_curr_time = 0

hand_in_exit_butt_inittime = time.time()
hand_in_exit_butt_curr_time = 0

hand_in_cont_butt_inittime = time.time()
hand_in_cont_butt_curr_time = 0

hand_in_ptimer_inittime = time.time()
hand_in_ptimer_curr_time = 0

hand_in_yoga_butt_inittime = time.time()
hand_in_yoga_butt_curr_time = 0

hand_in_workout_butt_inittime = time.time()
hand_in_workout_butt_curr_time = 0

lhand_in_wtimer = False
rhand_in_wtimer = False
lhand_in_ptimer = False
rhand_in_ptimer = False
lhand_in_exit_butt = False
rhand_in_exit_butt = False
lhand_in_cont_butt = False
rhand_in_cont_butt = False
lhand_in_yoga_butt = False
rhand_in_yoga_butt = False
lhand_in_workout_butt = False
rhand_in_workout_butt = False

ptimer_pressed = False
wtimer_pressed = False
exit_butt_pressed = False
cont_butt_pressed = False
yoga_butt_pressed = False
workout_butt_pressed = False

# Indexes of all landmarks
NOSE = detector.mpPose.PoseLandmark.NOSE.value
LEFT_EYE_INNER = detector.mpPose.PoseLandmark.LEFT_EYE_INNER.value
LEFT_EYE = detector.mpPose.PoseLandmark.LEFT_EYE.value
LEFT_EYE_OUTER = detector.mpPose.PoseLandmark.LEFT_EYE_OUTER.value
RIGHT_EYE_INNER = detector.mpPose.PoseLandmark.RIGHT_EYE_INNER.value
RIGHT_EYE = detector.mpPose.PoseLandmark.RIGHT_EYE.value
RIGHT_EYE_OUTER = detector.mpPose.PoseLandmark.RIGHT_EYE_OUTER.value
LEFT_EAR = detector.mpPose.PoseLandmark.LEFT_EAR.value
RIGHT_EAR = detector.mpPose.PoseLandmark.RIGHT_EAR.value
MOUTH_LEFT = detector.mpPose.PoseLandmark.MOUTH_LEFT.value
MOUTH_RIGHT = detector.mpPose.PoseLandmark.MOUTH_RIGHT.value
LEFT_SHOULDER = detector.mpPose.PoseLandmark.LEFT_SHOULDER.value
RIGHT_SHOULDER = detector.mpPose.PoseLandmark.RIGHT_SHOULDER.value
LEFT_ELBOW = detector.mpPose.PoseLandmark.LEFT_ELBOW.value
RIGHT_ELBOW = detector.mpPose.PoseLandmark.RIGHT_ELBOW.value
LEFT_WRIST = detector.mpPose.PoseLandmark.LEFT_WRIST.value
RIGHT_WRIST = detector.mpPose.PoseLandmark.RIGHT_WRIST.value
LEFT_PINKY = detector.mpPose.PoseLandmark.LEFT_PINKY.value
RIGHT_PINKY = detector.mpPose.PoseLandmark.RIGHT_PINKY.value
LEFT_INDEX = detector.mpPose.PoseLandmark.LEFT_INDEX.value
RIGHT_INDEX = detector.mpPose.PoseLandmark.RIGHT_INDEX.value
LEFT_THUMB = detector.mpPose.PoseLandmark.LEFT_THUMB.value
RIGHT_THUMB = detector.mpPose.PoseLandmark.RIGHT_THUMB.value
LEFT_HIP = detector.mpPose.PoseLandmark.LEFT_HIP.value
RIGHT_HIP = detector.mpPose.PoseLandmark.RIGHT_HIP.value
LEFT_KNEE = detector.mpPose.PoseLandmark.LEFT_KNEE.value
RIGHT_KNEE = detector.mpPose.PoseLandmark.RIGHT_KNEE.value
LEFT_ANKLE = detector.mpPose.PoseLandmark.LEFT_ANKLE.value
RIGHT_ANKLE = detector.mpPose.PoseLandmark.RIGHT_ANKLE.value
LEFT_HEEL = detector.mpPose.PoseLandmark.LEFT_HEEL.value
RIGHT_HEEL = detector.mpPose.PoseLandmark.RIGHT_HEEL.value
LEFT_FOOT_INDEX = detector.mpPose.PoseLandmark.LEFT_FOOT_INDEX.value
RIGHT_FOOT_INDEX = detector.mpPose.PoseLandmark.RIGHT_FOOT_INDEX.value

# Indexes of landmarks for pose classification and correction
main_lnms = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
            LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE,
            RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]


def drawPoints(img, landmarks, points=[]):
    points.sort()
    for p_idx, point in enumerate(points):
        # Draw point
        x1, y1, _ = landmarks[point]
        drawCircle(img, x1, y1)
        # Draw line and next point
        next_p_idx = p_idx + 1
        if next_p_idx < len(points):
            next_point = points[next_p_idx]
            x2, y2, _ = landmarks[next_point]
            cv2.line(img, (x1, y1), (x2, y2), white_clr, 3)
            drawCircle(img, x2, y2)
            # Draw angle
            if next_p_idx + 1 < len(points):
                angle = detector.findAngle(point, next_point, points[next_p_idx + 1])
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, white_clr, 2)

def drawCircle(img, x, y, clr = red_clr):
    cv2.circle(img, (x, y), 10, clr, cv2.FILLED)
    cv2.circle(img, (x, y), 15, clr, 2)

# Congrats
img_glasses = cv2.imread(os.path.join(dirname, 'imgs/glasses_cig.png'), cv2.IMREAD_UNCHANGED)
img_glasses = cv2.resize(img_glasses, (0, 0), None, 0.25, 0.25)
img_cows = cv2.imread(os.path.join(dirname, 'imgs/cows.jpg'), cv2.IMREAD_UNCHANGED)
img_cows = cv2.resize(img_cows,(window_width,window_height))

def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    *_, mask = cv2.split(imgFront)
    maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
    imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

    imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
    imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255

    try:
        imgMaskFull[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = imgRGB
        maskBGRInv = cv2.bitwise_not(maskBGR)
        imgMaskFull2[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = maskBGRInv
    except:
        pass

    imgBack = cv2.bitwise_and(imgBack, imgMaskFull2)
    imgBack = cv2.bitwise_or(imgBack, imgMaskFull)

    return imgBack



cv2.namedWindow('Fizruk', cv2.WINDOW_NORMAL)
#-------------------------------------------

while cap.isOpened():
    _, img = cap.read()

    # Flip the frame horizontally for natural (selfie-view) visualization.
    img = cv2.flip(img, 1)

    img, results = detector.findPose(img, draw=True)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        
        # Hands Menu Control
        lhand_x, lhand_y, _ = lmList[RIGHT_INDEX]
        rhand_x, rhand_y, _ = lmList[LEFT_INDEX]


        # Youga
        if yoga_active:

            # Right Arm
            angle = detector.findAngle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)


            # drawPoints(img, lmList, points=[NOSE])
            # drawPoints(img, lmList, points=[RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST])

            # TODO: Try to use async
            # Segmentation
            if enable_segmentation:
                


                # Read background video frame
                success, img_back = capBackground_yoga.read()

                # TODO: Refactor to cap.repeat()
                # Repeat video if it's end
                if not success:
                    # capBackground = cv2.VideoCapture(bg_video_name)
                    # capBackground.set(3,window_height)
                    # capBackground.set(4,window_width)
                    # success, img_back = capBackground.read()
                    capBackground_yoga.set(1, 0)

                else:
                    # TODO: Use video with correct resolution
                    # Resize video frame to be equal window's size
                    img_back = cv2.resize(img_back,(window_width,window_height),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                    # Add background frame to segmented user's frame
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    #-------------------------------------------
                    # Замена img.shape на img_back.shape
                    bg_image = np.zeros(img_back.shape, dtype=np.uint8)
                    bg_image[:] = img_back
                    img = np.where(condition, img, bg_image)
                    


            # Draw congrats
            if label_next_pose == 'Done!':
                
                # TODO: DRY
                # Add background frame to segmented user's frame
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(img.shape, dtype=np.uint8)
                bg_image[:] = img_cows
                img = np.where(condition, img, bg_image)

                reye_x, reye_y, _ = lmList[RIGHT_EYE_OUTER]
                leye_x, leye_y, _ = lmList[LEFT_EYE_OUTER]

                # # Resize glasses relative to depth (changed distance between eyes)
                # glasses_w = 1 #abs(int((leye_x - reye_x) * 1.3))
                # glasses_h = 1 #int(glasses_w * 1.05)
                # img_glasses = cv2.resize(img_glasses, (glasses_w, glasses_h),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                # print(glasses_w, glasses_h)
                img = overlayPNG(img, img_glasses, [reye_x - 20, reye_y])

            # Workout timer

            wtimer_pos_x = 10
            wtimer_pos_y = 10
            wtimer_width = 300
            wtimer_height = 100
            wtimer_width_step = wtimer_width / tng_duration
            
            if wtimer_active:
                
                if wtimer_left > 0:
                    wcurr_time = int(time.time() - winit_time)

                    if prev_wcurr_time != wcurr_time:
                        wtimer_curr_rec_width += int(wtimer_width_step)
                        prev_wcurr_time = wcurr_time
                        wtimer_left -= 1

                        wmins, wsecs = divmod(wtimer_left, 60)
                        wtimer = '{:02d}:{:02d}'.format(wmins, wsecs)
                else:
                    wtimer= 'Time is end!'
                    wtimer_text_size = 1
                    winit_time = time.time()
            else:
                winit_time = time.time()

            if wtimer_show:
                #Draw workout timer
                cv2.rectangle(img, 
                            (wtimer_pos_x, wtimer_pos_y), 
                            (wtimer_pos_x + wtimer_width, wtimer_pos_y + wtimer_height), 
                            black_clr, 
                            3)
                cv2.rectangle(img, 
                            (wtimer_pos_x, wtimer_pos_y), 
                            (wtimer_pos_x + wtimer_width, wtimer_pos_y + wtimer_height), 
                            grey_clr, 
                            cv2.FILLED)

                if wtimer_left > 0:
                    cv2.rectangle(img, 
                                (wtimer_pos_x, wtimer_pos_y), 
                                (wtimer_pos_x + wtimer_curr_rec_width, wtimer_pos_y + wtimer_height), 
                                black_clr, 
                                cv2.FILLED)
                else:
                    cv2.rectangle(img, 
                                (wtimer_pos_x, wtimer_pos_y), 
                                (wtimer_pos_x + wtimer_width, wtimer_pos_y + wtimer_height), 
                                black_clr, 
                                cv2.FILLED)
                
                #Workout timer text
                cv2.putText(img, 
                            'Pause', 
                            (wtimer_pos_x + 5, wtimer_pos_y + 65), 
                            cv2.FONT_HERSHEY_DUPLEX, 
                            1, 
                            white_clr, 
                            2)
                cv2.putText(img, 
                            wtimer, 
                            (wtimer_pos_x + 115, wtimer_pos_y + 65), 
                            cv2.FONT_HERSHEY_DUPLEX, 
                            wtimer_text_size, 
                            white_clr, 
                            2)

            
            #Pose timer
            ptimer_pos_y = 10
            ptimer_width = 300
            ptimer_height = 100
            #-------------------------------------------
            ptimer_pos_x = window_width - ptimer_width - 20 #650
            #-------------------------------------------
            ptimer_width_step = ptimer_width / pose_duration
            ptimer_text = 'Next'
            
            # Change pose timer button if it's active
            if ptimer_active:
                # Change anything only if we detect any pose
                if label != 'Unknown Pose':
                    # Fill pose timer button by increasing filled button's rectangle each time when
                    # we current pose is equal label_next_pose and when pose timer is not end.
                    if ptimer_left > 0 and label == label_next_pose:
                        # Calculate time that starts right after we detect correct pose (label == label_next_pose)
                        pcurr_time = int(time.time() - pinit_time)
                        # Change filled pose timer rectangle each second. One second is 
                        # passed when 'prev_pcurr_time != pcurr_time'
                        if prev_pcurr_time != pcurr_time:
                            # Increase filled pose timer rectangle for one time (+ ptimer_curr_rec_width)
                            ptimer_curr_rec_width += int(ptimer_width_step)
                            # Save current pose time (pcurr_time) to prev_pcurr_time for detect 
                            #  when 1 second is passed
                            prev_pcurr_time = pcurr_time
                            # Update timer that indicates how much time is left to end the pose.
                            ptimer_left -= 1
                    # Reset timers and pose label if pose time is ended.
                    elif ptimer_left <= 0:
                        ptimer_left = pose_duration
                        ptimer_curr_rec_width = 0

                        if label_next_pose == 'T Pose':
                            label_next_pose = 'Warrior Pose'
                            videoBack_yoga = 'Warrior_flip.mp4'
                        elif label_next_pose == 'Warrior Pose':
                            label_next_pose = 'Tree Pose'
                            videoBack_yoga = 'Tree_flip.mp4'
                        elif label_next_pose == 'Tree Pose':
                            label_next_pose = 'Done!'
                        
                        # Background video 
                        bg_video_name_yoga = os.path.join(dirname, videoBack_yoga)
                        capBackground_yoga = cv2.VideoCapture(bg_video_name_yoga)
                        # capBackground_yoga.set(3,window_height)
                        # capBackground_yoga.set(4,window_width)

                else:
                    pinit_time = time.time()


            if ptimer_show:
                cv2.rectangle(img, 
                            (ptimer_pos_x, ptimer_pos_y), 
                            (ptimer_pos_x + ptimer_width, ptimer_pos_y + ptimer_height), 
                            black_clr, 
                            3)
                cv2.rectangle(img, 
                            (ptimer_pos_x, ptimer_pos_y), 
                            (ptimer_pos_x + ptimer_width, ptimer_pos_y + ptimer_height), 
                            grey_clr, 
                            cv2.FILLED)

                if ptimer_left > 0:
                    cv2.rectangle(img, 
                                (ptimer_pos_x, ptimer_pos_y), 
                                (ptimer_pos_x + ptimer_curr_rec_width, ptimer_pos_y + ptimer_height), 
                                black_clr, 
                                cv2.FILLED)
                else:
                    cv2.rectangle(img, 
                                (ptimer_pos_x, ptimer_pos_y), 
                                (ptimer_pos_x + ptimer_width, ptimer_pos_y + ptimer_height), 
                                black_clr, 
                                cv2.FILLED)
                
                #Workout timer text
                cv2.putText(img, 
                            ptimer_text, 
                            (ptimer_pos_x + 100, ptimer_pos_y + 65), 
                            cv2.FONT_HERSHEY_DUPLEX, 
                            1, 
                            white_clr, 
                            2)

            # Wtimer
            if wtimer_active:
                lhand_x_in_wtimer_x = lhand_x > wtimer_pos_x and lhand_x < (wtimer_pos_x + wtimer_width)
                lhand_y_in_wtimer_y = lhand_y > wtimer_pos_y and lhand_y < (wtimer_pos_y + wtimer_height)
                lhand_in_wtimer = lhand_x_in_wtimer_x and lhand_y_in_wtimer_y

                rhand_x_in_wtimer_x = rhand_x > wtimer_pos_x and rhand_x < (wtimer_pos_x + wtimer_width)
                rhand_y_in_wtimer_y = rhand_y > wtimer_pos_y and rhand_y < (wtimer_pos_y + wtimer_height)
                rhand_in_wtimer = rhand_x_in_wtimer_x and rhand_y_in_wtimer_y

                if (lhand_in_wtimer or rhand_in_wtimer) and not (lhand_in_ptimer or rhand_in_ptimer):
                    cv2.rectangle(img, 
                                (wtimer_pos_x, wtimer_pos_y), 
                                (wtimer_pos_x + wtimer_width, wtimer_pos_y + wtimer_height), 
                                green_clr, 
                                3)

                    # Detect 'tap' on wtimer
                    hand_in_wtimer_curr_time = int(time.time() - hand_in_wtimer_inittime)
                    if hand_in_wtimer_curr_time >= 1 and not wtimer_pressed:

                        wtimer_tapped = True
                        exit_butt_active = True
                        cont_butt_active = True
                        
                        cv2.rectangle(img, 
                                (wtimer_pos_x, wtimer_pos_y), 
                                (wtimer_pos_x + wtimer_width, wtimer_pos_y + wtimer_height), 
                                green_clr, 
                                cv2.FILLED)

                        wtimer_pressed = True
                else:
                    hand_in_wtimer_inittime = time.time()
                    wtimer_pressed = False


            #--------------------------------------------------------------------------------------------------------------------------------------------
            # Perform Pose landmark detection.
            label = detector.classifyPose()

            # Exersize sequence

            # Update the color (to green) with which the label will be written on the image.
            label_clr = red_clr if label != label_next_pose else green_clr
            if label_next_pose == 'Done!': label_clr = green_clr
            
            cv2.putText(img, label_next_pose, (int((ptimer_pos_x - wtimer_pos_x + wtimer_width) / 2.5), 75),cv2.FONT_HERSHEY_PLAIN, 2, label_clr, 2)
            #--------------------------------------------------------------------------------------------------------------------------------------------

            # Draw pause buttons
            if wtimer_tapped:
                
                # Deactivate wtimer and ptimer buttons
                wtimer_active = False
                ptimer_active = False

                # Exit button
                if exit_butt_active:
                    
                    # Draw exit button
                    #-------------------------------------------
                    exit_butt_pos_x = (int((ptimer_pos_x - wtimer_pos_x + wtimer_width) / 4)) #int(window_width / 5.5)
                    #-------------------------------------------
                    exit_butt_pos_y = 200
                    exit_butt_width = 300
                    exit_butt_height = 100

                    cv2.rectangle(img, 
                                (exit_butt_pos_x, exit_butt_pos_y), 
                                (exit_butt_pos_x + exit_butt_width, exit_butt_pos_y + exit_butt_height), 
                                black_clr, 
                                3)
                    cv2.rectangle(img, 
                                (exit_butt_pos_x, exit_butt_pos_y), 
                                (exit_butt_pos_x + exit_butt_width, exit_butt_pos_y + exit_butt_height), 
                                black_clr, 
                                cv2.FILLED)
                    
                    # Exit button text
                    cv2.putText(img, 
                                'Exit', 
                                (exit_butt_pos_x + 100, exit_butt_pos_y + 65), 
                                cv2.FONT_HERSHEY_DUPLEX, 
                                1, 
                                white_clr, 
                                2)

                    lhand_x_in_exit_butt_x = lhand_x > exit_butt_pos_x and lhand_x < (exit_butt_pos_x + exit_butt_width)
                    lhand_y_in_exit_butt_y = lhand_y > exit_butt_pos_y and lhand_y < (exit_butt_pos_y + exit_butt_height)
                    lhand_in_exit_butt = lhand_x_in_exit_butt_x and lhand_y_in_exit_butt_y

                    rhand_x_in_exit_butt_x = rhand_x > exit_butt_pos_x and rhand_x < (exit_butt_pos_x + exit_butt_width)
                    rhand_y_in_exit_butt_y = rhand_y > exit_butt_pos_y and rhand_y < (exit_butt_pos_y + exit_butt_height)
                    rhand_in_exit_butt = rhand_x_in_exit_butt_x and rhand_y_in_exit_butt_y

                    if (lhand_in_exit_butt or rhand_in_exit_butt) and not (lhand_in_cont_butt or rhand_in_cont_butt):
                        cv2.rectangle(img, 
                                    (exit_butt_pos_x, exit_butt_pos_y), 
                                    (exit_butt_pos_x + exit_butt_width, exit_butt_pos_y + exit_butt_height), 
                                    green_clr, 
                                    3)

                        # Detect 'tap' on exit_butt
                        hand_in_exit_butt_curr_time = int(time.time() - hand_in_exit_butt_inittime)
                        if hand_in_exit_butt_curr_time >= 1 and not exit_butt_pressed:

                            cv2.rectangle(img, 
                                    (exit_butt_pos_x, exit_butt_pos_y), 
                                    (exit_butt_pos_x + exit_butt_width, exit_butt_pos_y + exit_butt_height), 
                                    green_clr, 
                                    cv2.FILLED)
                            
                            exit_butt_active = False
                            wtimer_tapped = False
                            yoga_active = False
                            #-------------------------------------------
                            workout_active =False
                            #-------------------------------------------
                            yoga_butt_active = True
                            workout_butt_active = True

                            exit_butt_pressed = True

                    else:
                        hand_in_exit_butt_inittime = time.time()
                        exit_butt_pressed = False
                
                

                # Cont button
                if cont_butt_active:
                    # Draw continue button
                    #-------------------------------------------
                    cont_butt_pos_x = exit_butt_pos_x + exit_butt_width + 60
                    #-------------------------------------------
                    cont_butt_pos_y = 200
                    cont_butt_width = 300
                    cont_butt_height = 100

                    cv2.rectangle(img, 
                                (cont_butt_pos_x, cont_butt_pos_y), 
                                (cont_butt_pos_x + cont_butt_width, cont_butt_pos_y + cont_butt_height), 
                                black_clr, 
                                3)
                    cv2.rectangle(img, 
                                (cont_butt_pos_x, cont_butt_pos_y), 
                                (cont_butt_pos_x + cont_butt_width, cont_butt_pos_y + cont_butt_height), 
                                black_clr, 
                                cv2.FILLED)
                    
                    # Exit button text
                    cv2.putText(img, 'Continue', (cont_butt_pos_x + 80, cont_butt_pos_y + 65), cv2.FONT_HERSHEY_DUPLEX, 1, 
                                white_clr, 2)

                    lhand_x_in_cont_butt_x = lhand_x > cont_butt_pos_x and lhand_x < (cont_butt_pos_x + cont_butt_width)
                    lhand_y_in_cont_butt_y = lhand_y > cont_butt_pos_y and lhand_y < (cont_butt_pos_y + cont_butt_height)
                    lhand_in_cont_butt = lhand_x_in_cont_butt_x and lhand_y_in_cont_butt_y

                    rhand_x_in_cont_butt_x = rhand_x > cont_butt_pos_x and rhand_x < (cont_butt_pos_x + cont_butt_width)
                    rhand_y_in_cont_butt_y = rhand_y > cont_butt_pos_y and rhand_y < (cont_butt_pos_y + cont_butt_height)
                    rhand_in_cont_butt = rhand_x_in_cont_butt_x and rhand_y_in_cont_butt_y

                    if (lhand_in_cont_butt or rhand_in_cont_butt) and not (lhand_in_exit_butt or rhand_in_exit_butt):
                        cv2.rectangle(img, 
                                    (cont_butt_pos_x, cont_butt_pos_y), 
                                    (cont_butt_pos_x + cont_butt_width, cont_butt_pos_y + cont_butt_height), 
                                    green_clr, 
                                    3)

                        # Detect 'tap' on cont_butt
                        hand_in_cont_butt_curr_time = int(time.time() - hand_in_cont_butt_inittime)
                        if hand_in_cont_butt_curr_time >= 1 and not cont_butt_pressed:
                            
                            cont_butt_active = False
                            wtimer_tapped = False
                            wtimer_active = True
                            ptimer_active = True

                            cv2.rectangle(img, 
                                    (cont_butt_pos_x, cont_butt_pos_y), 
                                    (cont_butt_pos_x + cont_butt_width, cont_butt_pos_y + cont_butt_height), 
                                    green_clr, 
                                    cv2.FILLED)

                            cont_butt_pressed = True

                    else:
                        hand_in_cont_butt_inittime = time.time()
                        cont_butt_pressed = False


            # Ptimer
            if ptimer_active:
                lhand_x_in_ptimer_x = lhand_x > ptimer_pos_x and lhand_x < (ptimer_pos_x + ptimer_width)
                lhand_y_in_ptimer_y = lhand_y > ptimer_pos_y and lhand_y < (ptimer_pos_y + ptimer_height)
                lhand_in_ptimer = lhand_x_in_ptimer_x and lhand_y_in_ptimer_y

                rhand_x_in_ptimer_x = rhand_x > ptimer_pos_x and rhand_x < (ptimer_pos_x + ptimer_width)
                rhand_y_in_ptimer_y = rhand_y > ptimer_pos_y and rhand_y < (ptimer_pos_y + ptimer_height)
                rhand_in_ptimer = rhand_x_in_ptimer_x and rhand_y_in_ptimer_y

                if (lhand_in_ptimer or rhand_in_ptimer) and not (lhand_in_wtimer or rhand_in_wtimer):
                    cv2.rectangle(img, 
                                (ptimer_pos_x, ptimer_pos_y), 
                                (ptimer_pos_x + ptimer_width, ptimer_pos_y + ptimer_height), 
                                green_clr, 
                                3)

                    # Detect 'tap' on ptimer
                    hand_in_ptimer_curr_time = int(time.time() - hand_in_ptimer_inittime)
                    if hand_in_ptimer_curr_time >= 1 and not ptimer_pressed:
                        ptimer_left = pose_duration
                        ptimer_curr_rec_width = 0

                        if label_next_pose == 'T Pose':
                            label_next_pose = 'Warrior Pose'
                            videoBack_yoga = 'Warrior_flip.mp4'
                        elif label_next_pose == 'Warrior Pose':
                            label_next_pose = 'Tree Pose'
                            videoBack_yoga = 'Tree_flip.mp4'
                        elif label_next_pose == 'Tree Pose':
                            label_next_pose = 'Done!'

                        
                        # Background video 
                        bg_video_name_yoga = os.path.join(dirname, videoBack_yoga)
                        capBackground_yoga = cv2.VideoCapture(bg_video_name_yoga)
                        # capBackground_yoga.set(3,window_height)
                        # capBackground_yoga.set(4,window_width)

                        cv2.rectangle(img, 
                                (ptimer_pos_x, ptimer_pos_y), 
                                (ptimer_pos_x + ptimer_width, ptimer_pos_y + ptimer_height), 
                                green_clr, 
                                cv2.FILLED)

                        ptimer_pressed = True

                else:
                    hand_in_ptimer_inittime = time.time()
                    ptimer_pressed = False
            
        #----------------------------------------------------------------------------------
        # WORKOUT TRAIN
        #----------------------------------------------------------------------------------
        if workout_active:

            # Button "Next" + counter exersices
            count_text_x = window_width - 80 #880
            count_text_y = 80
            
            
            count_bar_y = 10
            count_bar_weight = 300
            count_bar_height = 100
            count_bar_step = count_bar_weight / 6
            count_bar_text = 'Next'
            count_bar_x = window_width - count_bar_weight - 20
           
           
            # TODO: Try to use async
            # Segmentation
            if enable_segmentation:

                # Read background video frame
                success, img_back = capBackground_workout.read()

                # TODO: Refactor to cap.repeat()
                # Repeat video if it's end
                if not success:
                    # capBackground = cv2.VideoCapture(bg_video_name)
                    # capBackground.set(3,window_height)
                    # capBackground.set(4,window_width)
                    # success, img_back = capBackground.read()
                    capBackground_workout.set(1, 0)

                else:
                    # TODO: Use video with correct resolution
                    # Resize video frame to be equal window's size
                    img_back = cv2.resize(img_back,(window_width,window_height),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                    # Add background frame to segmented user's frame
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    bg_image = np.zeros(img_back.shape, dtype=np.uint8)
                    bg_image[:] = img_back
                    img = np.where(condition, img, bg_image)

            # Draw congrats
            if label_next_pose == 'Done!':
                
                # TODO: DRY
                # Add background frame to segmented user's frame
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(img.shape, dtype=np.uint8)
                bg_image[:] = img_cows
                img = np.where(condition, img, bg_image)

                reye_x, reye_y, _ = lmList[RIGHT_EYE_OUTER]
                leye_x, leye_y, _ = lmList[LEFT_EYE_OUTER]

                # Resize glasses relative to depth (changed distance between eyes)
                # glasses_w = 50#abs(int((leye_x - reye_x) * 1.3))
                # glasses_h = 50#int(glasses_w * 1.05)
                # img_glasses = cv2.resize(img_glasses, (glasses_w, glasses_h),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

                img = overlayPNG(img, img_glasses, [reye_x - 20, reye_y])

            def set_exercise_params_for_counter(p1_left, p2_left, p3_left, p1_right, p2_right, p3_right, min_angle_1, min_angle_2, max_angle_1, max_angle_2):
                # Find angles
                angle_left = detector.findAngle(p1_left, p2_left, p3_left)
                angle_right = detector.findAngle(p1_right, p2_right, p3_right)

                # Draw points
                drawPoints(img, lmList, points=[p1_left, p2_left, p3_left])
                drawPoints(img, lmList, points=[p1_right, p2_right, p3_right])
        
                # Set angle range for counter
                min_angle = min_angle_1 if angle_right or angle_left > 180 else min_angle_2
                max_angle = max_angle_1 if angle_right or angle_left > 180 else max_angle_2

                per_left = np.interp(angle_left, (min_angle, max_angle), (0, 100))
                per_right = np.interp(angle_right, (min_angle, max_angle), (0, 100))

                return per_left, per_right
            
            
            # def set_exercise_params_for_counter(angle_left=[], angle_right=[], min_angle=[], max_angle=[]):
            #     # Find angles
            #     angle_left = detector.findAngle(angle_left[0], angle_left[1], angle_left[2])
            #     angle_right = detector.findAngle(angle_right[0], angle_right[1], angle_right[2])

            #     # Draw points
            #     drawPoints(img, lmList, points=angle_left)
            #     drawPoints(img, lmList, points=angle_right)
        
            #     # Set angle range for counter
            #     min_angle = min_angle[0] if angle_right or angle_left > 180 else min_angle[1]
            #     max_angle = max_angle[0] if angle_right or angle_left > 180 else max_angle[1]

            #     per_left = np.interp(angle_left, (min_angle, max_angle), (0, 100))
            #     per_right = np.interp(angle_right, (min_angle, max_angle), (0, 100))

            #     return per_left, per_right

            if label_next_pose == 'Arms':
                per_left, per_right = set_exercise_params_for_counter(
                                                    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, 
                                                    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, 
                                                    35, 35, 
                                                    135, 135)

            elif label_next_pose == 'Legs':
                per_left, per_right = set_exercise_params_for_counter(
                                                    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, 
                                                    RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, 
                                                    90, 90, 
                                                    160, 160 )

            elif label_next_pose == 'Gym':
                per_left, per_right = set_exercise_params_for_counter(
                                                    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, 
                                                    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, 
                                                    75, 75, 
                                                    135, 130)
                
            #--------------------------------------------------------------------------
            #Counter
            #--------------------------------------------------------------------------
            if per_left > 90 and per_right > 90:
                if dir == 0:
                    if count < 3:
                        count += 0.5
                        curr_count_bar_scale += count_bar_step
                        dir = 1
            if per_left < 10 and per_right < 10:
                if dir == 1:
                    if count < 3:
                        count += 0.5
                        curr_count_bar_scale += count_bar_step
                        dir = 0

            #--------------------------------------------------------------------------------------
            # Draw count_bar
            #--------------------------------------------------------------------------------------
            cv2.rectangle(img, (count_bar_x, count_bar_y), 
                                (count_bar_x + count_bar_weight, count_bar_y + count_bar_height), 
                                    black_clr, 3)
            cv2.rectangle(img, (count_bar_x, count_bar_y), 
                                (count_bar_x + count_bar_weight, count_bar_y + count_bar_height), 
                                    grey_clr, cv2.FILLED)

            # Do nothing when count = 0, else we get error
            if count == 0:
                pass
                
            # Reset count and switch exercises when count = 10
            elif count == 3:
                # Reset count
                count = 0
                # Reset scale on button "Next"
                curr_count_bar_scale = 0

                # Switch next exersises
                if label_next_pose == 'Arms':
                    label_next_pose = 'Legs'
                    videoBack_workout = 'Legs_flip.mp4'
                elif label_next_pose == 'Legs':
                    label_next_pose = 'Gym'
                    videoBack_workout = 'Gym_flip.mp4'
                elif label_next_pose == 'Gym':
                    label_next_pose = 'Done!'
     
                # Background video 
                bg_video_name_workout = os.path.join(dirname, videoBack_workout)
                capBackground_workout = cv2.VideoCapture(bg_video_name_workout)
                # capBackground_workout.set(3,window_height)
                # capBackground_workout.set(4,window_width)

            # Draw scale count
            else:
                cv2.rectangle(img, (count_bar_x, count_bar_y), 
                                    (count_bar_x + int(curr_count_bar_scale), count_bar_y + count_bar_height), 
                                    black_clr, cv2.FILLED)
            # Draw text on button
            cv2.putText(img, count_bar_text, (count_bar_x + 100, count_bar_y + 65), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, white_clr, 2)
            cv2.putText(img, str(int(count)), (count_text_x, count_text_y), cv2.FONT_HERSHEY_PLAIN, 3,white_clr, 3)

            #--------------------------------------------------------------------------------------------
            # Workout timer 
            #--------------------------------------------------------------------------------------------
            wtimer_pos_x = 10
            wtimer_pos_y = 10
            wtimer_width = 300
            wtimer_height = 100
            wtimer_width_step = wtimer_width / tng_duration

            if wtimer_active:
                
                if wtimer_left > 0:
                    wcurr_time = int(time.time() - winit_time)

                    if prev_wcurr_time != wcurr_time:
                        wtimer_curr_rec_width += int(wtimer_width_step)
                        prev_wcurr_time = wcurr_time
                        wtimer_left -= 1

                        wmins, wsecs = divmod(wtimer_left, 60)
                        wtimer = '{:02d}:{:02d}'.format(wmins, wsecs)
                else:
                    wtimer= 'Time is end!'
                    wtimer_text_size = 1
                    winit_time = time.time()
            else:
                winit_time = time.time()

            if wtimer_show:
                #Draw workout timer
                cv2.rectangle(img, 
                            (wtimer_pos_x, wtimer_pos_y), 
                            (wtimer_pos_x + wtimer_width, wtimer_pos_y + wtimer_height), 
                            black_clr, 
                            3)
                cv2.rectangle(img, 
                            (wtimer_pos_x, wtimer_pos_y), 
                            (wtimer_pos_x + wtimer_width, wtimer_pos_y + wtimer_height), 
                            grey_clr, 
                            cv2.FILLED)

                if wtimer_left > 0:
                    cv2.rectangle(img, 
                                (wtimer_pos_x, wtimer_pos_y), 
                                (wtimer_pos_x + wtimer_curr_rec_width, wtimer_pos_y + wtimer_height), 
                                black_clr, 
                                cv2.FILLED)
                else:
                    cv2.rectangle(img, 
                                (wtimer_pos_x, wtimer_pos_y), 
                                (wtimer_pos_x + wtimer_width, wtimer_pos_y + wtimer_height), 
                                black_clr, 
                                cv2.FILLED)
                
                #Workout timer text
                cv2.putText(img, 
                            'Pause', 
                            (wtimer_pos_x + 5, wtimer_pos_y + 65), 
                            cv2.FONT_HERSHEY_DUPLEX, 
                            1, 
                            white_clr, 
                            2)
                cv2.putText(img, 
                            wtimer, 
                            (wtimer_pos_x + 115, wtimer_pos_y + 65), 
                            cv2.FONT_HERSHEY_DUPLEX, 
                            wtimer_text_size, 
                            white_clr, 
                            2)
            

            # Wtimer
            if wtimer_active:
                lhand_x_in_wtimer_x = lhand_x > wtimer_pos_x and lhand_x < (wtimer_pos_x + wtimer_width)
                lhand_y_in_wtimer_y = lhand_y > wtimer_pos_y and lhand_y < (wtimer_pos_y + wtimer_height)
                lhand_in_wtimer = lhand_x_in_wtimer_x and lhand_y_in_wtimer_y

                rhand_x_in_wtimer_x = rhand_x > wtimer_pos_x and rhand_x < (wtimer_pos_x + wtimer_width)
                rhand_y_in_wtimer_y = rhand_y > wtimer_pos_y and rhand_y < (wtimer_pos_y + wtimer_height)
                rhand_in_wtimer = rhand_x_in_wtimer_x and rhand_y_in_wtimer_y

                if (lhand_in_wtimer or rhand_in_wtimer) and not (lhand_in_ptimer or rhand_in_ptimer):
                    cv2.rectangle(img, 
                                (wtimer_pos_x, wtimer_pos_y), 
                                (wtimer_pos_x + wtimer_width, wtimer_pos_y + wtimer_height), 
                                green_clr, 
                                3)

                    # Detect 'tap' on wtimer
                    hand_in_wtimer_curr_time = int(time.time() - hand_in_wtimer_inittime)
                    if hand_in_wtimer_curr_time >= 1 and not wtimer_pressed:

                        wtimer_tapped = True
                        exit_butt_active = True
                        cont_butt_active = True
                        
                        cv2.rectangle(img, 
                                (wtimer_pos_x, wtimer_pos_y), 
                                (wtimer_pos_x + wtimer_width, wtimer_pos_y + wtimer_height), 
                                green_clr, 
                                cv2.FILLED)

                        wtimer_pressed = True
                else:
                    hand_in_wtimer_inittime = time.time()
                    wtimer_pressed = False

            #-------------------------------------------------------------------------------
            # Draw label
            #-------------------------------------------------------------------------------
            # Update the color (to green) with which the label will be written on the image.
            label_clr = red_clr if label != label_next_pose else green_clr
          
          
            cv2.putText(img, label_next_pose, (int((count_bar_x - wtimer_pos_x + wtimer_width) / 2.5), 75),cv2.FONT_HERSHEY_PLAIN, 2, label_clr, 2)


            # Draw pause buttons
            if wtimer_tapped:
                
                # Deactivate wtimer and ptimer buttons
                wtimer_active = False
                ptimer_active = False

                # Exit button
                if exit_butt_active:
                    
                    # Draw exit button
                    exit_butt_pos_x = int((count_bar_x - wtimer_pos_x + wtimer_width) / 4) #int(window_width / 5.5)
                    exit_butt_pos_y = 200
                    exit_butt_width = 300
                    exit_butt_height = 100

                    cv2.rectangle(img, 
                                (exit_butt_pos_x, exit_butt_pos_y), 
                                (exit_butt_pos_x + exit_butt_width, exit_butt_pos_y + exit_butt_height), 
                                black_clr, 
                                3)
                    cv2.rectangle(img, 
                                (exit_butt_pos_x, exit_butt_pos_y), 
                                (exit_butt_pos_x + exit_butt_width, exit_butt_pos_y + exit_butt_height), 
                                black_clr, 
                                cv2.FILLED)
                    
                    # Exit button text
                    cv2.putText(img, 
                                'Exit', 
                                (exit_butt_pos_x + 100, exit_butt_pos_y + 65), 
                                cv2.FONT_HERSHEY_DUPLEX, 
                                1, 
                                white_clr, 
                                2)

                    lhand_x_in_exit_butt_x = lhand_x > exit_butt_pos_x and lhand_x < (exit_butt_pos_x + exit_butt_width)
                    lhand_y_in_exit_butt_y = lhand_y > exit_butt_pos_y and lhand_y < (exit_butt_pos_y + exit_butt_height)
                    lhand_in_exit_butt = lhand_x_in_exit_butt_x and lhand_y_in_exit_butt_y

                    rhand_x_in_exit_butt_x = rhand_x > exit_butt_pos_x and rhand_x < (exit_butt_pos_x + exit_butt_width)
                    rhand_y_in_exit_butt_y = rhand_y > exit_butt_pos_y and rhand_y < (exit_butt_pos_y + exit_butt_height)
                    rhand_in_exit_butt = rhand_x_in_exit_butt_x and rhand_y_in_exit_butt_y

                    if (lhand_in_exit_butt or rhand_in_exit_butt) and not (lhand_in_cont_butt or rhand_in_cont_butt):
                        cv2.rectangle(img, 
                                    (exit_butt_pos_x, exit_butt_pos_y), 
                                    (exit_butt_pos_x + exit_butt_width, exit_butt_pos_y + exit_butt_height), 
                                    green_clr, 
                                    3)

                        # Detect 'tap' on exit_butt
                        hand_in_exit_butt_curr_time = int(time.time() - hand_in_exit_butt_inittime)
                        if hand_in_exit_butt_curr_time >= 1 and not exit_butt_pressed:

                            cv2.rectangle(img, 
                                    (exit_butt_pos_x, exit_butt_pos_y), 
                                    (exit_butt_pos_x + exit_butt_width, exit_butt_pos_y + exit_butt_height), 
                                    green_clr, 
                                    cv2.FILLED)
                            
                            exit_butt_active = False
                            wtimer_tapped = False
                            yoga_active = False
                            workout_active =False
                            yoga_butt_active = True
                            workout_butt_active = True

                            exit_butt_pressed = True

                    else:
                        hand_in_exit_butt_inittime = time.time()
                        exit_butt_pressed = False

                # Cont button
                if cont_butt_active:
                    # Draw continue button
                    cont_butt_pos_x = exit_butt_pos_x + exit_butt_width + 60
                    cont_butt_pos_y = 200
                    cont_butt_width = 300
                    cont_butt_height = 100

                    cv2.rectangle(img, 
                                (cont_butt_pos_x, cont_butt_pos_y), 
                                (cont_butt_pos_x + cont_butt_width, cont_butt_pos_y + cont_butt_height), 
                                black_clr, 
                                3)
                    cv2.rectangle(img, 
                                (cont_butt_pos_x, cont_butt_pos_y), 
                                (cont_butt_pos_x + cont_butt_width, cont_butt_pos_y + cont_butt_height), 
                                black_clr, 
                                cv2.FILLED)
                    
                    # Exit button text
                    cv2.putText(img, 'Continue', (cont_butt_pos_x + 80, cont_butt_pos_y + 65), cv2.FONT_HERSHEY_DUPLEX, 1, 
                                white_clr, 2)

                    lhand_x_in_cont_butt_x = lhand_x > cont_butt_pos_x and lhand_x < (cont_butt_pos_x + cont_butt_width)
                    lhand_y_in_cont_butt_y = lhand_y > cont_butt_pos_y and lhand_y < (cont_butt_pos_y + cont_butt_height)
                    lhand_in_cont_butt = lhand_x_in_cont_butt_x and lhand_y_in_cont_butt_y

                    rhand_x_in_cont_butt_x = rhand_x > cont_butt_pos_x and rhand_x < (cont_butt_pos_x + cont_butt_width)
                    rhand_y_in_cont_butt_y = rhand_y > cont_butt_pos_y and rhand_y < (cont_butt_pos_y + cont_butt_height)
                    rhand_in_cont_butt = rhand_x_in_cont_butt_x and rhand_y_in_cont_butt_y

                    if (lhand_in_cont_butt or rhand_in_cont_butt) and not (lhand_in_exit_butt or rhand_in_exit_butt):
                        cv2.rectangle(img, 
                                    (cont_butt_pos_x, cont_butt_pos_y), 
                                    (cont_butt_pos_x + cont_butt_width, cont_butt_pos_y + cont_butt_height), 
                                    green_clr, 
                                    3)

                        # Detect 'tap' on cont_butt
                        hand_in_cont_butt_curr_time = int(time.time() - hand_in_cont_butt_inittime)
                        if hand_in_cont_butt_curr_time >= 1 and not cont_butt_pressed:
                            
                            cont_butt_active = False
                            wtimer_tapped = False
                            wtimer_active = True
                            ptimer_active = True

                            cv2.rectangle(img, 
                                    (cont_butt_pos_x, cont_butt_pos_y), 
                                    (cont_butt_pos_x + cont_butt_width, cont_butt_pos_y + cont_butt_height), 
                                    green_clr, 
                                    cv2.FILLED)

                            cont_butt_pressed = True

                    else:
                        hand_in_cont_butt_inittime = time.time()
                        cont_butt_pressed = False


            # Ptimer
            if ptimer_active:
                lhand_x_in_ptimer_x = lhand_x > count_bar_x and lhand_x < (count_bar_x + count_bar_weight)
                lhand_y_in_ptimer_y = lhand_y > count_bar_y and lhand_y < (count_bar_y + count_bar_height)
                lhand_in_ptimer = lhand_x_in_ptimer_x and lhand_y_in_ptimer_y

                rhand_x_in_ptimer_x = rhand_x > count_bar_x and rhand_x < (count_bar_x + count_bar_weight)
                rhand_y_in_ptimer_y = rhand_y > count_bar_y and rhand_y < (count_bar_y + count_bar_height)
                rhand_in_ptimer = rhand_x_in_ptimer_x and rhand_y_in_ptimer_y

                if (lhand_in_ptimer or rhand_in_ptimer) and not (lhand_in_wtimer or rhand_in_wtimer):
                    cv2.rectangle(img, 
                                (count_bar_x, count_bar_y), 
                                (count_bar_x + count_bar_weight, count_bar_y + count_bar_height), 
                                green_clr, 
                                3)

                    # Detect 'tap' on ptimer
                    hand_in_ptimer_curr_time = int(time.time() - hand_in_ptimer_inittime)
                    if hand_in_ptimer_curr_time >= 1 and not ptimer_pressed:
                        # Reset count
                        count = 0
                        # Reset scale on button "Next"
                        curr_count_bar_scale = 0
                        # Switch next exersises
                        if label_next_pose == 'Arms':
                            label_next_pose = 'Legs'
                            videoBack_workout = 'Legs_flip.mp4'
                        elif label_next_pose == 'Legs':
                            label_next_pose = 'Gym'
                            videoBack_workout = 'Gym_flip.mp4'
                        elif label_next_pose == 'Gym':
                            label_next_pose = 'Done!'
                
                        # Background video 
                        bg_video_name_workout = os.path.join(dirname, videoBack_workout)
                        capBackground_workout = cv2.VideoCapture(bg_video_name_workout)
                        # capBackground_workout.set(3,window_height)
                        # capBackground_workout.set(4,window_width)  

                        cv2.rectangle(img, 
                                (count_bar_x, count_bar_y), 
                                (count_bar_x + count_bar_weight, count_bar_y + count_bar_height), 
                                green_clr, 
                                cv2.FILLED)

                        ptimer_pressed = True

                else:
                    hand_in_ptimer_inittime = time.time()
                    ptimer_pressed = False



        #---------------------------------------------
        # Menu buttons
        #---------------------------------------------------
        if yoga_butt_active:
            # Draw yoga button
            yoga_butt_pos_x = 10
            yoga_butt_pos_y = 10
            yoga_butt_width = 300
            yoga_butt_height = 100

            cv2.rectangle(img, 
                        (yoga_butt_pos_x, yoga_butt_pos_y), 
                        (yoga_butt_pos_x + yoga_butt_width, yoga_butt_pos_y + yoga_butt_height), 
                        black_clr, 
                        3)
            cv2.rectangle(img, 
                        (yoga_butt_pos_x, yoga_butt_pos_y), 
                        (yoga_butt_pos_x + yoga_butt_width, yoga_butt_pos_y + yoga_butt_height), 
                        black_clr, 
                        cv2.FILLED)
            
            # yoga button text
            cv2.putText(img, 
                        'Yoga', 
                        (yoga_butt_pos_x + 100, yoga_butt_pos_y + 65), 
                        cv2.FONT_HERSHEY_DUPLEX, 
                        1, 
                        white_clr, 
                        2)

            # Tap yoga button
            if yoga_butt_active:
                lhand_x_in_yoga_butt_x = lhand_x > yoga_butt_pos_x and lhand_x < (yoga_butt_pos_x + yoga_butt_width)
                lhand_y_in_yoga_butt_y = lhand_y > yoga_butt_pos_y and lhand_y < (yoga_butt_pos_y + yoga_butt_height)
                lhand_in_yoga_butt = lhand_x_in_yoga_butt_x and lhand_y_in_yoga_butt_y

                rhand_x_in_yoga_butt_x = rhand_x > yoga_butt_pos_x and rhand_x < (yoga_butt_pos_x + yoga_butt_width)
                rhand_y_in_yoga_butt_y = rhand_y > yoga_butt_pos_y and rhand_y < (yoga_butt_pos_y + yoga_butt_height)
                rhand_in_yoga_butt = rhand_x_in_yoga_butt_x and rhand_y_in_yoga_butt_y

                if (lhand_in_yoga_butt or rhand_in_yoga_butt) and not (lhand_in_workout_butt or rhand_in_workout_butt):
                    cv2.rectangle(img, 
                                (yoga_butt_pos_x, yoga_butt_pos_y), 
                                (yoga_butt_pos_x + yoga_butt_width, yoga_butt_pos_y + yoga_butt_height), 
                                green_clr, 
                                3)

                    # Detect 'tap' on yoga_butt
                    hand_in_yoga_butt_curr_time = int(time.time() - hand_in_yoga_butt_inittime)
                    if hand_in_yoga_butt_curr_time >= 1 and not yoga_butt_pressed:
                        
                        # Reset to yoga
                        hand_in_wtimer_inittime = 1000000000000
                        hand_in_ptimer_inittime = 1000000000000

                        label_next_pose = 'T Pose'
                        videoBack_yoga = 'T_pose_flip.mp4'
                        bg_video_name_yoga = os.path.join(dirname, videoBack_yoga)
                        capBackground_yoga = cv2.VideoCapture(bg_video_name_yoga)
                        # capBackground_yoga.set(3,window_height)
                        # capBackground_yoga.set(4,window_width)

                        wtimer_left = tng_duration
                        wtimer_text_size = 2
                        wtimer_curr_rec_width = 0
                        winit_time = time.time()
                        prev_wcurr_time = 0
                        wmins, wsecs = divmod(wtimer_left, 60)
                        wtimer = '{:02d}:{:02d}'.format(wmins, wsecs)

                        ptimer_left = pose_duration
                        ptimer_curr_rec_width = 0
                        pinit_time = time.time()
                        prev_pcurr_time = 0

                        wtimer_tapped = False
                        wtimer_active = True
                        wtimer_show = True
                        ptimer_active = True
                        ptimer_show = True

                        exit_butt_active = False
                        cont_butt_active = False

                        yoga_butt_active = False
                        workout_butt_active = False

                        yoga_active = True
                        workout_active = False
                        
                        yoga_butt_pressed = True

                        cv2.rectangle(img, 
                                (yoga_butt_pos_x, yoga_butt_pos_y), 
                                (yoga_butt_pos_x + yoga_butt_width, yoga_butt_pos_y + yoga_butt_height), 
                                green_clr, 
                                cv2.FILLED)

                else:
                    hand_in_yoga_butt_inittime = time.time()
                    yoga_butt_pressed = False

        if workout_butt_active:
            # Draw workout button
            workout_butt_pos_y = 10
            workout_butt_width = 300
            workout_butt_height = 100
            #-----------------------------------------------------------
            workout_butt_pos_x = window_width - workout_butt_width - 20
            #-----------------------------------------------------------

            cv2.rectangle(img, 
                        (workout_butt_pos_x, workout_butt_pos_y), 
                        (workout_butt_pos_x + workout_butt_width, workout_butt_pos_y + workout_butt_height), 
                        black_clr, 
                        3)
            cv2.rectangle(img, 
                        (workout_butt_pos_x, workout_butt_pos_y), 
                        (workout_butt_pos_x + workout_butt_width, workout_butt_pos_y + workout_butt_height), 
                        black_clr, 
                        cv2.FILLED)
            
            # workout button text
            cv2.putText(img, 
                        'Workout', 
                        (workout_butt_pos_x + 100, workout_butt_pos_y + 65), 
                        cv2.FONT_HERSHEY_DUPLEX, 
                        1, 
                        white_clr, 
                        2)

            # Tap workout button
            if workout_butt_active:
                lhand_x_in_workout_butt_x = lhand_x > workout_butt_pos_x and lhand_x < (workout_butt_pos_x + workout_butt_width)
                lhand_y_in_workout_butt_y = lhand_y > workout_butt_pos_y and lhand_y < (workout_butt_pos_y + workout_butt_height)
                lhand_in_workout_butt = lhand_x_in_workout_butt_x and lhand_y_in_workout_butt_y

                rhand_x_in_workout_butt_x = rhand_x > workout_butt_pos_x and rhand_x < (workout_butt_pos_x + workout_butt_width)
                rhand_y_in_workout_butt_y = rhand_y > workout_butt_pos_y and rhand_y < (workout_butt_pos_y + workout_butt_height)
                rhand_in_workout_butt = rhand_x_in_workout_butt_x and rhand_y_in_workout_butt_y

                if (lhand_in_workout_butt or rhand_in_workout_butt) and not (lhand_in_yoga_butt or rhand_in_yoga_butt):
                    cv2.rectangle(img, 
                                (workout_butt_pos_x, workout_butt_pos_y), 
                                (workout_butt_pos_x + workout_butt_width, workout_butt_pos_y + workout_butt_height), 
                                green_clr, 
                                3)

                    # Detect 'tap' on workout_butt
                    hand_in_workout_butt_curr_time = int(time.time() - hand_in_workout_butt_inittime)
                    if hand_in_workout_butt_curr_time >= 1 and not workout_butt_pressed:
                        
                        # Reset to workout
                        hand_in_wtimer_inittime = 1000000000000
                        hand_in_ptimer_inittime = 1000000000000

                        #-----------------------------------------------------------
                        label_next_pose = 'Arms'
                        videoBack_workout = 'Arms_flip.mp4'
                        bg_video_name_workout = os.path.join(dirname, videoBack_workout)
                        capBackground_workout = cv2.VideoCapture(bg_video_name_workout)
                        # capBackground_workout.set(3,window_height)
                        # capBackground_workout.set(4,window_width)
                        #-----------------------------------------------------------

                        wtimer_left = tng_duration
                        wtimer_text_size = 2
                        wtimer_curr_rec_width = 0
                        winit_time = time.time()
                        # prev_wcurr_time = 0
                        wmins, wsecs = divmod(wtimer_left, 60)
                        wtimer = '{:02d}:{:02d}'.format(wmins, wsecs)

                        #-----------------------------------------------------------
                        # ptimer_left = pose_duration
                        # ptimer_curr_rec_width = 0
                        # pinit_time = time.time()
                        # prev_pcurr_time = 0
                        count = 0
                        curr_count_bar_scale = 0
                        #-----------------------------------------------------------
                        
                        wtimer_tapped = False
                        wtimer_active = True
                        wtimer_show = True
                        ptimer_active = True
                        ptimer_show = True

                        exit_butt_active = False
                        cont_butt_active = False

                        yoga_butt_active = False
                        workout_butt_active = False

                        yoga_active = False
                        workout_active = True

                        cv2.rectangle(img, 
                                (workout_butt_pos_x, workout_butt_pos_y), 
                                (workout_butt_pos_x + workout_butt_width, workout_butt_pos_y + workout_butt_height), 
                                green_clr, 
                                cv2.FILLED)

                        workout_butt_pressed = True

                else:
                    hand_in_workout_butt_inittime = time.time()
                    workout_butt_pressed = False
    
        

    # else:
    #     img = black_matrix


    # # Draw framerate
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime
    # cv2.putText(img, str(int(fps)), (50, 500), cv2.FONT_HERSHEY_PLAIN, 5, blue_clr, 5)


    cv2.imshow("Fizruk", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()