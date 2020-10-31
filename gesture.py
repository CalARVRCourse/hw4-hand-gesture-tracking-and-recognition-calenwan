#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:11:33 2020

@author: kailunwan, yandali, yizhouwang
"""

import cv2
import numpy as np
import pyautogui

# global variables
window_name = 'gesture detection'
max_binary_value = 255
previous_image_value = 0
previous_ROI = 0
# original image
image_type = '0: original image \n 1: part 1 \n 2: part 2 \n 3: part 3'
image_value = 0
image_size = 3
# part 1
HSV_lower_type = ['H_lower:', 'S_lower:', 'V_lower:']
HSV_upper_type = ['H_upper:', 'S_upper:', 'V_upper:']
YCbCr_lower_type = ['Y_lower:', 'Cb_lower:', 'Cr_lower:']
YCbCr_upper_type = ['Y_upper:', 'Cb_upper:', 'Cr_upper:'] 
lower_HSV_value = [0, 40, 0]
upper_HSV_value = [25, 255, 255]
lower_YCrCb_value = [0, 138, 67]
upper_YCrCb_value = [255, 173, 133]
# part 2
binarization_type = 'OTSU or TRIANGLE:'
binarization_value = cv2.THRESH_OTSU
binarization_tracker_value = 0
# part 3
defects_type = 'defects angle 60 or 90:'
defects_value = np.pi / 3
defects_tracker_value = 0
# part 4 (part 2 + part 3)
timeWindow_type = 'time window (# frames):'
timeWindow_size = 30
timeWindow_value = 1 # frames in time window per second
threshold_type = 'threshold for comparsion:'
threshold_size = 100
threshold_value = 10 # change compare to previous time
machine_type = 'windows/mac:'
machine_value = 0
gesture_type = 'simple/complex gestures:'
gesture_value = 0

# [area1, X1, Y1, x1, y1, MA1, ma1, angle1, hasArea1, area2, X2, Y2, x2, y2, MA2, ma2, angle2, hasArea2, fingerCount1, cX, cY, hasFinger1, fingerCount2, cX2, cY2, hasFinger2]
num_attr = 26
attribute_dict = {'area1':0, 'X1':1, 'Y1': 2, 'x1':3, 'y1':4, 'MA1':5, 'ma1':6, 'angle1':7, 'hasArea1':8, 'area2':9, 'X2':10, 'Y2':11, 'x2':12, 'y2':13, 'MA2':14, 'ma2':15, 'angle2':16, 'hasArea2':17, 'fingerCount1':18, 'cX1':19, 'cY1':20, 'hasFinger1':21, 'fingerCount2':22, 'cX2':23, 'cY2':24, 'hasFinger2': 25}
# current variables
current_attr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# previous variables
previous_attr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def nothing(x):
    pass

# set attributes value from tracker
def SetAttributesValue(image_value, previous_ROI, current_ROI):
    global previous_image_value
    global lower_HSV_value
    global upper_HSV_value
    global lower_YCrCb_value
    global upper_YCrCb_value
    global binarization_value
    global binarization_tracker_value
    global defects_value
    global defects_tracker_value
    global timeWindow_value
    global threshold_value
    global machine_value
    global gesture_value
    
    if image_value == 1:
        # for part 1, show image type trackbar, and HSV/YCbCr/kernal/GaussianBlur trackbar
        if previous_image_value == image_value:
            # if previous image type is 1, read values from trackbar and update
                
            # get those info
            HSV_BL = cv2.getTrackbarPos(HSV_lower_type[0], window_name)
            HSV_GL = cv2.getTrackbarPos(HSV_lower_type[1], window_name)
            HSV_RL = cv2.getTrackbarPos(HSV_lower_type[2], window_name)
            HSV_BU = cv2.getTrackbarPos(HSV_upper_type[0], window_name)
            HSV_GU = cv2.getTrackbarPos(HSV_upper_type[1], window_name)
            HSV_RU = cv2.getTrackbarPos(HSV_upper_type[2], window_name)
            YCbCr_BL = cv2.getTrackbarPos(YCbCr_lower_type[0], window_name)
            YCbCr_GL = cv2.getTrackbarPos(YCbCr_lower_type[1], window_name)
            YCbCr_RL = cv2.getTrackbarPos(YCbCr_lower_type[2], window_name)
            YCbCr_BU = cv2.getTrackbarPos(YCbCr_upper_type[0], window_name)
            YCbCr_GU = cv2.getTrackbarPos(YCbCr_upper_type[1], window_name)
            YCbCr_RU = cv2.getTrackbarPos(YCbCr_upper_type[2], window_name)
        
            # set new values for global variables
            lower_HSV_value = [HSV_BL, HSV_GL, HSV_RL]
            upper_HSV_value = [HSV_BU, HSV_GU, HSV_RU]
            lower_YCrCb_value = [YCbCr_BL, YCbCr_GL, YCbCr_RL]
            upper_YCrCb_value = [YCbCr_BU, YCbCr_GU, YCbCr_RU]
        
        else:
            # rebuild image window with new values if image type changes
            # check previous frame to destroy related windows
            if previous_image_value == 2:
                if previous_ROI == 1:
                    cv2.destroyWindow('ROI 2')
                elif previous_ROI == 2:
                    cv2.destroyWindow('ROI 2')
                    cv2.destroyWindow('ROI 3')
            elif previous_image_value == 3:
                cv2.destroyWindow(window_name+" 3a")
            cv2.destroyWindow(window_name)
            cv2.namedWindow(window_name)
            cv2.createTrackbar(image_type, window_name, image_value, image_size, nothing)
            # create other trackers 
            cv2.createTrackbar(HSV_lower_type[0], window_name, lower_HSV_value[0], max_binary_value, nothing)
            cv2.createTrackbar(HSV_lower_type[1], window_name, lower_HSV_value[1], max_binary_value, nothing)
            cv2.createTrackbar(HSV_lower_type[2], window_name, lower_HSV_value[2], max_binary_value, nothing)
            cv2.createTrackbar(HSV_upper_type[0], window_name, upper_HSV_value[0], max_binary_value, nothing)
            cv2.createTrackbar(HSV_upper_type[1], window_name, upper_HSV_value[1], max_binary_value, nothing)
            cv2.createTrackbar(HSV_upper_type[2], window_name, upper_HSV_value[2], max_binary_value, nothing)
            cv2.createTrackbar(YCbCr_lower_type[0], window_name, lower_YCrCb_value[0], max_binary_value, nothing)
            cv2.createTrackbar(YCbCr_lower_type[1], window_name, lower_YCrCb_value[1], max_binary_value, nothing)
            cv2.createTrackbar(YCbCr_lower_type[2], window_name, lower_YCrCb_value[2], max_binary_value, nothing)
            cv2.createTrackbar(YCbCr_upper_type[0], window_name, upper_YCrCb_value[0], max_binary_value, nothing)
            cv2.createTrackbar(YCbCr_upper_type[1], window_name, upper_YCrCb_value[1], max_binary_value, nothing)
            cv2.createTrackbar(YCbCr_upper_type[2], window_name, upper_YCrCb_value[2], max_binary_value, nothing)
    
    elif image_value == 2:
        # for part 2, show image type trackbar, and OTSU or TRIANGLE binarization trackbar
        if previous_image_value == image_value:
            # if previous image type is 2, read values from trackbar and update

            # get info
            binarization_tracker_value = cv2.getTrackbarPos(binarization_type, window_name)
            if binarization_tracker_value == 0:
                binarization_value = cv2.THRESH_OTSU
            else:
                binarization_value = cv2.THRESH_TRIANGLE    
            timeWindow_value = cv2.getTrackbarPos(timeWindow_type, window_name)
            threshold_value = cv2.getTrackbarPos(threshold_type, window_name)
            machine_value = cv2.getTrackbarPos(machine_type, window_name)
            gesture_value = cv2.getTrackbarPos(gesture_type, window_name)
            if timeWindow_value == 0:
                timeWindow_value = 1
            
            # if no hand detected, destory ROI window
            if current_ROI == 0:
                if previous_ROI == 1:
                    cv2.destroyWindow('ROI 2')
                elif previous_ROI == 2:
                    cv2.destroyWindow('ROI 2')
                    cv2.destroyWindow('ROI 3')
            elif current_ROI == 1:
                if previous_ROI == 2:
                    cv2.destroyWindow('ROI 3')
        else:           
            # rebuild image window with new values
            # check previous frame to destroy related windows
            if previous_image_value == 1:
                cv2.destroyWindow(window_name+" 1")
            elif previous_image_value == 3:
                cv2.destroyWindow(window_name+" 3a")
            cv2.destroyWindow(window_name)
            cv2.namedWindow(window_name)
            cv2.createTrackbar(image_type, window_name, image_value, image_size, nothing)
            # create other trackers 
            cv2.createTrackbar(binarization_type, window_name, binarization_tracker_value, 1, nothing)
            # for part 2, also add tracker for average time window and threshold for comparsion
            cv2.createTrackbar(timeWindow_type, window_name, timeWindow_value, timeWindow_size, nothing)
            cv2.createTrackbar(threshold_type, window_name, threshold_value, threshold_size, nothing)
            # for part 4, also add tracker for machine type (windows/mac) and gesture type (simple gestures or complex gestures)
            cv2.createTrackbar(machine_type, window_name, machine_value, 1, nothing)
            cv2.createTrackbar(gesture_type, window_name, gesture_value, 1, nothing)
        
    elif image_value == 3:
        # for part 3, show image type trackbar, and 60 or 90 degree choice for defects trackbar
        if previous_image_value == image_value:
            # if previous image type is 3, read values from trackbar and update
            
            # get info
            defects_tracker_value = cv2.getTrackbarPos(defects_type, window_name)
            if defects_tracker_value == 0:
                defects_value = np.pi / 3
            else:
                defects_value = np.pi / 2  
            timeWindow_value = cv2.getTrackbarPos(timeWindow_type, window_name)
            threshold_value = cv2.getTrackbarPos(threshold_type, window_name)
            machine_value = cv2.getTrackbarPos(machine_type, window_name)
            gesture_value = cv2.getTrackbarPos(gesture_type, window_name)
            if timeWindow_value == 0:
                timeWindow_value = 1
                
        else:         
            # rebuild image window with new values
            # check previous frame to destroy related windows
            if previous_image_value == 1:
                cv2.destroyWindow(window_name+" 1")
            elif previous_image_value == 2:
                if previous_ROI == 1:
                    cv2.destroyWindow('ROI 2')
                elif previous_ROI == 2:
                    cv2.destroyWindow('ROI 2')
                    cv2.destroyWindow('ROI 3')
            cv2.destroyWindow(window_name)
            cv2.namedWindow(window_name)
            cv2.createTrackbar(image_type, window_name, image_value, image_size, nothing)                
            # create other trackers 
            cv2.createTrackbar(defects_type, window_name, defects_tracker_value, 1, nothing)
            # for part 3, also add tracker for average time window and threshold for comparsion
            cv2.createTrackbar(timeWindow_type, window_name, timeWindow_value, timeWindow_size, nothing)
            cv2.createTrackbar(threshold_type, window_name, threshold_value, threshold_size, nothing)            
            # for part 4, also add tracker for machine type (windows/mac) and gesture type (simple gestures or complex gestures)
            cv2.createTrackbar(machine_type, window_name, machine_value, 1, nothing)
            cv2.createTrackbar(gesture_type, window_name, gesture_value, 1, nothing)
            
    else:
        if previous_image_value != image_value:
            # for original image, only show image type trackbar
            # check previous frame to destroy related windows
            if previous_image_value == 1:
                cv2.destroyWindow(window_name+" 1")
            elif previous_image_value == 2:
                if previous_ROI == 1:
                    cv2.destroyWindow('ROI 2')
                elif previous_ROI == 2:
                    cv2.destroyWindow('ROI 2')
                    cv2.destroyWindow('ROI 3')
            elif previous_image_value == 3:
                cv2.destroyWindow(window_name+" 3a")
            cv2.destroyWindow(window_name)
            cv2.namedWindow(window_name)
            cv2.createTrackbar(image_type, window_name, image_value, image_size, nothing)
        
    previous_image_value = image_value

def CheckOneHandOrTwoHands(frame):
    # threshold and binarize the image using triangle binarization
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+binarization_value )
    thresh1 = thresh.copy() # copy image, not use reference
    thresh2 = thresh.copy() # copy image, not use reference
    
    # then, check if one hand or two hands
    results = np.all(thresh, axis=0)
    # get first and last column with hands
    rHands = np.where(results == 0)[0]
    numHands = 1
    
    if len(rHands) > 2: # has at least one hand
        # limit the results to the first and last column which have hand characteristics
        # and find if any column in between has all background
        # if so, then we have two hands
        results2 = np.nonzero(results[rHands[0]:rHands[-1]+1])[0]
        if len(results2) > 0:
            # two hands
            # separate thresh into two images, where each image contain only one hand
            numHands = 2
            thresh1[:,results2[0]:] = max_binary_value
            thresh2[:,:results2[0]] = max_binary_value
    # otherwise, do nothing, and regard it as one hand
    return numHands, thresh, thresh1, thresh2
    
# Part 1: Extracting Hand from the feed
def HandExtraction(frame):    
    # HSV and YCbCr color space to segment the skin
    lower_HSV = np.array(lower_HSV_value, dtype = "uint8")    
    upper_HSV = np.array(upper_HSV_value, dtype = "uint8")  
  
    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)  
  
    lower_YCrCb = np.array(lower_YCrCb_value, dtype = "uint8")
    upper_YCrCb = np.array(upper_YCrCb_value, dtype = "uint8")  
      
    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)  
  
    skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb)
    
    # apply erosion and dilation and apply standard guassian blur 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)  
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)  
  
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) 
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    """
    cv2.imshow(window_name+str(1), skin)
    k = cv2.waitKey(1) #k is the key pressed
    """
    return skin
   
# Part 2: Processing the hand image with connected component analysis (one hand or two hands)
def HandConnectedComponent(numHands, frame, frame1, frame2):
    global current_attr
    # apply connected component analysis to image
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(frame,ltype=cv2.CV_16U)
    
    # way 1 (force change the background to orange)
    '''
    markers = np.array(markers, dtype=np.uint8)  
    label_hue = np.uint8(179*markers/np.max(markers))  
    # blank_ch = 255*np.ones_like(label_hue)
    green_channel = 165*markers
    red_channel = 255*markers
    labeled_img = cv2.merge([label_hue, green_channel, red_channel])
    '''
    
    # way 2 (code snippet from instruction)
    markers = np.array(markers, dtype=np.uint8)  
    label_hue = np.uint8(179*markers/np.max(markers))  
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    
    subImg1 = labeled_img
    subImg2 = labeled_img
    isValid = 0 

    if numHands == 2: # if two hands, reset stats for frame1
        ret, markers, stats, centroids = cv2.connectedComponentsWithStats(frame1,ltype=cv2.CV_16U)
    if (ret>2):  
        try:
            statsSortedByArea = stats[np.argsort(stats[:, 4])]
            roi = statsSortedByArea[-3][0:4]    
            x, y, w, h = roi
            subImg = labeled_img[y:y+h, x:x+w]  
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY)        
            _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            maxCntLength = 0 
            # print(len(contours))
            for i in range(0,len(contours)):  
                cntLength = len(contours[i])  
                if(cntLength>maxCntLength):  
                    cnt = contours[i]  
                    maxCntLength = cntLength
            if(maxCntLength>=5):
                ellipseParam = cv2.fitEllipse(cnt)  
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB)                 
                subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)
                (x1,y1),(MA,ma),angle = ellipseParam
                print('(x,y) = (' + str(x1) + ',' + str(y1) + '), (MA,ma) = (' + str(MA) + ',' + str(ma) + '), angle = ' + str(angle))               
                
            subImg = cv2.resize(subImg, (0,0), fx=3, fy=3) 
            subImg1 = subImg.copy()
            
            # if valid, store info
            current_attr[attribute_dict['area1']] += w * h
            current_attr[attribute_dict['x1']] += x1
            current_attr[attribute_dict['y1']] += y1
            current_attr[attribute_dict['X1']] += (x + w/2)
            current_attr[attribute_dict['Y1']] += (y + h/2)
            current_attr[attribute_dict['MA1']] += MA
            current_attr[attribute_dict['ma1']] += ma
            current_attr[attribute_dict['angle1']] += angle
            current_attr[attribute_dict['hasArea1']] += 1
            isValid += 1
        except:  
            print("No hand found")
            
    # get roi for second hand, if exists
    if numHands == 2:
        ret, markers, stats, centroids = cv2.connectedComponentsWithStats(frame2,ltype=cv2.CV_16U)
        if (ret>2):  
            try:
                statsSortedByArea = stats[np.argsort(stats[:, 4])]
                roi2 = statsSortedByArea[-3][0:4]
                x, y, w, h = roi2  
                subImg = labeled_img[y:y+h, x:x+w]  
                subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY)
                _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
                maxCntLength = 0 
                # print(len(contours))
                for i in range(0,len(contours)):  
                    cntLength = len(contours[i])  
                    if(cntLength>maxCntLength):  
                        cnt = contours[i]  
                        maxCntLength = cntLength
                if(maxCntLength>=5):
                    ellipseParam = cv2.fitEllipse(cnt)  
                    subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);                 
                    subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)
                    (x2,y2),(MA2,ma2),angle2 = ellipseParam
                    print('(x2,y2)=(' + str(x2) + ',' + str(y2) + '), (MA2,ma2)=(' + str(MA2) + ',' + str(ma2) + '), angle2=' + str(angle2))
                    
                subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)
                subImg2 = subImg.copy()
                
                # if valid, store info
                current_attr[attribute_dict['area2']] += w * h
                current_attr[attribute_dict['x2']] += x2
                current_attr[attribute_dict['y2']] += y2
                current_attr[attribute_dict['X2']] += (x + w/2)
                current_attr[attribute_dict['Y2']] += (y + h/2)
                current_attr[attribute_dict['MA2']] += MA2
                current_attr[attribute_dict['ma2']] += ma2
                current_attr[attribute_dict['angle2']] += angle2
                current_attr[attribute_dict['hasArea2']] += 1
                isValid += 1
            except:  
                print("No hand found")
    
    return labeled_img, subImg1, subImg2, isValid

# Part 3: Tracking 2D finger positions (one hand or two hands)
def TrackingFingers(numHands, frame):
    global current_attr
    # Part 3a: Processing the hand image with contour and hull analysis
    
    # draw the largest contour and convexity defects
    # first, convert black in white to white in black
    frame_inv = cv2.subtract(255, frame)
    # then, convert this binary image to RGB image format for future use
    thresh_rgb = cv2.merge([frame_inv, frame_inv, frame_inv])
    thresh_rgb2 = cv2.merge([frame_inv, frame_inv, frame_inv])
    fingerCount = 0 # initial finger count number
    fingerCount2 = 0
  
    _, contours, _ = cv2.findContours(frame_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
    contours=sorted(contours,key=cv2.contourArea,reverse=True) 
    if len(contours)>1:  
        largestContour = contours[0]  
        hull = cv2.convexHull(largestContour, returnPoints = False)
        for cnt in contours[:1]:  
            defects = cv2.convexityDefects(cnt,hull)  
            if(not isinstance(defects,type(None))):
                for i in range(defects.shape[0]):  
                    s,e,f,d = defects[i,0]  
                    start = tuple(cnt[s][0])  
                    end = tuple(cnt[e][0])  
                    far = tuple(cnt[f][0])
                    
                    cv2.line(thresh_rgb,start,end,[0,255,0],2)  
                    cv2.circle(thresh_rgb,far,5,[0,0,255],-1)
        # another hand (if exists)
        if numHands == 2:
            largestContour2 = contours[1]  
            hull2 = cv2.convexHull(largestContour2, returnPoints = False)
            for cnt in contours[1:2]:  
                defects = cv2.convexityDefects(cnt,hull2)  
                if(not isinstance(defects,type(None))):
                    for i in range(defects.shape[0]):  
                        s,e,f,d = defects[i,0]  
                        start = tuple(cnt[s][0])  
                        end = tuple(cnt[e][0])  
                        far = tuple(cnt[f][0])
                        
                        cv2.line(thresh_rgb,start,end,[0,255,0],2)  
                        cv2.circle(thresh_rgb,far,5,[0,0,255],-1)
    
    # Part 3b: Detecting fingers in the image (heuristic filtering to eliminate most of the noisy defect points)
    _, contours, _ = cv2.findContours(frame_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
    contours=sorted(contours,key=cv2.contourArea,reverse=True) 
    if len(contours)>1:  
        largestContour = contours[0]  
        hull = cv2.convexHull(largestContour, returnPoints = False)     
        for cnt in contours[:1]:
            defects = cv2.convexityDefects(cnt,hull)
            if(not isinstance(defects,type(None))):
                for i in range(defects.shape[0]):  
                    s,e,f,d = defects[i,0]  
                    start = tuple(cnt[s][0])  
                    end = tuple(cnt[e][0])  
                    far = tuple(cnt[f][0])
                    
                    cv2.line(thresh_rgb2,start,end,[0,255,0],2)
                    
                    c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
                    a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
                    b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
                    angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))
                    
                    if angle <= defects_value:
                        fingerCount += 1
                        cv2.circle(thresh_rgb2,far,5,[0,0,255],-1)
        if fingerCount > 0:
            fingerCount = fingerCount+1
        M = cv2.moments(largestContour)
        cX = 0.0
        cY = 0.0
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])  
            cY = int(M["m01"] / M["m00"])

        # if valid, store info
        current_attr[attribute_dict['fingerCount1']] += fingerCount
        current_attr[attribute_dict['cX1']] += cX
        current_attr[attribute_dict['cY1']] += cY
        current_attr[attribute_dict['hasFinger1']] += 1
        
        # another hand (if exists)
        if numHands == 2:
            largestContour2 = contours[1]  
            hull2 = cv2.convexHull(largestContour2, returnPoints = False)     
            for cnt in contours[1:2]:  
                defects = cv2.convexityDefects(cnt,hull2)  
                if(not isinstance(defects,type(None))):
                    for i in range(defects.shape[0]):  
                        s,e,f,d = defects[i,0]  
                        start = tuple(cnt[s][0])  
                        end = tuple(cnt[e][0])  
                        far = tuple(cnt[f][0])
                        
                        cv2.line(thresh_rgb2,start,end,[0,255,0],2)
                        
                        c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
                        a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
                        b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
                        angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))
                        
                        if angle <= defects_value:
                            fingerCount2 += 1
                            cv2.circle(thresh_rgb2,far,5,[0,0,255],-1)
            if fingerCount2 > 0:
                fingerCount2 = fingerCount2+1
            M = cv2.moments(largestContour2)
            cX = 0
            cY = 0
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])  
                cY = int(M["m01"] / M["m00"])
               
            # if valid, store info
            current_attr[attribute_dict['fingerCount2']] += fingerCount2
            current_attr[attribute_dict['cX2']] += cX
            current_attr[attribute_dict['cY2']] += cY
            current_attr[attribute_dict['hasFinger2']] += 1
            
    text = "number of fingers: " + str(fingerCount+fingerCount2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    thresh_rgb2 = cv2.putText(thresh_rgb2, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    return thresh_rgb, thresh_rgb2

# Part 4: window averaging
def GetAverageCurrentValues():
    # use global variables
    global current_attr

    # after window, take average of those values
    # for first several elements, we need to calcuate the average based on valid area number
    # avoid divide by 0 error
    if current_attr[attribute_dict['hasArea1']] == 0:
        current_attr[attribute_dict['hasArea1']] = 1
    if current_attr[attribute_dict['hasArea2']] == 0:
        current_attr[attribute_dict['hasArea2']] = 1
    if current_attr[attribute_dict['hasFinger1']] == 0:
        current_attr[attribute_dict['hasFinger1']] = 1
    if current_attr[attribute_dict['hasFinger2']] == 0:
        current_attr[attribute_dict['hasFinger2']] = 1
    for i in range(attribute_dict['hasArea1']):
        current_attr[i] /= current_attr[attribute_dict['hasArea1']]
    for i in range(attribute_dict['hasArea1']+1, attribute_dict['hasArea2']):
        current_attr[i] /= current_attr[attribute_dict['hasArea2']]
    for i in range(attribute_dict['hasArea2']+1, attribute_dict['hasFinger1']):
        current_attr[i] /= current_attr[attribute_dict['hasFinger1']]
    for i in range(attribute_dict['hasFinger1']+1, attribute_dict['hasFinger2']):
        current_attr[i] /= current_attr[attribute_dict['hasFinger2']]
    # round finger number to int
    current_attr[attribute_dict['fingerCount1']] = round(current_attr[attribute_dict['fingerCount1']])
    current_attr[attribute_dict['fingerCount2']] = round(current_attr[attribute_dict['fingerCount2']])

# Part 4: apply custom gestures to control mouse and keyboard
def ControlMouseAndKeyboardUsingCustomGestures():
    # use global variables
    global current_attr
    global previous_attr

    winmac = 'ctrl'
    winmac2 = 'alt'
    if machine_value == 1:
        # if mac, change string to 'command'
        winmac = 'command'
        winmac2 = 'option'
    # [area1, x1, y1, MA1, ma1, angle1, hasArea1, area2, x2, y2, MA2, ma2, angle2, hasArea2, fingerCount1, cX1, cY1, hasFinger1, fingerCount2, cX2, cY2, hasFinger2]
    # use these attributes for simple and complex gestures
    
    # for angle, right hand (angle1) points up -> 30 degrees; left hand (angle2) points up -> 150 degrees
    
    if gesture_value == 0:
        # only apply simple gestures       
        
        # example gesture to exhibit (on desktop and in words)
        # finger==0, one hand hull moving
        # angle>120 or angle<60, single right click 
        # finger==2, double click
        
        # finger==4+4, press space (two hands)
        # two roi points up, select all (two hands)
        # one roi is significantly above another one, close window
        
        # part 2 info
        hasArea1 = current_attr[attribute_dict['hasArea1']]
        hasArea2 = current_attr[attribute_dict['hasArea2']]       
        angle1 = current_attr[attribute_dict['angle1']]
        angle2 = current_attr[attribute_dict['angle2']]
        y1 = current_attr[attribute_dict['Y1']]
        y2 = current_attr[attribute_dict['Y2']]
        
        # part 3 info
        hasFinger1 = current_attr[attribute_dict['hasFinger1']]
        hasFinger2 = current_attr[attribute_dict['hasFinger2']]
        finger1 = current_attr[attribute_dict['fingerCount1']]
        finger2 = current_attr[attribute_dict['fingerCount2']]
        cX = current_attr[attribute_dict['cX1']]
        cY = current_attr[attribute_dict['cY1']]
        
        if image_value == 2:
            # gesture 2: angle>120 or angle<60, single right click (one hand)
            if hasArea1 > 1 and hasArea2 == 1 and (angle1 > 120 or angle1 < 60):
                # the ring on hand points either up or down
                pyautogui.rightClick()
        
            # gesture 5: two roi points up, select all (two hands)
            if hasArea1 > 1 and hasArea2 > 1 and angle1 < 60 and angle2 > 120:
                pyautogui.hotkey(winmac, 'a')
            
            # gesture 6: one roi is significantly above another one, close window
            if y1 > y2 + threshold_value or y2 > y1 + threshold_value:
                pyautogui.hotkey(winmac, 'w')
        if image_value == 3:
            # gesture 1: finger==0, one hand hull moving
            if hasFinger1 > 1 and hasFinger2 == 1 and finger1 == 0:
                # one hand position moving, if cX and cY are not out of bound
                mcX = offsetX + scaleX * cX
                mcY = offsetY + scaleY * cY
                if mcX > 0 and mcY > 0 and mcX < monwidth and mcY < monheight:
                    pyautogui.moveTo(cX, cY, duration=0.02, tween=pyautogui.easeInOutQuad)                
                
            # gesture 3: finger==2, double click
            if hasFinger1 > 1 and hasFinger2 == 1 and finger1 == 2:
                pyautogui.doubleClick()
                
            # gesture 4: finger==4+4, press space (two hands)
            if hasFinger1 > 1 and hasFinger2 > 1 and finger1 == 4 and finger2 == 4:
                pyautogui.press('space')
        # etc ...
    elif gesture_value == 1:
        # only apply complex gestures
        
        # example gestures to exhibit (in windows photo app)
        # one roi area decrease and angle does not change too much, save as (ctrl + s)
        # one roi angle changes, rotate (ctrl + r)  
        # finger change from 2 to 4, add current photo to album (ctrl + d)
        # finger keeps 5 and hand move right (wave), back to collection (esc)

        # two roi distance changing, zoom in and out (ctrl + / ctrl -)
        # finger changes from 2+2 to 4+4, file info (alt + enter)        
        # two roi area decrease, go into selection mode (space)
        # two hands wave together (2 fingers each), select this image (enter)

        # part 2 info
        hasArea1 = current_attr[attribute_dict['hasArea1']]
        hasArea2 = current_attr[attribute_dict['hasArea2']]
        area1 = current_attr[attribute_dict['area1']]
        area2 = current_attr[attribute_dict['area2']]
        angle1 = current_attr[attribute_dict['angle1']]
        angle2 = current_attr[attribute_dict['angle2']]
        x1 = current_attr[attribute_dict['X1']]
        y1 = current_attr[attribute_dict['Y1']]
        x2 = current_attr[attribute_dict['X2']]
        y2 = current_attr[attribute_dict['Y2']]
        
        phasArea1 = previous_attr[attribute_dict['hasArea1']]
        phasArea2 = previous_attr[attribute_dict['hasArea2']]
        parea1 = previous_attr[attribute_dict['area1']]
        parea2 = previous_attr[attribute_dict['area2']]
        pangle1 = previous_attr[attribute_dict['angle1']]
        pangle2 = previous_attr[attribute_dict['angle2']]
        px1 = previous_attr[attribute_dict['X1']]
        py1 = previous_attr[attribute_dict['Y1']]
        px2 = previous_attr[attribute_dict['X2']]
        py2 = previous_attr[attribute_dict['Y2']]
        
        # part 3 info
        hasFinger1 = current_attr[attribute_dict['hasFinger1']]
        hasFinger2 = current_attr[attribute_dict['hasFinger2']]
        finger1 = current_attr[attribute_dict['fingerCount1']]
        finger2 = current_attr[attribute_dict['fingerCount2']]
        cX1 = current_attr[attribute_dict['cX1']]
        cY1 = current_attr[attribute_dict['cY1']]
        cX2 = current_attr[attribute_dict['cX2']]
        cY2 = current_attr[attribute_dict['cY2']]
        
        phasFinger1 = previous_attr[attribute_dict['hasFinger1']]
        phasFinger2 = previous_attr[attribute_dict['hasFinger2']]
        pfinger1 = previous_attr[attribute_dict['fingerCount1']]
        pfinger2 = previous_attr[attribute_dict['fingerCount2']]
        pcX1 = previous_attr[attribute_dict['cX1']]
        pcY1 = previous_attr[attribute_dict['cY1']]
        pcX2 = previous_attr[attribute_dict['cX2']]
        pcY2 = previous_attr[attribute_dict['cY2']]
 
        if image_value == 2:
            # gesture 1: one roi area decrease and angle does not change too much, save as (ctrl + s)
            if hasArea1 > 1 and phasArea1 > 1 and hasArea2 == 1 and phasArea2 == 1 and area1 < parea1 - threshold_value*10 and angle1 > pangle1 - threshold_value and angle1 < pangle1 + threshold_value :
                pyautogui.hotkey(winmac, 's')
                
            # gesture 2: one roi angle changes to larger than 120, rotate (ctrl + r)
            if hasArea1 > 1 and phasArea1 > 1 and hasArea2 == 1 and phasArea2 == 1 and (angle1 < pangle1 - threshold_value or angle1 > pangle1 + threshold_value):
                pyautogui.hotkey(winmac, 'r')                
            
            # gesture 5: two roi distance changing, zoom in and out (ctrl + / ctrl -)
            if hasArea1 > 1 and phasArea1 > 1 and hasArea2 > 1 and phasArea2 > 1:
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                pdist = np.sqrt((px1-px2)**2 + (py1-py2)**2)
                if dist > pdist + threshold_value:
                    pyautogui.hotkey(winmac, '+')
                elif dist < pdist - threshold_value:
                    pyautogui.hotkey(winmac, '-')
            
            # gesture 7: two roi area decrease, go into selection mode (space)
            if hasArea1 > 1 and phasArea1 > 1 and hasArea2 > 1 and phasArea2 > 1 and area1 < parea1 - threshold_value and area2 < parea2 - threshold_value:
                pyautogui.press('space')
        elif image_value == 3:
            # gesture 3: finger change from 2 to 4,add current photo to album (ctrl + d)
            if hasFinger1 > 1 and phasFinger1 > 1 and hasFinger2 == 1 and phasFinger2 == 1 and finger1 == 4 and pfinger1 == 2:
                pyautogui.hotkey(winmac, 'd')
            
            # gesture 4: finger keeps 5 and hand move right (wave), back to collection (esc)
            if hasFinger1 > 1 and phasFinger1 > 1 and hasFinger2 == 1 and phasFinger2 == 1 and finger1 == 5 and pfinger1 == 5 and cX1 > pcX1 + threshold_value:
                pyautogui.press('esc')
                
            # gesture 6: finger changes from 4 to 8, file info (alt + enter)
            if hasFinger1 > 1 and phasFinger1 > 1 and hasFinger2 > 1 and phasFinger2 > 1 and finger1 == 4 and pfinger1 == 2 and finger2 == 4 and pfinger2 == 2:
                pyautogui.hotkey(winmac2, 'enter')
                
            # gesture 8: two hands wave together, select this image (enter)
            if hasFinger1 > 1 and phasFinger1 > 1 and hasFinger2 > 1 and phasFinger2 > 1 and finger1 == 2 and pfinger1 == 2 and finger2 == 2 and pfinger2 == 2:
                dist = abs(cX1 - cX2)
                pdist = abs(pcX1 - pcX2)
                if dist < pdist - threshold_value:
                    pyautogui.press('enter')
        # etc ...
    
    # move current_attr to previous_attr
    previous_attr = current_attr


# main  
# get input of monitor resolution
monwidth = 0
monheight = 0
monwidth = int(input("Please provide your screen width resolution (empty for default 1920): ") or '1920')
monheight = int(input("Please provide your screen height resolution (empty for default 1080): ") or '1080')
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 turns OFF auto exp
cam.set(cv2.CAP_PROP_AUTO_WB, 0.25) # 0.25 turns OFF auto WB'
cv2.namedWindow(window_name)
cv2.createTrackbar(image_type, window_name, 0, image_size, nothing)
# get scaleX/scaleY and offsetX/offsetY
camwidth = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
camheight = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
scaleX = monwidth / camwidth
scaleY = monheight / camheight
offsetX = 0
offsetY = 0

while True:  
    needStop = 0
    
    # loop timeWindow_value times for averaging window
    # first, reset current attributes
    current_attr = [0.0] * num_attr
    
    # then, loop timeWindow_value times to calculate average value
    for i in range(timeWindow_value):
        ret, frame = cam.read()
        if not ret:
            needStop = 1
            break

        img_p1 = HandExtraction(frame) # part 1
        numHands, thresh, thresh1, thresh2 = CheckOneHandOrTwoHands(img_p1) # part 1.5 and part 5, check one hand or two hands
        img_p2, img_p2_1, img_p2_2, isValid = HandConnectedComponent(numHands, thresh, thresh1, thresh2) # part 2 (+ part 5)
        img_p3_1, img_p3_2 = TrackingFingers(numHands, thresh) # part 3 (+ part 5)
        
        # get trackbar value
        image_value = cv2.getTrackbarPos(image_type, window_name)
        SetAttributesValue(image_value, previous_ROI, isValid)
        # for different image type, show different image
        k = 0
        if image_value == 1:
            # show part 1            
            cv2.imshow(window_name + " 1", img_p1)
            k = cv2.waitKey(1) #k is the key pressed
            previous_ROI = 0
        elif image_value == 2:
            # show part 2:
            cv2.imshow(window_name, img_p2)
            k = cv2.waitKey(1) #k is the key pressed
            previous_ROI = isValid
            if isValid == 1:
                # one hand, show one separate ROI
                cv2.imshow('ROI 2', img_p2_1)
                k = cv2.waitKey(1) #k is the key pressed
            elif isValid == 2:
                # two hands, show two separate ROIs              
                cv2.imshow('ROI 2', img_p2_1)
                k = cv2.waitKey(1) #k is the key pressed
                cv2.imshow('ROI 3', img_p2_2)
                k = cv2.waitKey(1) #k is the key pressed
        elif image_value == 3:
            # show part 3: (3a and 3b, 3b in main window)
            cv2.imshow(window_name+" 3a", img_p3_1)
            k = cv2.waitKey(1) #k is the key pressed            
            cv2.imshow(window_name, img_p3_2)
            k = cv2.waitKey(1) #k is the key pressed
            previous_ROI = 0
        else:
            # show original frame
            cv2.imshow(window_name, frame)
            k = cv2.waitKey(1) #k is the key pressed
            previous_ROI = 0
        
        if k == 27 or k ==113:  #27, 113 are ascii for escape and q respectively
            needStop = 1
            break
    
    if needStop == 1:
        cv2.destroyAllWindows()
        cam.release()
        break
    # then, use average those values
    GetAverageCurrentValues()
    # and apply pyautogui
    ControlMouseAndKeyboardUsingCustomGestures()