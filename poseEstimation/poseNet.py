#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#


"""
api reference:
https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.html

jetson-inference (package)
https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html

jetson-utils (package)
https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.utils.html
https://github.com/dusty-nv/jetson-utils/tree/master
https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md

"""


"""
## jetson inference builtins.object

struct ObjectPose
	{		
		uint32_t ID;	/**< Object ID in the image frame, starting with 0 */
		
		float Left;	/**< Bounding box left, as determined by the left-most keypoint in the pose */
		float Right;	/**< Bounding box right, as determined by the right-most keypoint in the pose */
		float Top;	/**< Bounding box top, as determined by the top-most keypoint in the pose */
		float Bottom;	/**< Bounding box bottom, as determined by the bottom-most keypoint in the pose */
		
		/**
		 * A keypoint or joint in the topology. A link is formed between two keypoints.
		 */
		struct Keypoint
		{
			uint32_t ID;	/**< Type ID of the keypoint - the name can be retrieved with poseNet::GetKeypointName() */
			float x;		/**< The x coordinate of the keypoint */
			float y;		/**< The y coordinate of the keypoint */
		};
		
		std::vector<Keypoint> Keypoints;			/**< List of keypoints in the object, which contain the keypoint ID and x/y coordinates */
		std::vector<std::array<uint32_t, 2>> Links;	/**< List of links in the object.  Each link has two keypoint indexes into the Keypoints list */

		/**< Find a keypoint index by it's ID, or return -1 if not found.  This returns an index into the Keypoints list */
		inline int FindKeypoint(uint32_t id) const;         
		
		/**< Find a link index by two keypoint ID's, or return -1 if not found.  This returns an index into the Links list */
		inline int FindLink(uint32_t a, uint32_t b) const;  
	};
"""

"""
Num of Keypoints : 18

topology_keypoint = {
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck"
};
"""



import sys
import argparse
import requests
from PIL import Image
from datetime import datetime
import requests
import json
import threading
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, cudaDrawRect, cudaFont, cudaToNumpy


# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


def detect_fall(pose):
    global detection_flag
    global wait_flag
    global bias

    x_diff = abs(pose.Left -  pose.Right)
    y_diff = abs(pose.Top - pose.Bottom)
            
    if x_diff > y_diff * bias:
        cudaDrawRect(img, (pose.Left, pose.Top, pose.Right, pose.Bottom), line_color=(0, 75, 255, 200))
        detection_flag = True
        wait_flag = True # deactivate the function in few seconds 
    else :
        detection_flag = False

def saveImage(img, format="JPEG"):
    file_name = datetime.now().strftime('%Y_%m_%dT%H:%M:%S') + ".jpeg"
    img_array = cudaToNumpy(img)
    pil_image = Image.fromarray(img_array, 'RGB')
    image_path = "detected/" + file_name
    pil_image.save(image_path, format=format)
    
    return image_path, file_name

################################
### HTTP POST REQUESTS START ###
def printResponse(response):
    print("Status Code : ", response.status_code)
    try:
         response_json = response.json()
         print("Response JSON: ", response_json)
    except ValueError:
         print("Response content: ", response.text)

def sendImage(file_path):
    with open(file_path, 'rb') as file:
        data = MultipartEncoder(
            fields={
                'file': (os.path.basename(file_path), file, 'image/jpeg')
            }
        )
        headers = {
            "Content-Type": data.content_type
        }
        response = requests.post("http://118.219.42.214:8080/api/image/upload", headers=headers, data=data)
        printResponse(response)

def getFcm(title, body):
    fcm = {
        'token' : 'eb9PJybxQ66VOlakaltS42:APA91bHEQpdDaUS1LRxZoMl701oYkN4ntF7uNdDfN0C_mSfe1CO4TnR-wEpa19ofi_RhmkG_Ew90FfRoBTMoW9jEgxvlev3DT0iC2D-x1ZzwjJEKlJYbzpdp4TaRvRPbLPtMhUAWHoav',
        'title' : title,
        'body' : body
    }
    
    return fcm
    
def getLog(content, camId=1, fileName="/path/to/image.jpg"):
    log = {
        "camId": camId,
        "content": content,
        "imagePath": fileName, 
        "createdAt": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        "isChecked": False
    }
    
    return log

def getLocation(loc):
    location = {
        "location" : loc
    }
    return location

def sendHttpPost(url, data):
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    printResponse(response)

def sendFcm(fcm):
    sendHttpPost("http://118.219.42.214:8080/api/fcm/send", fcm)

def sendLog(log):
    sendHttpPost("http://118.219.42.214:8080/api/logs", log)

def sendLocation(loc):
    sendHttpPost("http://118.219.42.214:8080/api/cameras", loc)
    
def sendFallDetection(img): #thread
    # save the image to a file and send it to the server
    imgPath, fileName = saveImage(img)
    sendImage(imgPath)
                
    # send a log to the server
    log = getLog("넘어짐 감지됨", fileName=fileName)
    sendLog(log)
                
    # send a FCM to the user
    fcm = getFcm("위험 상황이 감지되었습니다", "환자가 넘어졌습니다 on Jetson TX2")
    sendFcm(fcm)
                
### HTTP POST REQUESTS END ###
##############################

##############################
### MAIN ###
# send location data to the server
location = getLocation("디지털관 2층 1번 카메라")
sendLocation(location)

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
camera = videoSource(args.input, argv=sys.argv)
display = videoOutput(args.output, argv=sys.argv)
font = cudaFont() # font object to print texts on the screen

# initialize global variables
fall_count = 0
wait_count = 0
wait_flag = False
detection_flag = False
bias = 1.12 

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = camera.Capture()

    # check captured image format 
    # print("IMG format :  ",img.format) # format:  rgb8

    if img is None: # timeout
        continue  

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    # reset the wait_flag to activate detect_fall() 
    if wait_count > 18:
          wait_flag = False
          wait_count = 0

    # check each pose whether it has fallen
    for pose in poses:        
            nose_id = pose.FindKeypoint('nose')
            neck_id = pose.FindKeypoint('neck')
            r_ankle_id = pose.FindKeypoint('right_ankle')
            l_ankle_id = pose.FindKeypoint('left_ankle')

            # put the pose's ID on the screen above it's head.
            if nose_id >= 0:
                  nose = pose.Keypoints[nose_id]
                  font.OverlayText(img, 
                                    text=f"{pose.ID}",
                                    x=int(nose.x),
                                    y=int(nose.y -15),
                                    color=font.White)

            # if neck_id < 0:
            #       continue
                  
            # skip the pose when ankles aren't detected
            if r_ankle_id + l_ankle_id < 0:
                continue   

            # pause the detector function in seconds when the image has posted
            if wait_flag:
                continue

            detect_fall(pose)

            if detection_flag :
                fall_count += 1
                
                postThread = threading.Thread(target=sendFallDetection, args=(img,))
                postThread.start()
                

    # deactivate detect_fall() function until the wait_flag becomes false
    if wait_flag:
          wait_count += 1
    
    # print fall count on the screen
    font.OverlayText(img, text=f"Fall Count: {fall_count}", 
                         x=5, y=5,
                         color=font.White, background=font.Gray40)
            
            

    # render the image
    display.Render(img)

    # update the title bar
    display.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    # net.PrintProfilerTimes()

    # exit on input/output EOS
    if not camera.IsStreaming() or not display.IsStreaming():
        break