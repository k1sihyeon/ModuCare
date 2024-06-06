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
import numpy as np

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log


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

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
camera = videoSource(args.input, argv=sys.argv)
display = videoOutput(args.output, argv=sys.argv)



def preprocess_keypoint(keypoints):
    if not keypoints:
         return None
    keypoints = np.array(keypoints)
    keypoints -= keypoints[0] # Center the keypoints

    return keypoints


def dect_fall_helper():
        left_ankle_idx = pose.FindKeypoint('left_ankle')
        right_ankle_idx = pose.FindKeypoint('right_ankle')
        neck_idx = pose.FindKeypoint('neck')

        if neck_idx < 0 :
            pass

        ankle_idx = 0
        if left_ankle_idx < 0 and right_ankle_idx < 0:
            pass

        # if left_ankle_idx >= 0 and right_ankle_idx >= 0:
        #     pass

        # left_ankle = pose.Keypoints[left_ankle_idx]
        # right_ankle = pose.Keypoints[right_ankle_idx]
        if left_ankle_idx < 0:
            ankle_idx = right_ankle_idx
        else :
             ankle_idx = left_ankle_idx

        ankle = pose.Keypoinst[ankle_idx]             

        ankle_point_x = ankle.x
        ankle_point_y = ankle.x

        neck = pose.Keypoints[neck_idx]
        neck_point_x = neck.x
        neck_point_y = neck.y



def detect_fall(keypoints):     
    if not keypoints:
        return False
     
    if len(keypoints) < 17:
        pass
        # dect_fall_helper() # unfinished function         
     
    # upper_body_keypoints
    ubks = keypoints[:11]

    # lower_body_keypoints
    lbks = keypoints[11:]

    """ extra works are needed for normalization and handling keypoints structure"""
    ## these are buggy codes
    # y_diff = np.abs(ubks[:, 1] - lbks[:, 1])
    # x_diff = np.abs(ubks[:, 0] - lbks[:, 0])

    # if np.max(y_diff) <= 0.05 and np.max(x_diff) > 0.5 :
    #      return True
    return False

counters = np.zeros(16) 
fall_threshold = 2

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = camera.Capture()

    if img is None: # timeout
        continue  

    # print(f" image width: {img.width} image height {img.height}")

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:        
        # pose == ObjectPose
        keypoints = np.array(len(pose.Keypoints))
        preprocess_keypoint(keypoints)

        if detect_fall(keypoints):
            counters[pose.ID] += 1
        
        
    check = np.where(counters > 2, True, False)

    if sum(check) > 0:
        print("fall detected!!!")


    # render the image
    display.Render(img)

    # update the title bar
    display.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not camera.IsStreaming() or not display.IsStreaming():
        break