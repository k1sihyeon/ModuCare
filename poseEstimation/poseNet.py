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

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, cudaDrawRect, cudaFont


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
font = cudaFont()

fall_count = 0
# process frames until EOS or the user exits
while True:
    # capture the next image
    img = camera.Capture()

    if img is None: # timeout
        continue  

    print(f" image width: {img.width} image height {img.height}")

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))


    for pose in poses:        
            font.OverlayText(img, text=f"Fall Count: {fall_count}", 
                         x=5, y=5 * (font.GetSize() + 5),
                         color=font.White, background=font.Gray40)
            neck_id = pose.FindKeypoint('neck')
            r_ankle_id = pose.FindKeypoint('right_ankle')
            l_ankle_id = pose.FindKeypoint('left_ankle')

            if neck_id < 0:
                  continue
            if r_ankle_id + l_ankle_id < 0:
                  continue            
            cudaDrawRect(img, (pose.Left, pose.Top, pose.Right, pose.Bottom), line_color=(0, 75, 255, 200))
            x_diff = abs(pose.Left -  pose.Right)
            y_diff = abs(pose.Top - pose.Bottom)
            if x_diff > y_diff:
                  fall_count += 1
                  

    # render the image
    display.Render(img)

    # update the title bar
    display.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not camera.IsStreaming() or not display.IsStreaming():
        break