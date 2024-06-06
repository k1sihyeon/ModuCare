import jetson_inference
import jetson_utils

import numpy as np

net = jetson_inference.poseNet("resnet18-body", threshold=0.15)

n = net.GetNumKeypoints()

# print(n)

# for i in range(n):
#     print(net.GetKeypointName(i))

a = np.zeros(4)

print(a)