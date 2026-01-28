#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import gsd.hoomd
import MDAnalysis as mda
from MDAnalysis.analysis.base import (AnalysisBase,
                                      AnalysisFromFunction,
                                      analysis_class)
from MDAnalysis.transformations import unwrap
import numpy as np
import os
import tqdm
import pandas as pd

frame_per_ns = 10

usage = '''
Program: hps_boxsize by Yiming Tang @ Fudan
This program get a single frame and generate a single GSD file
Input files:
       -f   [md.gsd]  : trajectory file
Output files:
       -o   [out.gsd] : outout file
Parameters:
       -id            : Frame index starting from 0
'''


# We get parameters

input_trajectory_file_path = "md.gsd"
output_trajectory_file_path = "out.gsd"
specify_frame_index = False 

arg_count = 1
while arg_count < len(sys.argv):
    if sys.argv[arg_count] == '-h':
        print(usage)
        quit()
    elif sys.argv[arg_count] == "-f":
        input_trajectory_file_path = sys.argv[arg_count + 1]
        arg_count += 1
    elif sys.argv[arg_count] == "-o":
        output_trajectory_file_path = sys.argv[arg_count + 1]
        arg_count += 1
    elif sys.argv[arg_count] == "-id":
        frame_index = int(sys.argv[arg_count + 1])
        specify_frame_index = True
        arg_count += 1
    else:
        print("Unknown command line parameter \"%s\"" % sys.argv[arg_count])
        quit()
    arg_count += 1

if not specify_frame_index:
    print("Which frame to output?")
    quit()

if not os.path.exists(input_trajectory_file_path):
    print("File not exist error: %s" % input_trajectory_file_path)
    quit()
else:
    print("** Will read trajectory file from: %s" % input_trajectory_file_path)

input_trajectory = gsd.hoomd.open(input_trajectory_file_path, 'r')

print("** There are altogether %d frames." % (input_trajectory.__len__()))

if frame_index >= input_trajectory.__len__():
    print("** Cannot read frame number %d." % frame_index)
    quit()

print("** Will read frame number %d." % frame_index)

frame = input_trajectory[frame_index]
box = frame.configuration.box[0:3]
print("** Box size of this frame: %f, %f, %f." % (box[0], box[1], box[2]))

if ".gsd" not in output_trajectory_file_path:
    output_trajectory_file_path = output_trajectory_file_path + ".gsd"

print("** Will write to file: %s." % output_trajectory_file_path)
trajectory = gsd.hoomd.open(output_trajectory_file_path, 'w')
trajectory.append(frame)

input_trajectory.close()
trajectory.close()

