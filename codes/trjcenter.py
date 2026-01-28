#!/usr/bin/env python3

# trajectory center tool in CGPS package by Yiming Tang @ Fudan
# Development started on Jan 23rd 2025

from argparse import ArgumentParser
import gsd.hoomd
from copy import deepcopy
import numpy as np

prog = "trjcenter"
desc = '''This program center the dense phase for all frames in a gsd file.'''

parser = ArgumentParser(prog=prog, description=desc)

parser.add_argument('-f', '--input', type=str, required=True, 
                    help="GSD file which is taken as input.")

parser.add_argument('-o', '--output', type=str, required=True, 
                    help="GSD file which to write editted frames.")

parser.add_argument('-cx', '--center-x', default=False, action='store_true',
                    help="Whether center the dense phase along the x direction. Default: False.")

parser.add_argument('-cy', '--center-y', default=False, action='store_true',
                    help="Whether center the dense phase along the y direction. Default: False.")

parser.add_argument('-cz', '--center-z', default=False, action='store_true',
                    help="Whether center the dense phase along the z direction. Default: False.")

parser.add_argument('-t', '--dense-phase-threshold', default=0.8, type=float,
                    help="Threshold for dense phase which is a multiplier of the highest density.")

args = parser.parse_args()

output_file_prefix = args.output[0:-4] if ".gsd" in args.output else args.output
output_file_name = output_file_prefix + ".gsd"

# We first get the gsd file and transform it into frame
try:
    raw_frames = gsd.hoomd.open(args.input)
    print("## Open trajectory file %s." % args.input)
except:
    print("## An exception occurred when trying to open trajectory file %s." % args.input)
    quit()

print(f"## Will process trajectory file with {len(raw_frames)} frames.")

if args.center_x:
    print(f"## Will center the dense phase in the x axis")
if args.center_y:
    print(f"## Will center the dense phase in the y axis")
if args.center_z:
    print(f"## Will center the dense phase in the z axis")

print(f"## Blocks with density higher than {args.dense_phase_threshold} * highest-density will be considered the dense phase.")

if (not args.center_x) and (not args.center_y) and (not args.center_z):
    print(f"ERROR: No axes specified. The program will quit.")
    quit()

# We get cell information
cell = raw_frames[0].configuration.box
print(f"## The cell size of the simulation box at initial is ({cell[0]:.2f} * {cell[1]:.2f} * {cell[2]:.2f}).")

# We now create output file.
try:
    ofile = gsd.hoomd.open(output_file_name, 'w')
    print("## Open trajectory file %s for output." % output_file_name)
except:
    print("## An exception occurred when trying to open trajectory file %s for output." % args.input)
    quit()

def center_dense_phase(coordinates, box, axis, dense_phase_threshold):

    raw_coordinates_axis = coordinates[:, axis]
    axis_min = - box[axis] / 2
    axis_max = box[axis] / 2
    box_length = box[axis]
    #print(raw_coordinates_axis)

    # Wrap coordinates into the range [-c/2, c/2] using periodic boundary conditions
    coordinates_wrapped = (raw_coordinates_axis - axis_min) % box_length + axis_min

    # Create a histogram of the coordinates
    num_bins = int(box_length / 0.5)  # Adjust bin size (0.1) as needed
    histogram, bin_edges = np.histogram(coordinates_wrapped, bins=num_bins, range=(axis_min, axis_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Identify the dense phase
    max_density = np.max(histogram)
    density_threshold = dense_phase_threshold * max_density
    dense_bins = histogram > density_threshold

    # Find the continuous dense phase regions considering periodic boundary conditions
    extended_dense_bins = np.concatenate([dense_bins, dense_bins])  # Extend for periodicity
    dense_region_indices = np.where(extended_dense_bins)[0]

    #print(dense_region_indices)

    # Find the largest continuous region in the dense bins
    dense_phases = []
    current_phase = [dense_region_indices[0]]
    for idx in dense_region_indices[1:]:
        if idx == current_phase[-1] + 1:
            current_phase.append(idx)
        else:
            dense_phases.append(current_phase)
            current_phase = [idx]
    dense_phases.append(current_phase)

    # Map back to the original bins and choose the largest region
    largest_dense_phase = max(dense_phases, key=len)
    largest_dense_phase = [idx % num_bins for idx in largest_dense_phase]
    dense_bin_centers = bin_centers[largest_dense_phase]

    # Calculate the center of the dense phase
    dense_center = np.mean(dense_bin_centers)

    # Shift coordinates to center the dense phase at 0
    coordinate_shift = - dense_center
    coordinates_centered = (coordinates_wrapped + coordinate_shift - axis_min) % box_length + axis_min

    # Update the coordinates array
    new_coordinates = coordinates.copy()
    new_coordinates[:,axis] = coordinates_centered

    return new_coordinates


for frame in raw_frames:
    # We now process a single frame.
    new_frame = deepcopy(frame)
    temp_coordinate = new_frame.particles.position

    # We now center the coordinate
    if args.center_x:
        temp_coordinate = deepcopy(center_dense_phase(temp_coordinate, frame.configuration.box, 0, args.dense_phase_threshold))

    if args.center_y:
        temp_coordinate = deepcopy(center_dense_phase(temp_coordinate, frame.configuration.box, 1, args.dense_phase_threshold))
    
    if args.center_z:
        temp_coordinate = deepcopy(center_dense_phase(temp_coordinate, frame.configuration.box, 2, args.dense_phase_threshold))
    
    new_frame.particles.position = temp_coordinate

    ofile.append(new_frame)

ofile.close()
print(f"## Processed trajectory written to {output_file_name}.")   

