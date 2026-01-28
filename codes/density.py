#!/usr/bin/env python3

# density calculator tool in CGPS package by Yiming Tang @ Fudan
# Development started on Jan 24th 2025

from argparse import ArgumentParser
import numpy as np

import MDAnalysis as mda
from MDAnalysis.analysis import lineardensity as lin
import tqdm

from math import floor

def shift_density_center(density_profile, dense_phase_threshold):
    """
    Centers the dense phase in the z direction of a slab-like simulation box,
    considering the periodic boundary condition.

    Parameters:
        density_profile (numpy.ndarray): 1D array representing the density profile along the z-axis.

    Returns:
        numpy.ndarray: Shifted density profile with the dense phase centered.
    """

    z_length = len(density_profile)
    max_density = np.max(density_profile)
    threshold_density = dense_phase_threshold * max_density

    dense_bins = density_profile > threshold_density

    # Find the continuous dense phase regions considering periodic boundary conditions
    extended_dense_bins = np.concatenate([dense_bins, dense_bins])  # Extend for periodicity
    dense_region_indices = np.where(extended_dense_bins)[0]


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
    num_bins = len(density_profile)
    largest_dense_phase = max(dense_phases, key=len)
    dense_phase_center = floor(np.mean(largest_dense_phase) % num_bins)
    
    shift = floor(num_bins / 2 - dense_phase_center)
    shifted_density_profile = np.roll(density_profile, shift)

    return shifted_density_profile


prog = "density"
desc = '''This program calculate the density profile of a simulation box along one axis'''

parser = ArgumentParser(prog=prog, description=desc)

parser.add_argument('-f', '--input', type=str, required=True, 
                    help="GSD file which is taken as input trajectory.")

parser.add_argument('-o', '--output', type=str, required=True, 
                    help="File to write density profile.")

parser.add_argument('-s', '--axis', type=str, choices=['x', 'y', 'z'], default='z',
                    help="Axis along which the density profile will be calculated. Default: z")

parser.add_argument('-b', '--start-frame', type=int, default=0,
                    help="Index of the first frame to calculate.")

parser.add_argument('-e', '--end-frame', type=int, default=9999999,
                    help="Index of the last frame to calculate.")

parser.add_argument('-df', '--delta-frame', type=int, default=1,
                    help="Number of frames between calculating frames.")

parser.add_argument('-nc', '--no-center', default=False, action='store_true',
                    help="Do not center density profile for each time window. Default is center.")

parser.add_argument('-t', '--dense-phase-threshold', default=0.8, type=float,
                    help="Threshold for dense phase which is a multiplier of the highest density.")

args = parser.parse_args()

output_file_prefix = args.output[0:-4] if ".txt" in args.output else args.output
output_file_name = output_file_prefix + ".txt"

# We now create output file.
try:
    ofile = open(output_file_name, 'w')
    print("## Open text file %s for output." % output_file_name)
except:
    print("## An exception occurred when trying to open text file %s for output." % output_file_name)
    quit()
    

# load trajectory into memory
try:
    u = mda.Universe(args.input, in_memory=True)
    print("## Open trajectory file %s." % args.input)
except:
    print("## An exception occurred when trying to open trajectory file %s." % args.input)
    quit()

if args.start_frame < 0 or args.end_frame > len(u.trajectory):
    print(f"ERROR: Trajectory file of length {len(u.trajectory)} not contain frame {args.start_frame} - {args.end_frame}.")
    quit()

print(f"## Will process a trajectory file with {len(u.trajectory)} frames.")
print(f"## Will use frame {args.start_frame} to {args.end_frame} with window size of {args.delta_frame}.")
print(f"## Will calculate density profile along the {args.axis} axis.")

if args.no_center:
    print(f"## Will not center the density profile.")
else:
    print(f"## Will center dense which is defined as blocks with density higher than {args.dense_phase_threshold} * highest-density.")

# Build a frame list
frames = range(args.start_frame, args.end_frame, args.delta_frame)

# Initialize the main density profile array
density_profile = None
period_number = len(frames) - 1

for index in tqdm.tqdm(range(period_number), desc='## Calculating'):
    start_frame = frames[index]
    end_frame = frames[index + 1]

    # Load selected frames into MDAnalysis        
    density = lin.LinearDensity(u.atoms, grouping='atoms', binsize=0.5).run(start=start_frame, stop=end_frame)
    raw_density = np.array(density.results[args.axis]['pos'])

    # Center density profile if specified
    if not args.no_center:
        output_density = shift_density_center(raw_density, args.dense_phase_threshold)
    else:
        output_density = raw_density

    # Accumulate the density profiles
    if density_profile is None:
        density_profile = output_density
    else:
        density_profile += output_density

# Average density profile across frames
density_profile /= period_number

# Save the density profile to file
for z, density in enumerate(density_profile):
    ofile.write(f"{z / 4.0:.2f}\t{density}\n")
ofile.close()

print(f"## Density profile written to {output_file_name}.")   