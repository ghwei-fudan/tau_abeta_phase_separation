#!/usr/bin/env python3

# density calculator tool in CGPS package by Yiming Tang @ Fudan
# Development started on Jan 24th 2025

from argparse import ArgumentParser
import numpy as np

import MDAnalysis as mda
from MDAnalysis.analysis import lineardensity as lin
import tqdm

from math import floor

import gsd.hoomd
import networkx as nx

def build_molecules_from_bonds(gsd_file):
    with gsd.hoomd.open(gsd_file, 'r') as traj:
        frame = traj[0]
        n_atoms = frame.particles.N
        bonds = frame.bonds.group

    # Build a graph from bonds
    G = nx.Graph()
    G.add_nodes_from(range(n_atoms))
    G.add_edges_from(bonds)

    # Find connected components (chains)
    chains = list(nx.connected_components(G))
    chains = [[int(number) for number in chain] for chain in chains]

    atom_to_chain = {}
    for chain_idx, atom_indices in enumerate(chains):
        for atom in atom_indices:
            atom_to_chain[atom] = chain_idx
    
    chain_length_for_chain = [len(chain) for chain in chains]
    chain_length_list = np.sort(list(set(chain_length_for_chain)))[-1::-1]

    chain_type_index_for_chain = [int(np.where(chain_length_list == len(chain))[0][0]) for chain in chains]

    print(f"## Found {len(chain_length_list)} chains with chain length of {', '.join([f"{length}" for length in chain_length_list])}.")

    bead_index_in_molecules = [list() for chain_type_index in range(len(chain_length_list))]
    
    for chain_index in range(len(chains)):
        bead_index_in_molecules[chain_type_index_for_chain[chain_index]].extend(chains[chain_index])

    for molecule_index in range(len(bead_index_in_molecules)):

        bead_number = len(bead_index_in_molecules[molecule_index])
        chain_length = chain_length_list[molecule_index]

        if bead_number % chain_length != 0:
            print(f"ERROR: Chain length {chain_length} is not a multiplier of bead_number {bead_number}.")
            print(f"QUITING...")
            quit()
        
        print(f"## Molecule type {molecule_index + 1}: {int(bead_number / chain_length)} chains, length: {chain_length}, bead number: {bead_number}.")
    
    return bead_index_in_molecules, chain_length_list
    

def shift_density_center(density_profiles, dense_phase_threshold):
    """
    Centers the dense phase in the z direction of a slab-like simulation box,
    considering the periodic boundary condition.

    Parameters:
        density_profile (numpy.ndarray): 1D array representing the density profile along the z-axis.

    Returns:
        numpy.ndarray: Shifted density profile with the dense phase centered.
    """

    density_profile = density_profiles[0]

    #z_length = len(density_profile)
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
    shifted_density_profiles = [np.roll(density_profile_single, shift) for density_profile_single in density_profiles]

    return shifted_density_profiles


prog = "density"
desc = '''This program calculate the density profile of a simulation box along one axis'''

parser = ArgumentParser(prog=prog, description=desc)

parser.add_argument('-f', '--input', type=str, required=True, 
                    help="GSD file which is taken as input trajectory.")

parser.add_argument('-o', '--output', type=str, required=True, 
                    help="File to write density profile.")

parser.add_argument('-p', '--split', type=bool, default=False,
                    help="Whether separate output for each molecule.")

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

selections = [u.atoms]
system_title_text = "All"

if args.split == True:
    print(f"## Will process seperate molecules.")
    molecules, chain_length_list = build_molecules_from_bonds(args.input)
    chain_length_text = '\t'.join([f"{chain_length}" for chain_length in chain_length_list])
    system_title_text = f"{system_title_text}\t{chain_length_text}"

    #molecules_string = [','.join([f"{atom}" for atom in molecule]) for molecule in molecules] 
    #sel = [u.select_atoms(f"index {molecule_string}") for molecule_string in molecules_string]
    selections.extend([u.atoms[molecule] for molecule in molecules])

# Build a frame list
frames = range(args.start_frame, args.end_frame, args.delta_frame)

# Initialize the main density profile array
density_profiles = None
period_number = len(frames) - 1

for index in tqdm.tqdm(range(period_number), desc='## Calculating'):
    start_frame = frames[index]
    end_frame = frames[index + 1]

    # Load selected frames into MDAnalysis
    density_profiles_temp = [lin.LinearDensity(selection, grouping='atoms', binsize=0.5).run(start=start_frame, stop=end_frame)
                             for selection in selections]
    #density = lin.LinearDensity(u.atoms, grouping='atoms', binsize=0.5).run(start=start_frame, stop=end_frame)
    
    raw_densities = [np.array(density.results[args.axis]['pos']) for density in density_profiles_temp]
    #raw_density = np.array(density.results[args.axis]['pos'])

    # Center density profile if specified
    if not args.no_center:
        output_densities = np.array(shift_density_center(raw_densities, args.dense_phase_threshold))
    else:
        output_densities = np.array(raw_densities)
    
    # Accumulate the density profiles
    if density_profiles is None:
        density_profiles = output_densities
    else:
        density_profiles += output_densities
    
# Average density profile across frames

density_profiles /= period_number
density_profiles = density_profiles.transpose()


# We now create output file.
try:
    ofile = open(output_file_name, 'w')
    print("## Open text file %s for output." % output_file_name)
except:
    print("## An exception occurred when trying to open text file %s for output." % output_file_name)
    quit()

#title_text = f"all\t{'\t'.join([ for chain_length])}"

ofile.write(f"z\t{system_title_text}\n")

# Save the density profile to file
for z, density in enumerate(density_profiles):
    density_text = '\t'.join([f"{number}" for number in density])
    ofile.write(f"{z / 4.0:.2f}\t{density_text}\n")
ofile.close()

print(f"## Density profile written to {output_file_name}.")   
