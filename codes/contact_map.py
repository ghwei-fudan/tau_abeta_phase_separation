#!/usr/bin/env python3

# contact map tool in CGPS package by Yiming Tang @ Fudan
# Development started on Jan 4th 2025

from argparse import ArgumentParser
import gsd.hoomd
import numpy as np
from copy import copy, deepcopy
import MDAnalysis as mda
from MDAnalysis.analysis import contacts, distances
from tqdm import tqdm
import pandas as pd

prog = "contact_map"
desc = '''This program calculate contact map from a gsd file.'''

parser = ArgumentParser(prog=prog, description=desc)

parser.add_argument('-f', '--structure', type=str, required=True, 
                    help="GSD file which is taken as input.")

parser.add_argument('-o', '--output', type=str, required=True, 
                    help="Prefix for writting contact maps.")

parser.add_argument('-c', '--cutoff', type=float, default=0.7,
                    help="Cutoff distance for contact calculation.")

parser.add_argument('-b', '--start-frame', type=int, default=0,
                    help="Index of the first frame to calculate.")

parser.add_argument('-e', '--end-frame', type=int, default=9999999,
                    help="Index of the last frame to calculate.")

parser.add_argument('-df', '--delta-frame', type=int, default=1,
                    help="Number of frames between calculating frames.")

parser.add_argument('-sc', '--split-chain-criteria', type=str, choices=["sequence", "length"], default="sequence", 
                    help="Chains with different criteria is considered different molecules.")

args = parser.parse_args()
output_file_prefix = args.output[0:-4] if ".gsd" in args.output else args.output

# We first get the gsd file
try:
    trajectory = gsd.hoomd.open(args.structure)
    print("## Open structure file %s." % args.structure)
except:
    print("## An exception occurred when trying to open structure file %s." % args.structure)
    quit()

frame_number = len(trajectory)
if args.end_frame == 9999999:
    end_frame_index = frame_number
else:
    end_frame_index = args.end_frame

start_frame_index = args.start_frame
delta_frame = args.delta_frame

frame_list = list(range(start_frame_index, end_frame_index, delta_frame))
frame_pass = all(elem in range(0, frame_number) for elem in frame_list)
if not frame_pass:
    print(f"ERROR: Frame index out of trajectory range. Trajectory contains {frame_number} frames.")
    quit()

print(f"## Will process {len(frame_list)} frames from {frame_list[0]} to {frame_list[-1]}.")

# We first get molecule list

bond_list = np.array(trajectory[0].bonds.group).tolist()
atom_number = trajectory[0].particles.N
molecule_list = list()
treating_index = 0

while(treating_index < atom_number):
    molecule_temp = [copy(treating_index)]
    treating_index += 1
    while([treating_index - 1, treating_index] in bond_list):
        molecule_temp.append(copy(treating_index))
        treating_index += 1
    molecule_list.append(deepcopy(molecule_temp))

# We now get types for each molecule

if args.split_chain_criteria == "sequence":

    print("## Will split molecules using sequence as criteria.")

    sequence_all = np.array([trajectory[0].particles.types[id] for id in trajectory[0].particles.typeid])

    molecule_types = list()
    molecules_in_type = list()

    for molecule in molecule_list:
        molecule_sequence = ','.join(sequence_all[molecule])
        
        if molecule_sequence not in molecule_types:
            molecule_types.append(deepcopy(molecule_sequence))
            molecules_in_type.append(list())
        
        molecule_index = molecule_types.index(molecule_sequence)
        molecules_in_type[molecule_index].append(molecule)

elif args.split_chain_criteria == "length":

    print("## Will split molecules using length as criteria.")

    molecule_types = list()
    molecules_in_type = list()

    for molecule in molecule_list:
        molecule_length = len(molecule)
        
        if molecule_length not in molecule_types:
            molecule_types.append(molecule_length)
            molecules_in_type.append(list())
        
        molecule_index = molecule_types.index(molecule_length)
        molecules_in_type[molecule_index].append(molecule)

else:
    print(f"ERROR. Quiting.")
    quit()

# We now get lists for residues in each type of molecules for calculation of inter-residue contact

residues_in_molecules = list()

for molecule_index in range(len(molecule_types)):
    molecule_length = len(molecules_in_type[molecule_index][0])
    molecule_number = len(molecules_in_type[molecule_index])

    residues_lists = list()     
    for residue_index in range(molecule_length):
        residues_lists.append([molecules_in_type[molecule_index][chain_index][residue_index] 
                               for chain_index in range(molecule_number)])
    residues_in_molecules.append(deepcopy(residues_lists))

print(f"## Got {len(molecule_types)} types of molecules.")
for molecule_index in range(len(molecule_types)):

    sequence_length = molecule_types[molecule_index] if args.split_chain_criteria == "length" else len(molecule_types[molecule_index].split(','))

    print(f"## The sequence length of molecule type {molecule_index + 1} is {sequence_length};")
    print(f"## Molecule number of the molecule type {molecule_index + 1} is {len(residues_in_molecules[molecule_index][0])}.")

# We now initialize the MDTrajectory engine

system_universe = mda.Universe(args.structure, args.structure)
atom_number = len(system_universe.trajectory[0])

print(f"## Universe for trajectory {args.structure} initialized.")

contact_map = np.zeros((atom_number, atom_number))

print(f"## Raw data will contains contact maps between {atom_number}*{atom_number} pairs.")
print(f"## Cutoff distance for contacts set as {args.cutoff}")
print(f"## Calculating for requested frames.")

for frame_index in tqdm(frame_list):
    distance_map = distances.distance_array(system_universe.trajectory[frame_index], 
                                            system_universe.trajectory[frame_index], system_universe.dimensions)
    contact_map_temp = (distance_map < args.cutoff).astype(int)
    contact_map += contact_map_temp

contact_map = contact_map / len(frame_list) 

print(f"## Raw {atom_number}*{atom_number} contact map calculation completed.")

# We now calculate inter contact maps.

print(f"## Calculating inter contact maps.")

for molecule_index_1 in range(len(molecule_types)):
    for molecule_index_2 in range(len(molecule_types)):
        indices_1 = np.array(molecules_in_type[molecule_index_1])
        indices_2 = np.array(molecules_in_type[molecule_index_2])
        if molecule_index_1 != molecule_index_2:
            inter_contact_map = np.sum([contact_map[molecule_calculate_1[0]:molecule_calculate_1[-1]+1,\
                molecule_calculate_2[0]:molecule_calculate_2[-1]+1] 
                                         for molecule_calculate_1 in indices_1 for molecule_calculate_2 in indices_2], axis=0)
        else:
            inter_contact_map = np.sum([contact_map[molecule_calculate_1[0]:molecule_calculate_1[-1]+1,\
                molecule_calculate_2[0]:molecule_calculate_2[-1]+1] 
                                         for molecule_calculate_1 in indices_1 for molecule_calculate_2 in indices_2
                                         if not np.array_equal(molecule_calculate_1, molecule_calculate_2)], axis=0)
        
        print(f"## Inter-residue contact map between molecule {molecule_index_1 + 1} and {molecule_index_2 + 1} calculated.")
        inter_contact_map_file_name = output_file_prefix + f"_mol{molecule_index_1 + 1}_mol{molecule_index_2 + 1}.xlsx"
        df = pd.DataFrame(inter_contact_map)
        df.to_excel(inter_contact_map_file_name, index=False, header=False)  
        print(f"     -> written to {inter_contact_map_file_name}.")



# We now calculate intra contact maps.
print(f"## Calculating intra contact maps.")

for molecule_index in range(len(molecule_types)):

    molecule_length = len(molecules_in_type[molecule_index][0])

    indices = np.array(molecules_in_type[molecule_index])
    # [[1,2,3,4,5,6], [7,8,9,10,11,12]] for two molecules each with six residues.

    intra_contact_map = np.mean([contact_map[molecule_calculate[0]:molecule_calculate[-1]+1,molecule_calculate[0]:molecule_calculate[-1]+1] 
                                 for molecule_calculate in indices], axis=0)
    
    print(f"## Intra-residue contact map between molecule {molecule_index + 1} and itself calculated.")
    
    intra_contact_map_file_name = output_file_prefix + f"_mol{molecule_index + 1}_intra.xlsx"
    df = pd.DataFrame(intra_contact_map)
    df.to_excel(intra_contact_map_file_name, index=False, header=False)  
    print(f"     -> written to {intra_contact_map_file_name}.")




