#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import gsd.hoomd
import MDAnalysis as mda
from MDAnalysis.coordinates.GSD import GSDReader
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
from MDAnalysis.transformations import unwrap

def build_chains_from_bonds(gsd_file):
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
    atom_to_chain = {}
    for chain_idx, atom_indices in enumerate(chains):
        for atom in atom_indices:
            atom_to_chain[atom] = chain_idx
    return atom_to_chain, len(chains)

def radius_of_gyration(positions, masses):

    com = np.average(positions, axis=0, weights=masses)
    squared_distances = np.sum((positions - com)**2, axis=1)
    return np.sqrt(np.average(squared_distances, weights=masses))


def compute_rg_distribution(gsd_file, frame_start, frame_end, frame_step):
    atom_to_chain, n_chains = build_chains_from_bonds(gsd_file)

    # Setup minimal topology
    with gsd.hoomd.open(gsd_file, 'r') as traj:
        n_atoms = traj[0].particles.N

    u = mda.Universe(gsd_file, gsd_file)
    u.trajectory.add_transformations(unwrap(u.atoms))

    # Invert atom_to_chain: get atom indices per chain
    chain_atoms = [[] for _ in range(n_chains)]
    for atom_idx, chain_id in atom_to_chain.items():
        chain_atoms[chain_id].append(atom_idx)

    # Calculate Rg per chain per frame
    rg_dict = {chain_id: [] for chain_id in range(n_chains)}

    for ts in tqdm(u.trajectory[frame_start:frame_end:frame_step]):
        positions = u.atoms.positions
        for chain_id, atom_indices in enumerate(chain_atoms):
            pos = positions[atom_indices]
            masses = np.ones(len(pos))  # or customize if needed
            rg = radius_of_gyration(pos, masses)
            rg_dict[chain_id].append(rg)

    return rg_dict, chain_atoms


prog = "gyrate"
desc = '''This program calculate the radius of gyration of all chains in a gsd trajectory.'''

parser = ArgumentParser(prog=prog, description=desc)

parser.add_argument('-f', '--input', type=str, required=True, 
                    help="GSD file which is taken as input trajectory.")

parser.add_argument('-v', '--verbose', type=str, 
                    help="File to write verbose radius of gyration.")

parser.add_argument('-o', '--output', type=str, 
                    help="File to write distribution of gyration.")

parser.add_argument('-bw', '--bin-width', type=float, default=0.1,
                    help="Bin width of output distribution.")

parser.add_argument('-b', '--start-frame', type=int, default=0,
                    help="Index of the first frame to calculate.")

parser.add_argument('-e', '--end-frame', type=int, default=9999999,
                    help="Index of the last frame to calculate.")

parser.add_argument('-df', '--delta-frame', type=int, default=1,
                    help="Number of frames between calculating frames.")

args = parser.parse_args()

# We now try to load input file.
# load trajectory into memory
try:
    u = mda.Universe(args.input)
    print("## Open trajectory file %s." % args.input)
except:
    print("## An exception occurred when trying to open trajectory file %s." % args.input)
    quit()

if args.end_frame == 9999999:
    args.end_frame = len(u.trajectory)

if args.start_frame < 0 or args.end_frame > len(u.trajectory):
    print(f"ERROR: Trajectory file of length {len(u.trajectory)} not contain frame {args.start_frame} - {args.end_frame}.")
    quit()

print(f"## Will process a trajectory file with {len(u.trajectory)} frames.")
print(f"## Will use frame {args.start_frame} to {args.end_frame} with window size of {args.delta_frame}.")

rg_dict, chain_atoms = compute_rg_distribution(args.input, args.start_frame, args.end_frame, args.delta_frame)
chain_length = [len(chain) for chain in chain_atoms]

# We now print verbose output

if args.verbose is not None:
    verbose_output = np.array([rg_dict[chain_index] for chain_index in range(len(chain_length))]).transpose()
    verbose_file_name = args.verbose + ".xlsx" if ".xlsx" not in args.verbose else args.verbose

    df = pd.DataFrame(verbose_output)
    df.to_excel(verbose_file_name, index=False, header=False)  

    print(f"Anverage Rg: {np.mean(verbose_output)}")

if args.output is not None:
    print("Distribution output not implemented.")
    


