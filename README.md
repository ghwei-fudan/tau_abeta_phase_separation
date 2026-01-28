This repository contains the code used for the MD simulations in:

#### Alzheimer’s Aβ Catalyzes Tau Phase Separation and Aggregation via Early Nanocluster Solubilization

Xun Sun, Yiming Tang, Xue Wang, Guadalupe Pereira Curia, Rebecca Sternke-Hoffmann, Cecilia Mörman, Juan Atilio Gerez, Roland Riek, Guanghong Wei,*, Jinghui Luo1,*

#### File description:

**trajectory/Tau/**: Contains the gsd files for initial and final steps of Tau phase separation production run at different temperatures. The naming follows the pattern: Tau_(temperature)K_(initial or final state).gsd

**trajectory/ABeta/**: Contains the gsd files for initial and final steps of ABeta phase separation production run at different temperatures. The naming follows the pattern: ABeta_(temperature)K_(initial or final state).gsd

**trajectory/Tau_ABeta/**: Contains the gsd files for initial and final steps of Tau and ABeta co-phase separation production run at different temperatures. The naming follows the pattern: Tau_ABeta_(temperature)K_(initial or final state).gsd

**codes/**: Contains the HOOMD script used in this study,
- **contact_map.py** calculates inter- and intra-molecular contact number between each two residues.
- **density.py** calculates the density profile of phase separation systems.
- **density_multiple.py** calculates the density profile of each molecule type within co-phase separation systems.
- **gyrate.py** calculates the distribution of radius of gyration of all molecules within a simulation system.
- **hys_getFrame.py** extracts a single frame from a HOOMD simulation trajectory.
- **trjcenter.py** centers and dense phase for visualization.

