# Simulation of Ocean, Atmosphere, and Climate (SOAC) Project

## Overview

This repository contains the code and data for a computational study on tidal phenomena in fjords, specifically focusing on the dynamics around sill areas which are often characterized by large tidal currents. The study employs a one-dimensional model derived from shallow water equations, aimed at understanding the tidal behaviors in basins similar to those found in Kangiqtualuk Uqquqti (formerly Sam Ford Fjord) in northern Canada and Van Mijenfjorden in Svalbard.

The goal is to demonstrate that our model can replicate the qualitative aspects of tidal behaviors observed in fjords and provide insights into the role of sill height in influencing the basin's resonance characteristics.

## Contents

- `main.py`: The main script for running the one-dimensional tidal model simulations.
- `fft.py`: This script is used for conducting Fourier analysis on time-series data.
- `depth_profile.py`: Script to generate depth profile.
- `tides_Hprofile.py`: Script to simulate tidal height profiles.
- `Su.npy` and `Szeta.npy`: NumPy array files containing simulation data for the water surface velocity (`Su`) and elevation (`Szeta`).
- `depth_profile.png`: Visualization of the depth profile used in the simulations.
- `experiment.png`: Graphical results from the experiment.
- `z and u.png`: Visual representation of the variables 'z' and 'u' from the simulations.

## Simulation Details

The study employs numerical methods, specifically the Runge-Kutta 4th order (RK4) method, to advance the simulations through time steps. The code takes into account both linear and nonlinear aspects of the shallow water dynamics to better reflect the physical processes at play in real-world fjords.

## Installation and Usage

To run the simulations and analyze the results, ensure that you have Python 3.x installed along with the following libraries:

- NumPy
- Matplotlib
- SciPy

Run the `main.py` script to execute the model. Data generated from the simulations will be saved to `.npy` files, which can then be visualized using the accompanying scripts.

