"""Fluid injection into a pre-existing fracture: solvers, analysis, and figures.

Pipeline
--------
    3DEC / FVM simulation  ->  HDF5 file  ->  fracinj.io.read_hdf5  ->  RunData
        ->  fracinj.analysis / fracinj.detection  ->  fracinj.plotting

Subpackages
-----------
    fracinj.solvers   numerical + (semi-)analytical solvers
    fracinj.physics   dimensional analysis and material coefficients
    fracinj.analysis  power-law fits of front / injection-point histories
    fracinj.io        readers and writers for run data
"""
