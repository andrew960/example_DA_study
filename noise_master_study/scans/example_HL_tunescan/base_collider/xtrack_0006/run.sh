#!/bin/bash
source /afs/cern.ch/work/a/afornara/public/xmask/example_DA_study/noise_master_study/../miniconda/bin/activate
cd /afs/cern.ch/work/a/afornara/public/xmask/example_DA_study/noise_master_study/scans/example_HL_tunescan/base_collider/xtrack_0006
python 2_configure_and_track.py > output.txt 2> error.txt
rm -rf final_* modules optics_repository optics_toolkit tools tracking_tools temp mad_collider.log __pycache__ twiss* errors fc* optics_orbit_at*
