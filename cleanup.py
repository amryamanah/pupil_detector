import os
import shutil


for dirpath, dirnames, files in os.walk("/Volumes/amryhd/dataset-final/WA02/day/816"):
    if dirpath.endswith("nopl"):
        shutil.rmtree(dirpath)