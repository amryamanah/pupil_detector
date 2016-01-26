import os
import argparse
import asyncio
from pprint import pprint
from time import time

from livestockwatch.pupil_finder import main_detector
from livestockwatch.utils import is_skipped_folder
from livestockwatch.config import SKIP_FOLDER
from IPython import embed

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to the scanned folder")
ap.add_argument("-m", "--main-step", required=False, type=int, default=64, help="Main step")
ap.add_argument("-s", "--secondary-step", required=False, type=int, default=8, help="Secondary step")
ap.add_argument("-d", "--duration", required=False, type=float, default=3.0, help="duration of video to be analyzed")
ap.add_argument("--debug", required=False, type=bool, default=False, help="debug flag")
args = vars(ap.parse_args())
print(args)

scanned_folder = os.path.abspath(args["input"])
lst_job = []
for dirpath, dirnames, files in os.walk(scanned_folder):
    part_dirpath = dirpath.split(os.sep)
    if is_skipped_folder(dirpath, SKIP_FOLDER):
        pass
    else:
        for filename in files:
            if filename.endswith(".avi"):
                job_kwargs = {
                    "dirpath": dirpath,
                    "filename": filename,
                    "debug": args["debug"],
                    "duration": args["duration"],
                    "main_step": args["main_step"],
                    "secondary_step": args["secondary_step"]
                }
                lst_job.append(job_kwargs)

start_time = time()
pprint("[START] Processing {} capture session".format(len(lst_job)))
for job in lst_job:
    main_detector(**job)

pprint("[END] Processing {} capture session done in {:.3f} minute".format(
        len(lst_job), (time() - start_time) / 60))
loop = asyncio.get_event_loop()
loop.close()
