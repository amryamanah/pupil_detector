import cv2
import os
import numpy as np

import json
import shutil
from livestockwatch.pupil_analysis import Parameter, save_pupil_analysis_result, detect_pupil, EllipseAnalysis, \
    ellipse_calculate_ca, ellipse_normalized_area
from livestockwatch.utils import is_skipped_folder, write_csv_result, write_as_png
from livestockwatch.filters import savitzky_golay
from scipy.signal import savgol_filter, medfilt
from IPython import embed
import matplotlib.pyplot as plt


skip_folder = ["pending", "finished",
               "nopl", "pl",
               "bbox_img", "candidate", "final", "final_extended", "plr_result",
               "hard_neg", "raw", "hist_equ", "hist_equ_blur", "positive", "raw", "union"]

result_header = [
    "cs_name", "frametime", "framecount", "area", "svg_area",
    "max_svg_area", "max_area", "normalized_area", "normalized_svg_area",
    "ipr", "eccentricity",
    "major_axis", "minor_axis", "max_ipr", "ca", "svg_ca"
]

lst_frame = [(8, 0.53), (9, 0.6), (10, 0.67), (11, 0.73), (12, 0.8), (13, 0.87), (14, 0.93), (15, 1.0), (16, 1.07),
             (17, 1.13), (18, 1.2), (19, 1.27),
             (20, 1.33), (21, 1.4), (22, 1.47), (23, 1.53), (24, 1.6), (25, 1.67), (26, 1.73), (27, 1.8), (28, 1.87),
             (29, 1.93), (30, 2.0), (31, 2.07),
             (32, 2.13), (33, 2.2), (34, 2.27), (35, 2.33), (36, 2.4), (37, 2.47), (38, 2.53), (39, 2.6), (40, 2.67),
             (41, 2.73), (42, 2.8), (43, 2.87),
             (44, 2.93)]

blank_img = cv2.imread("/Users/fitram/Project/computer_vision/BISE/pupil_detector/blank.png")

basedir = os.getcwd()
cs_root = os.path.join(basedir, "data_analysis", "high_low")
config_path = os.path.join(cs_root, "config.json")

with open(config_path) as f:
    lst_config = json.load(f)

for config in lst_config:
    cs_dir = os.path.join(cs_root, config["totd"], config["stall_name"], config["cattle_id"], config["cs_name"])
    hue_min = config["hue_min"]
    hue_max = config["hue_max"]
    sat_min = config["sat_min"]
    sat_max = config["sat_max"]
    val_min = config["val_min"]
    val_max = config["val_max"]

    for k,v in lst_frame:
        img_path = os.path.join(cs_dir, "{}_{}.png".format(k, v))
        if not os.path.exists(img_path):
            write_as_png(img_path, blank_img)

    lst_filename = []
    for filename in os.listdir(cs_dir):
        if filename.endswith(".png"):
            frame_count = filename.split("_")[0]
            lst_filename.append((int(frame_count), os.path.join(cs_dir, filename)))

    lst_filename = sorted(lst_filename)
    global_count = 0
    lst_result = []
    lst_index = []
    csv_path = os.path.join(cs_dir, "plr.csv")
    for frame_count, img_path in lst_filename:
        part_dirpath, file_extension = os.path.splitext(img_path)
        part_dirpath = part_dirpath.split(os.sep)
        filename = img_path.split(os.sep)[-1]
        cnt = 0
        if os.path.exists(csv_path):
            os.remove(csv_path)

        nopl_result_path = os.path.join(cs_dir, "plr_result")
        os.makedirs(nopl_result_path, exist_ok=True)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        param = Parameter(
                hue_min=hue_min,
                hue_max=hue_max,
                val_min=val_min,
                val_max=val_max,
                sat_max=sat_max,
                sat_min=sat_min
        )
        result_dict = {
            "cs_name": part_dirpath[-2],
            "frametime": float(part_dirpath[-1].split("_")[1]),
            "framecount": float(part_dirpath[-1].split("_")[0]),
            "ca": None,
            "svg_ca": None,
            "ipr": None,
            "max_ipr": None,
            "area": None,
            "svg_area": None,
            "max_area": None,
            "max_svg_area": None,
            "normalized_area": None,
            "normalized_svg_area": None,
            "eccentricity": None,
            "minor_axis": None,
            "major_axis": None
        }
        props, pupil_area = detect_pupil(img, param)
        if props:
            result_path = os.path.join(nopl_result_path, filename)
            save_pupil_analysis_result(img, props, result_path)
            result_dict["minor_axis"] = props.minor_axis_length
            result_dict["major_axis"] = props.major_axis_length
            ea = EllipseAnalysis(result_dict["major_axis"], result_dict["minor_axis"])
            result_dict["area"] = ea.area
            result_dict["ipr"] = ea.ipr
            result_dict["eccentricity"] = ea.eccentricity
            lst_index.append(global_count)

        lst_result.append(result_dict)
        global_count += 1

    lst_area = [lst_result[x]["area"] for x in lst_index]
    if len(lst_index) > 0:
        y_area = np.array(lst_area)
        # ysg_area = savitzky_golay(y_area, 21, 4)
        ysg_area = savgol_filter(y_area, window_length=27, polyorder=4, mode="nearest")
        for i, v in enumerate(lst_index):
            lst_result[v]["svg_area"] = ysg_area[i]

        max_svg_area = max([lst_result[x]["svg_area"] for x in lst_index])
        max_area = max([lst_result[x]["area"] for x in lst_index])
        max_ipr = max([lst_result[x]["ipr"] for x in lst_index])
        for i, v in enumerate(lst_index):
            lst_result[v]["max_svg_area"] = max_svg_area
            lst_result[v]["max_area"] = max_area
            lst_result[v]["normalized_area"] = ellipse_normalized_area(lst_result[v]["area"], max_area)
            lst_result[v]["normalized_svg_area"] = ellipse_normalized_area(lst_result[v]["svg_area"],
                                                                           max_svg_area)
            lst_result[v]["max_ipr"] = max_ipr
            lst_result[v]["svg_ca"] = ellipse_calculate_ca(lst_result[v]["svg_area"], max_svg_area)
            lst_result[v]["ca"] = ellipse_calculate_ca(lst_result[v]["area"], max_area)

    for result in lst_result:
        write_csv_result(csv_path, result_header, result)
