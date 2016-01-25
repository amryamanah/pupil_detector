import os
import pytz
import datetime
from pprint import pprint
import logging
from time import time
import asyncio
import shutil
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from odo import odo
import pymongo
from . import LOCAL_TIMEZONE
from . import config
from .utils import get_specific_channel, equ_hist_color_image, write_as_png, svg_smoothing
from .pupil_analysis import pupil_fit_ellipse, EllipseAnalysis, ellipse_calculate_ca
from .descriptors import HogDescriptor
from IPython import embed

logger = logging.getLogger(__name__)


def main_detector(**kwargs):
    start_time = time()
    dirpath = kwargs["dirpath"]
    filename = kwargs["filename"]
    debug = kwargs["debug"]
    duration = kwargs["duration"]
    main_step = kwargs["main_step"]
    secondary_step = kwargs["secondary_step"]

    part_dirpath = dirpath.split(os.sep)
    finished_dir = os.path.join(os.sep.join(part_dirpath[:-1]), "finished")
    os.makedirs(finished_dir, exist_ok=True)
    descriptor = HogDescriptor(
            orientation=config.ORIENTATION,
            pixels_per_cell=config.PPC,
            cells_per_block=config.CPB
    )
    pupil_finder = PupilFinder(
            descriptor=descriptor,
            svm_classifier_path=config.SVM_CLASSIFIER_PATH,
            win_size=config.WIN_SIZE,
            main_step_size=main_step,
            secondary_step_size=secondary_step,
            img_width=config.IMG_WIDTH, img_height=config.IMG_HEIGHT,
            channel_type=config.COLOR_CHANNEL, blur_kernel=config.BLUR_KERNEL, blur_type=config.BLUR_TYPE,
            svm_kernel_type=config.KERNEL_TYPE,
            debug=debug)

    logger.info("[PROCESSED] dirpath {}".format(dirpath))

    connection = pymongo.MongoClient(config.MONGO_URL)
    db = connection.livestockwatch
    capture_session = db.capture_session

    acquisition_type = "nopl"
    nopl_result_path = os.path.join(dirpath, acquisition_type)
    if os.path.exists(nopl_result_path):
        shutil.rmtree(nopl_result_path)
    os.makedirs(nopl_result_path, exist_ok=True)
    video_path = os.path.join(dirpath, filename)
    print("Start processing {}".format(video_path))
    video = cv2.VideoCapture(video_path)
    first_eye_timestamp = None
    frame_count = 1
    cs_name = part_dirpath[-2]
    lst_dct_result = []
    while True:
        (grabbed, frame) = video.read()

        if not grabbed:
            break

        frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.00
        if frame_timestamp == 0.0:
            frame_timestamp = frame_count * 0.06666666666666667

        frame_timestamp = float("{:.2f}".format(frame_timestamp))
        filename = "{}_{}".format(frame_count, frame_timestamp)

        if not first_eye_timestamp and frame_timestamp > 0.5:
            final_eye_flag, dct_result = pupil_finder.analyze_pupil(frame, nopl_result_path, filename)
            if final_eye_flag and dct_result["area"] is not np.nan:
                print("Set First frame timestamp at {}s".format(frame_timestamp))
                cs_name = dct_result["cs_name"]
                first_eye_timestamp = frame_timestamp
                dct_result["firstframe"] = True
                lst_dct_result.append(dct_result)

        elif first_eye_timestamp:
            if frame_timestamp < duration:
                print("Analyzing frame_timestamp {}".format(frame_timestamp))
                final_eye_flag, dct_result = pupil_finder.analyze_pupil(
                        frame, nopl_result_path, filename
                )
                lst_dct_result.append(dct_result)

        frame_count += 1

    video.release()

    for dct_result in lst_dct_result:
        retry = True
        while retry:
            try:
                capture_session.update_one(
                        {"_id": dct_result["_id"]},
                        {"$set": dct_result},
                        upsert=True
                )
                retry = False
            except NameError as e:
                print("[BREAK], {} {}".format(type(e), e))
                retry = False
            except ConnectionResetError as e:
                print("[RETRY], {} {}".format(type(e), e))
                connection.close()
                connection = pymongo.MongoClient(config.MONGO_URL)
                db = connection.livestockwatch
                capture_session = db.capture_session
                retry = True

    cs_df = odo(lst_dct_result, pd.DataFrame)
    cs_df = cs_df[cs_df.cs_name == cs_name]
    cs_df.sort_values("framecount", ascending=True, inplace=True)

    svg_major_axis = svg_smoothing(cs_df.pupil_major_axis)
    svg_minor_axis = svg_smoothing(cs_df.pupil_minor_axis)
    for i, idx in enumerate(cs_df.index):
        print("i = {} idx ={}".format(i, idx))
        major_axis = svg_major_axis[i]
        minor_axis = svg_minor_axis[i]

        if major_axis is None or minor_axis is None:
            pass
        elif major_axis < 0 or minor_axis < 0:
            pass
        elif major_axis - minor_axis < 0:
            pass
        else:
            logging.info("Axis checker pass")
            ea = EllipseAnalysis(svg_major_axis[i], svg_minor_axis[i])
            cs_df.set_value(idx, "svg_pupil_major_axis", svg_major_axis[i])
            cs_df.set_value(idx, "svg_pupil_minor_axis", svg_minor_axis[i])
            cs_df.set_value(idx, "svg_ipr", ea.ipr)
            cs_df.set_value(idx, "svg_area", ea.area)
            cs_df.set_value(idx, "svg_eccentricity", ea.eccentricity)
            cs_df.set_value(idx, "svg_perimeter", ea.perimeter)

    max_area = cs_df.area.max()
    svg_max_area = cs_df.svg_area.max()

    for idx in cs_df.index:
        cs_df.set_value(idx, "max_area", max_area)
        cs_df.set_value(idx, "svg_max_area", max_area)

        ca = ellipse_calculate_ca(cs_df.loc[idx]["area"], max_area)
        svg_ca = ellipse_calculate_ca(cs_df.loc[idx]["svg_area"], svg_max_area)

        cs_df.set_value(idx, "ca", ca)
        cs_df.set_value(idx, "svg_ca", svg_ca)
        while True:
            try:
                capture_session.find_one_and_update(
                        {'_id': cs_df.loc[idx]["_id"]},
                        {"$set": {
                            "ca": cs_df.loc[idx]["ca"],
                            "svg_ca": cs_df.loc[idx]["svg_ca"],
                            "max_area": cs_df.loc[idx]["max_area"],
                            "svg_max_area": cs_df.loc[idx]["svg_max_area"],

                            "svg_area": cs_df.loc[idx]["svg_area"],
                            "svg_pupil_major_axis": cs_df.loc[idx]["svg_pupil_major_axis"],
                            "svg_pupil_minor_axis": cs_df.loc[idx]["svg_pupil_minor_axis"],
                            "svg_ipr": cs_df.loc[idx]["svg_ipr"],
                            "svg_eccentricity": cs_df.loc[idx]["svg_eccentricity"],
                            "svg_perimeter": cs_df.loc[idx]["svg_perimeter"],
                        }}
                )
                break
            except ConnectionResetError as e:
                connection.close()
                connection = pymongo.MongoClient(config.MONGO_URL)
                db = connection.livestockwatch
                capture_session = db.capture_session
                print("[RETRY], {} {}".format(type(e), e))
                pass

    csv_path = os.path.join(dirpath, "{}.csv".format(cs_name))
    cs_df.to_csv(csv_path, encoding='utf-8')
    connection.close()
    shutil.move(dirpath, finished_dir, )
    logger.info("Finish processing {} done in {:.3f} minute".format(video_path, (time() - start_time) / 60))


class PupilFinder:
    BBOX_COLOR_POSITIVE = [
        (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 255, 255),
    ]
    BBOX_COLOR_NEGATIVE = (0, 0, 255)

    IMG_CLASS = ["negative", "positive"]

    def __init__(self, descriptor, svm_classifier_path,
                 win_size, main_step_size, secondary_step_size,
                 img_width, img_height,
                 channel_type, blur_kernel, blur_type,
                 svm_kernel_type, debug=False, extent_pixel=32):
        self.loop = asyncio.get_event_loop()
        self.svm_classifier = joblib.load(svm_classifier_path)
        self.descriptor = descriptor
        self.win_size = win_size
        self.main_step_size = main_step_size
        self.secondary_step_size = secondary_step_size
        self.img_width = img_width
        self.img_height = img_height
        self.channel_type = channel_type
        self.blur_kernel = blur_kernel
        self.blur_type = blur_type
        self.svm_kernel_type = svm_kernel_type
        self.debug = debug
        self.extent_pixel = extent_pixel

    def _gen_convol(self, start_point=(0, 0), win_size=None, step_size=None, img_width=None, img_height=None):
        if not win_size:
            win_size = self.win_size

        if not step_size:
            step_size = self.main_step_size

        if not img_height:
            img_height = self.img_height

        if not img_width:
            img_width = self.img_width

        x_points = [(a, a + win_size) for a in range(step_size - 1, img_width, step_size)
                    if a + win_size < img_width]

        y_points = [(a, a + win_size) for a in range(step_size - 1, img_height, step_size)
                    if a + win_size < img_height]

        if start_point[0] == 0:
            x_points.insert(0, (0, win_size))
        if start_point[1] == 0:
            y_points.insert(0, (0, win_size))

        points = []
        for y_point in y_points:
            for x_point in x_points:
                assert y_point[1] - y_point[0] == win_size, "patch height {} != {}".format(
                        y_point[1] - y_point[0], win_size
                )
                assert x_point[1] - x_point[0] == win_size, "patch width {} != {}".format(
                        x_point[1] - x_point[0], win_size
                )
                points.append(((y_point[0], y_point[1]), (x_point[0], x_point[1])))

        logger.info("Gen {} convol using win_size = {}, step_size = {}, img_height = {}, img_width = {}".format(
                len(points), win_size, step_size, img_height, img_width
        ))

        return points

    def is_overlap(self, bb1, bb2):
        """Overlapping rectangles overlap both horizontally & vertically
        """
        bb1y, bb1x = bb1
        bb2y, bb2x = bb2

        # (x[0], y[0]), (x[1], y[1])
        h_overlaps = bb1x[0] in range(bb2x[0], bb2x[1]) or bb1x[1] in range(bb2x[0], bb2x[1])
        v_overlaps = bb1y[0] in range(bb2y[0], bb2y[1]) or bb1y[1] in range(bb2y[0], bb2y[1])
        return h_overlaps and v_overlaps

    def get_union_bbox(self, lst_output):
        # if there are no boxes, return an empty list
        if len(lst_output) == 0:
            return []

        # initialize the list of picked indexes
        points = [point for _, _, point in lst_output]
        boxes = [(x[0], y[0], x[1], y[1]) for y, x in points]
        box_ids = [x for x, _ in enumerate(lst_output)]
        union = []

        # if len(box_ids) == 1:
        #     return boxes

        for box_id in box_ids:
            lst_overlap = []
            for other_id in box_ids:
                if other_id != box_id:
                    if self.is_overlap(points[box_id], points[other_id]):
                        lst_overlap.append((box_id, other_id))
            union.append(lst_overlap)

        bbox_union = []
        if len(union) > 0:
            for x in union:
                bbox_area = np.array([boxes[idx] for idx in np.unique(x)])
                if len(bbox_area) > 0:
                    x_start = np.min(bbox_area[:, 0])
                    y_start = np.min(bbox_area[:, 1])
                    x_end = np.max(bbox_area[:, 2])
                    y_end = np.max(bbox_area[:, 3])
                    bbox_union.append((x_start, y_start, x_end, y_end))

        return [x for x in set(bbox_union)]

    def check_patch_async(self, img_channel, points):
        output = []

        async def check_patch(img_channel, point):
            t0 = time()
            # logger.info("[START] Classify patch ")

            y, x = point
            patch = img_channel[y[0]:y[1], x[0]:x[1]]

            img_feature = self.descriptor.describe(patch).reshape(1, -1)

            eye_flag = self.svm_classifier.predict(img_feature)[0]
            confidence = self.svm_classifier.decision_function(img_feature)

            # logger.info("[FINISH] done in %0.3fs" % (time() - t0))
            return confidence, eye_flag, point

        async def gather_result(img_channel, point):
            result = await check_patch(img_channel, point)
            confidence, eye_flag, point = result
            if eye_flag:
                if confidence > 0.1:
                    output.append(result)

        tasks = [asyncio.ensure_future(gather_result(img_channel, point)) for point in points]
        self.loop.run_until_complete(asyncio.wait(tasks))

        return output

    def analyze_pupil(self, image, result_path, file_name, dummy=False):
        ellipse_fit_pathr = os.path.join(result_path, "ellipse_fitting")
        os.makedirs(ellipse_fit_pathr, exist_ok=True)

        part_dirpath = result_path.split(os.sep)
        totd = part_dirpath[-4]
        cattle_id = part_dirpath[-3]
        cs_name = part_dirpath[-2]
        cs_name_part = cs_name.split("_")
        stall_name = cs_name_part[0]
        frametime = float(file_name.split("_")[1])
        framecount = int(file_name.split("_")[0])

        timestamp = datetime.datetime(
                year=int(cs_name_part[1]),
                month=int(cs_name_part[2]),
                day=int(cs_name_part[3]),
                hour=int(cs_name_part[4]),
                minute=int(cs_name_part[5]),
                second=int(cs_name_part[6]),
                tzinfo=LOCAL_TIMEZONE
        )

        dct_result = OrderedDict([
            ("_id", "_".join([cs_name, str(frametime)])),
            ("cs_name", cs_name),
            ("cattle_id", cattle_id),
            ("stall_name", stall_name),
            ("totd", totd),
            ("timestamp", timestamp),
            ("firstframe", False),
            ("frametime", frametime),
            ("framecount", framecount),

            # Non savitzky golay
            ("ca", np.nan),
            ("ipr", np.nan),
            ("area", np.nan),
            ("max_area", np.nan),
            ("eccentricity", np.nan),
            ("perimeter", np.nan),
            ("pupil_major_axis", np.nan),
            ("pupil_minor_axis", np.nan),

            # With savitzky golay
            ("svg_ca", np.nan),
            ("svg_ipr", np.nan),
            ("svg_area", np.nan),
            ("svg_max_area", np.nan),
            ("svg_eccentricity", np.nan),
            ("svg_perimeter", np.nan),
            ("svg_pupil_major_axis", np.nan),
            ("svg_pupil_minor_axis", np.nan),

            # contour data
            ("angle", np.nan),
            ("aspect_ratio", np.nan),
            ("extent", np.nan),
            ("solidity", np.nan),
            ("contour_area", np.nan),
            ("contour_perimeter", np.nan),
        ])

        if dummy:
            return dct_result

        final_eye_flag, _, final_patch_equ = self.detect_pupil(image, result_path, file_name)
        if final_eye_flag:
            major_axis, minor_axis, angle, ellipse_img, cnt = pupil_fit_ellipse(final_patch_equ)
            if major_axis is None or minor_axis is None:
                return False, dct_result

            if major_axis < 0 or minor_axis < 0:
                return False, dct_result

            if major_axis - minor_axis < 0:
                return False, dct_result

            write_as_png(os.path.join(ellipse_fit_pathr, "{}.png".format(file_name)), ellipse_img)
            ea = EllipseAnalysis(major_axis, minor_axis)
            dct_result["ipr"] = ea.ipr
            dct_result["area"] = ea.area
            dct_result["eccentricity"] = ea.eccentricity
            dct_result["perimeter"] = ea.perimeter
            dct_result["pupil_major_axis"] = major_axis
            dct_result["pupil_minor_axis"] = minor_axis
            dct_result["angle"] = angle

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            dct_result["aspect_ratio"] = aspect_ratio

            contour_area = cv2.contourArea(cnt)
            extent = float(contour_area) / float(w * h)
            dct_result["extent"] = extent

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(contour_area) / hull_area
            dct_result["solidity"] = solidity

            dct_result["contour_area"] = contour_area

            perimeter = cv2.arcLength(cnt, True)
            dct_result["contour_perimeter"] = perimeter

        return final_eye_flag, dct_result

    def detect_pupil(self, image, result_path, file_name):
        final_eye_flag = False
        lst_final_output = []
        final_point = None
        final_extended_patch = None

        untouched_image = image.copy()
        candidate_image = image.copy()
        final_image = image.copy()

        final_path = os.path.join(result_path, "final")
        final_extended_path = os.path.join(result_path, "final_extended")

        bbox_path = os.path.join(result_path, "bbox_img")
        bbox_final_path = os.path.join(bbox_path, "final")
        bbox_candidate_path = os.path.join(bbox_path, "candidate")
        raw_path = os.path.join(result_path, "raw")
        raw_hist_equ_path = os.path.join(result_path, "raw_hist_equ")
        union_path = os.path.join(result_path, "union")

        too_dark_path = os.path.join(result_path, "too_dark")
        positive_path = os.path.join(result_path, "positive")
        hard_neg_path = os.path.join(result_path, "hard_neg")
        hist_equ_path = os.path.join(result_path, "hist_equ")
        hist_equ_blur_path = os.path.join(result_path, "hist_equ_blur")

        folder_list = [result_path, final_path, final_extended_path]

        if self.debug:
            folder_list = folder_list + [union_path, bbox_path, positive_path, hard_neg_path,
                                         bbox_final_path, bbox_candidate_path, raw_path, raw_hist_equ_path,
                                         too_dark_path, hist_equ_blur_path, hist_equ_path]

        for folder in folder_list:
            os.makedirs(folder, exist_ok=True)

        t0 = time()
        logger.info("[START] Pupil Detection")

        if self.debug:
            # Save raw frame
            write_as_png(os.path.join(raw_path, "{}.png".format(file_name)), untouched_image)
            # Save equ frame
            write_as_png(os.path.join(raw_hist_equ_path, "{}.png".format(file_name)),
                         equ_hist_color_image(untouched_image))

        img_channel = get_specific_channel(image, self.channel_type,
                                           blur_kernel=self.blur_kernel, blur_type=self.blur_type)
        print(img_channel.shape)
        assert img_channel.shape == (self.img_height, self.img_width)

        points = self._gen_convol()
        output = self.check_patch_async(img_channel, points)

        best_output = sorted(output, reverse=True)[:3]
        false_output = sorted(output, reverse=True)[3:]
        if self.debug:

            for count, output in enumerate(best_output):
                confidence, eye_flag, point = output
                y, x = point
                cv2.rectangle(candidate_image, (x[0], y[0]), (x[1], y[1]), PupilFinder.BBOX_COLOR_POSITIVE[count], 3)
                cv2.putText(candidate_image, "{0:.2f}".format(confidence[0]),
                            (int(x[0] + self.win_size / 2), int(y[0] + self.win_size / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, PupilFinder.BBOX_COLOR_POSITIVE[count], 3)
                pos_patch = untouched_image[y[0]:y[1], x[0]:x[1]]
                result_name = "{:.2f}_{}.png".format(confidence[0], file_name)
                write_as_png(os.path.join(positive_path, result_name), pos_patch)

            write_as_png(os.path.join(bbox_candidate_path, "{}.png".format(file_name)), candidate_image)

            for count, output in enumerate(false_output):
                confidence, eye_flag, point = output
                y, x = point
                hard_neg_patch = untouched_image[y[0]:y[1], x[0]:x[1]]
                result_name = "{:.2f}_{}.png".format(confidence[0], file_name)
                write_as_png(os.path.join(hard_neg_path, result_name), hard_neg_patch)

        if len(best_output) == 1:
            for count, output in enumerate(best_output):
                confidence, eye_flag, point = output
                y, x = point
                final_image = candidate_image
                lst_final_output.append((confidence, (x[0], y[0], x[1], y[1])))

        else:
            pprint(best_output)
            pick = self.get_union_bbox(best_output)
            for (startX, startY, endX, endY) in pick:
                cv2.rectangle(final_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                union_patch = untouched_image[startY:endY, startX:endX]
                result_name = "{}.png".format(file_name)
                if self.debug:
                    write_as_png(os.path.join(union_path, result_name), union_patch)
                union_channel = get_specific_channel(union_patch, self.channel_type,
                                                     blur_kernel=self.blur_kernel, blur_type=self.blur_type)
                fp_height, fp_width = union_patch.shape[:2]

                fp_points = self._gen_convol(step_size=self.secondary_step_size,
                                             img_width=fp_width, img_height=fp_height)
                fp_output = self.check_patch_async(union_channel, fp_points)

                if len(fp_output) == 0:
                    for count, output in enumerate(best_output[:1]):
                        confidence, eye_flag, point = output
                        y, x = point
                        lst_final_output.append((confidence, (x[0], y[0], x[1], y[1])))
                else:
                    fp_best_output = sorted(fp_output, reverse=True)[:1]
                    for count, output in enumerate(fp_best_output[:1]):
                        confidence, eye_flag, point = output
                        y, x = point
                        fp_startX = startX + x[0]
                        fp_startY = startY + y[0]
                        fp_endX = fp_startX + self.win_size
                        fp_endY = fp_startY + self.win_size
                        lst_final_output.append((confidence, (fp_startX, fp_startY, fp_endX, fp_endY)))

        lst_final_output = sorted(lst_final_output, reverse=True)[:1]
        for confidence, point in lst_final_output:
            final_point = point
            final_eye_flag = True
            final_patch = untouched_image[point[1]:point[3], point[0]:point[2]]
            cv2.rectangle(final_image, (point[0], point[1]), (point[2], point[3]), (0, 255, 255), 3)

            result_name = "{}.png".format(file_name)
            write_as_png(os.path.join(final_path, result_name), final_patch)

            final_extended_patch = self.extend_patch(untouched_image, point, self.extent_pixel)
            write_as_png(os.path.join(final_extended_path, result_name), final_extended_patch)

            if self.debug:
                final_patch_equ = equ_hist_color_image(final_extended_patch)
                blurred = cv2.bilateralFilter(final_patch_equ, 50, 75, 75)
                blurred = cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
                write_as_png(os.path.join(hist_equ_blur_path, result_name), blurred)
                write_as_png(os.path.join(hist_equ_path, result_name), final_patch_equ)

        if self.debug:
            write_as_png(os.path.join(bbox_final_path, "{}.png".format(file_name)), final_image)

        logger.info("[FINISH] Pupil Detection done in %0.3fs" % (time() - t0))

        return final_eye_flag, final_point, final_extended_patch

    def extend_patch(self, img, point, extent_px):
        half_extent_px = extent_px / 2
        h, w = img.shape[:2]
        startX = point[0]
        startY = point[1]
        endX = point[2]
        endY = point[3]

        check_startX = startX - half_extent_px
        check_startY = startY - half_extent_px
        check_endX = (w - 1) - (endX + half_extent_px)
        check_endY = (h - 1) - (endY + half_extent_px)

        if check_startX < 0:
            left_overX = abs(check_startX)
            startX -= half_extent_px - left_overX
            endX = endX + left_overX + half_extent_px
        elif check_endX < 0:
            left_overX = abs(check_endX)
            endX = w - 1
            startX = startX - left_overX - half_extent_px
        else:
            startX -= half_extent_px
            endX += half_extent_px

        if check_startY < 0:
            left_overY = abs(check_startY)
            startY -= half_extent_px - left_overY
            endY = endY + left_overY + half_extent_px
        elif check_endY < 0:
            left_overY = abs(check_endY)
            endY = h - 1
            startY = startY - left_overY - half_extent_px
        else:
            startY -= half_extent_px
            endY += half_extent_px

        extended_patch = img[startY:endY, startX:endX]
        try:
            assert extended_patch.shape[:2] == ((self.win_size + extent_px), (self.win_size + extent_px))
        except AssertionError as e:
            print(e)
            embed()
        return img[startY:endY, startX:endX]
