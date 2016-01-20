import os
import uuid
from pprint import pprint
import logging
from time import time
import asyncio
import cv2

import numpy as np
from sklearn.externals import joblib
import imutils
from .utils import get_specific_channel, equ_hist_color_image, write_as_png, is_too_dark
from IPython import embed

logger = logging.getLogger(__name__)


class PupilFinder:
    BBOX_COLOR_POSITIVE = [
        (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 255, 255),
    ]
    BBOX_COLOR_NEGATIVE = (0, 0, 255)

    IMG_CLASS = ["negative", "positive"]

    def __init__(self, loop, descriptor, svm_classifier_path,
                 win_size, step_size,
                 img_width, img_height,
                 channel_type, blur_kernel, blur_type,
                 svm_kernel_type, debug=False, extent_pixel=32):
        self.loop = loop
        self.svm_classifier = joblib.load(svm_classifier_path)
        self.descriptor = descriptor
        self.win_size = win_size
        self.step_size = step_size
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
            step_size = self.step_size

        if not img_height:
            img_height = self.img_height

        if not img_width:
            img_width = self.img_width

        x_points = [(a, a+win_size) for a in range(step_size-1, img_width, step_size)
                    if a+win_size < img_width]

        y_points = [(a, a+win_size) for a in range(step_size-1, img_height, step_size)
                    if a+win_size < img_height]

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

    def detect_pupil(self, image, result_path, file_name):
        final_eye_flag = False
        lst_final_output = []
        final_point = None
        final_patch_equ = None

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
            write_as_png(os.path.join(raw_hist_equ_path, "{}.png".format(file_name)), equ_hist_color_image(untouched_image))

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
                            (int(x[0]+self.win_size/2), int(y[0]+self.win_size/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, PupilFinder.BBOX_COLOR_POSITIVE[count], 3)
                pos_patch = untouched_image[y[0]:y[1], x[0]:x[1]]
                result_name = "{:.2f}_{}.png".format(confidence[0], file_name)
                write_as_png(os.path.join(positive_path,result_name), pos_patch)

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
                fp_points = self._gen_convol(step_size=32, img_width=fp_width, img_height=fp_height)

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
                        fp_startX = startX+x[0]
                        fp_startY = startY+y[0]
                        fp_endX = fp_startX+self.win_size
                        fp_endY = fp_startY+self.win_size
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

        return final_eye_flag, final_point, final_patch_equ

    def extend_patch(self, img, point, extent_px):
        half_extent_px = extent_px/2
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
            endX = w-1
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
            endY = h-1
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










