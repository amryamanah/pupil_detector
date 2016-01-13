import os
import cv2

import numpy as np
from IPython import embed

from livestockwatch.descriptors import DenseLBPDescriptor
from sklearn.externals import joblib
import imutils
import uuid


class BowPupilFinder:
    BBOX_COLOR = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 255, 255), (0, 0, 0), (128, 128, 128)
    ]

    IMG_CLASS = ["negative", "positive"]

    def __init__(self, lbp_point, lbp_radius, cluster_formation_path, svm_classifier_path, step):
        self.cluster_formation = joblib.load(cluster_formation_path)
        self.svm_classifier = joblib.load(svm_classifier_path)
        self.lbp_descriptor = DenseLBPDescriptor(lbp_point, lbp_radius)
        self.step = step # normally 80

    def _gen_convol(self):
        x_points = [(a, a+320) for a in range(79, 1280, self.step) if a+320 < 1280]
        y_points = [(a, a+320) for a in range(79, 960, self.step) if a+320 < 960]

        x_points.insert(0,(0, 319))
        y_points.insert(0,(0, 319))

        return x_points, y_points

    def _prepare_bg_image(self, image):
        b, g, r = cv2.split(image)
        r = np.zeros(b.shape, dtype=np.uint8)
        bgr = cv2.merge([b, g, r])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return gray

    def _prepare_gray_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def _specific_channel_img(self, image, channel_type):
        blue, green, red = cv2.split(image)
        hue, sat, val = cv2.split(image)
        if channel_type == "blue":
            return blue
        elif channel_type == "green":
            return green
        elif channel_type == "red":
            return red
        elif channel_type == "hue":
            return hue
        elif channel_type == "sat":
            return sat
        elif channel_type == "val":
            return val

    def check_patch(self, patch, image, x_point, y_point, yend, xend, lst_bbox):
        lbp_feature = self.lbp_descriptor.describe(patch)
        lbp_labels = self.cluster_formation.predict(lbp_feature)
        lbp_hist = np.empty((self.cluster_formation.n_clusters,))
        for x in range(0, self.cluster_formation.n_clusters):
            lbp_hist[x] = lbp_labels.tolist().count(x)
        eye_flag = self.svm_classifier.predict(lbp_hist)[0]
        confidence = self.svm_classifier.decision_function(lbp_hist)
        if eye_flag:
            bbox = (confidence, (x_point[0], y_point[0]), (x_point[1], y_point[1]),
                    image[y_point[0]:yend, x_point[0]:xend])
            lst_bbox.append(bbox)
        # else:
        #     res_filename = os.path.join("result", "patch", "negative", "{}_{}.bmp".format(confidence[0], uuid.uuid4()))
        #     # cv2.imwrite(res_filename, image[y_point[0]:yend, x_point[0]:xend])

    def detect_pupil(self, image, frame_timestamp, channel_type):
        lst_bbox = []
        x_points, y_points = self._gen_convol()
        result_img = image.copy()
        if channel_type == "bg":
            gray = self._prepare_bg_image(image)
        elif channel_type == "gray":
            gray = self._prepare_gray_image(image)
        else:
            gray = self._specific_channel_img(image, channel_type)

        for y_point in y_points:
            for x_point in x_points:
                if 0 in x_point:
                    xend = x_point[1] + 1
                else:
                    xend = x_point[1]

                if 0 in y_point:
                    yend = y_point[1] + 1
                else:
                    yend = y_point[1]

                patch = gray[y_point[0]:yend, x_point[0]:xend]
                # print("ypoint = {}, xpoint = {}, patch shape = {}".format(y_point, x_point, patch.shape))
                lbp_feature = self.lbp_descriptor.describe(patch)
                lbp_labels = self.cluster_formation.predict(lbp_feature)
                lbp_hist = np.empty((self.cluster_formation.n_clusters,))
                for x in range(0, self.cluster_formation.n_clusters):
                    lbp_hist[x] = lbp_labels.tolist().count(x)
                eye_flag = self.svm_classifier.predict(lbp_hist)[0]
                confidence = self.svm_classifier.decision_function(lbp_hist)
                if eye_flag:
                    bbox = (confidence, (x_point[0], y_point[0]), (x_point[1], y_point[1]),
                            image[y_point[0]:yend, x_point[0]:xend])
                    lst_bbox.append(bbox)
                # else:
                #     res_filename = os.path.join("result", "patch", "negative", "{}_{}.bmp".format(confidence[0], uuid.uuid4()))
                #     cv2.imwrite(res_filename, image[y_point[0]:yend, x_point[0]:xend])

        if len(lst_bbox) > 0:
            best_bbox = sorted(lst_bbox, reverse=True)[:1]
            other_bbox = sorted(lst_bbox, reverse=True)[1:]
            for confidence, start_point, end_point, patch in best_bbox:
                cv2.rectangle(result_img, start_point, end_point,
                              (255, 0, 0), 3)
                res_filename = os.path.join("result", "patch", "positive", "{}_{}.bmp".format(confidence[0], uuid.uuid4()))
                cv2.imwrite(res_filename, patch)

                cv2.putText(result_img, str(confidence), start_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                # if confidence > 0.85:
                #     cv2.rectangle(result_img, start_point, end_point,
                #                   (255, 0, 0), 3)
                #     res_filename = os.path.join("result", "patch", "positive", "{}_{}.bmp".format(confidence[0], uuid.uuid4()))
                #     cv2.imwrite(res_filename, patch)
                #
                #     cv2.putText(result_img, str(confidence), start_point,
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                # else:
                #     if len(other_bbox) > 0:
                #         other_bbox.append([confidence, start_point, end_point, patch])

            for confidence, start_point, end_point, patch in other_bbox:
                if 0.3 < confidence <= 0.85:
                    res_filename = os.path.join("result", "patch", "hard_negative", "{}_{}.bmp".format(confidence[0], uuid.uuid4()))
                    cv2.imwrite(res_filename, patch)

        res_filename = os.path.join("result", "bbox_img", "{}.bmp".format(frame_timestamp))
        cv2.imwrite(res_filename, result_img)

        result_img = imutils.resize(result_img, 600)
        return result_img