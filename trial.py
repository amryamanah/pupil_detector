import os
import csv
import cv2
import uuid
from sklearn.externals import joblib
from livestockwatch.simple_classifier import SimpleClassifier
from livestockwatch.descriptors import HogDescriptor
from IPython import embed

import imutils



def gen_convol():
    x_points = [(a, a+256) for a in range(79, 1280, 16) if a+320 < 1280]
    y_points = [(a, a+256) for a in range(79, 960, 16) if a+320 < 960]

    x_points.insert(0,(0, 255))
    y_points.insert(0,(0, 255))

    return x_points, y_points


descriptor = HogDescriptor(orientation=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2))

classifier = SimpleClassifier("dataset_oct", 238, 118, "gray", descriptor)
classifier.load_dataset()
classifier.training()

svm_classifier_path = os.path.join("hog_model", "hog_svm.pkl")
svm_classifier = joblib.load(svm_classifier_path)

video_path = os.path.join("dataset_oct", "video", "nopl10", "10.avi")
print(video_path)
video = cv2.VideoCapture(video_path)
frame_count = 0
while True:
    (grabbed, frame) = video.read()

    if not grabbed:
        print(video_path)
        print("Not grabbed")
        break

    frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.00
    if frame_timestamp > 0.5:
        lst_bbox = []
        x_points, y_points = gen_convol()
        result_img = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
                img_feature = descriptor.describe(patch)
                eye_flag = svm_classifier.predict(img_feature)[0]
                confidence = svm_classifier.decision_function(img_feature)
                if eye_flag:
                    bbox = (confidence, (x_point[0], y_point[0]), (x_point[1], y_point[1]),
                            frame[y_point[0]:yend, x_point[0]:xend])
                    lst_bbox.append(bbox)

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

        res_filename = os.path.join("result", "bbox_img", "{}.bmp".format(frame_timestamp))
        cv2.imwrite(res_filename, result_img)
        result_img = imutils.resize(result_img, 600)

        cv2.imshow("Video", result_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Bye-bye")
        break

video.release()
cv2.destroyWindow("video")


