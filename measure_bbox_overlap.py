import os
import cv2
from skimage.filters import threshold_adaptive
import csv
from livestockwatch.pupil_finder import PupilFinder
from livestockwatch.descriptors import HogDescriptor
from livestockwatch.utils import Rectangle, measure_overlap_ratio
from IPython import embed

orientation = 9
ppc = 4
cpb = 1
kernel_type = "linear"
blur_kernel = 15
blur_type = "bilateral"
color_channel = "b_star_hist_equal"
svm_classifier_path = os.path.join("hog_model",
                                   "{}".format(kernel_type),
                                   "o{}_ppc{}_cpb{}".format(orientation, ppc, cpb),
                                   "{}".format(color_channel),
                                   "hog_svm.pkl")
debug = False
win_size = 256
step_size = 64
img_width = 1280
img_height = 960
analysis_type = "partial"
max_timestamp = 2.0
step_second = 1.0

skip_folder = ["pending", "finished",
               "nopl", "pl",
               "bbox_img", "candidate", "final",
               "hard_neg", "raw", "hist_equ", "hist_equ_blur", "positive", "raw", "union"]

descriptor = HogDescriptor(orientation=orientation, pixels_per_cell=ppc, cells_per_block=cpb)
pupil_finder = PupilFinder(descriptor, svm_classifier_path=svm_classifier_path,
                           win_size=win_size, step_size=step_size,
                           img_width=img_width, img_height=img_height,
                           channel_type=color_channel, blur_kernel=blur_kernel, blur_type=blur_type,
                           svm_kernel_type=kernel_type,
                           debug=debug)

data_folder = "test_bbox_overlap"
annotation_path = os.path.join(data_folder, "annotations.csv")
real_bbox_ann_dict = {}
if os.path.exists(annotation_path):
    with open(annotation_path, newline='\n') as csvfile:
        annotation_reader = csv.reader(csvfile)
        for row in annotation_reader:
            if row[0] == "filename":
                continue
            else:
                real_bbox_ann_dict[row[0]] = row[1], row[2], row[3], row[4]

for dirpath, dirnames, files in os.walk(data_folder):
    for filename in files:
        if len(dirpath.split(os.path.sep)) == 1:
            if (filename.startswith("pl") or filename.startswith("nopl")) and filename.endswith(".bmp"):
                if filename.startswith("pl"):
                    acquisition_type = "pl"
                elif filename.startswith("nopl"):
                    acquisition_type = "nopl"

                pl_result_path = os.path.join(dirpath, "{}".format(kernel_type),
                                              "o{}_ppc{}_cpb{}".format(orientation, ppc, cpb),
                                              color_channel, acquisition_type)
                os.makedirs(pl_result_path, exist_ok=True)
                img_path = os.path.join(dirpath, filename)

                img = cv2.imread(img_path)
                untouched_image = img.copy()
                final_eye_flag, predicted_point, final_patch_equ = pupil_finder.detect_pupil(
                        img, pl_result_path, filename
                )

                img_type, img_class, real_startX, real_startY = real_bbox_ann_dict[filename]
                real_startX = int(real_startX)
                real_startY = int(real_startY)
                real_endX = real_startX + win_size
                real_endY = real_startY + win_size
                real_point = (real_startX, real_startY, real_endX, real_endY)
                cv2.rectangle(img,
                              (predicted_point[0], predicted_point[1]),
                              (predicted_point[2], predicted_point[3]),
                              (0, 0, 255), 3)
                cv2.rectangle(img, (real_point[0], real_point[1]), (real_point[2], real_point[3]), (0, 255, 0), 3)
                cv2.imshow("Image", img)
                cv2.waitKey(0)
                cv2.destroyWindow("Image")
                pred_rect = Rectangle(predicted_point)
                real_rect = Rectangle(real_point)
                overlap_flag, overlap_ratio = measure_overlap_ratio(pred_rect, real_rect, win_size)
                print("Prediction rect = {}, Real rect = {}".format(predicted_point, real_point))
                print("overlap flag = {}, overlap ratio = {:.2f} px".format(overlap_flag, overlap_ratio))

                pupil_bbox = untouched_image[predicted_point[1]:predicted_point[3],
                                             predicted_point[0]:predicted_point[2]]
                pupil_bbox_untouched = pupil_bbox.copy()
                cv2.imshow("pupil_bbox", pupil_bbox)
                cv2.waitKey(0)

# img_path = os.path.join(data_folder, "nopl1.bmp")
# img = cv2.imread(img_path)
# predicted_point = (0, 0, 256, 256)
# predicted_patch = img[predicted_point[1]:predicted_point[3], predicted_point[0]:predicted_point[2]]
# assert predicted_patch.shape[:2] == (256, 256)
#
# real_point = (338, 382, 594, 638)
# real_patch = img[real_point[1]:real_point[3], real_point[0]:real_point[2]]
# assert real_patch.shape[:2] == (256, 256)
#
# cv2.rectangle(img, (predicted_point[0], predicted_point[1]), (predicted_point[2], predicted_point[3]), (0, 0, 255), 3)
# cv2.rectangle(img, (real_point[0], real_point[1]), (real_point[2], real_point[3]), (0, 255, 0), 3)
#
# pred_rect = Rectangle(predicted_point)
# real_rect = Rectangle(real_point)
# overlap_flag, overlap_ratio = measure_overlap_ratio(pred_rect, real_rect, win_size)
#
# print("Prediction rect = {}, Real rect = {}".format(predicted_point, real_point))
# print("overlap flag = {}, overlap ratio = {:.2f} px".format(overlap_flag, overlap_ratio))
#
# cv2.imshow("Image", img)
# cv2.waitKey(0)
