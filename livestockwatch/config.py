
import os
# MONGO_URL = "mongodb://biseadmin:bise2014@162.243.1.142:48203/livestockwatch"
MONGO_URL = "mongodb://localhost:27017/livestockwatch_local"
ORIENTATION = 9
PPC = 4
CPB = 1
KERNEL_TYPE = "linear"
BLUR_KERNEL = 15
BLUR_TYPE = "bilateral"
COLOR_CHANNEL = "b_star_hist_equal"
SVM_CLASSIFIER_PATH = os.path.join("hog_model",
                                   "{}".format(KERNEL_TYPE),
                                   "o{}_ppc{}_cpb{}".format(ORIENTATION, PPC, CPB),
                                   "{}".format(COLOR_CHANNEL),
                                   "hog_svm.pkl")

DCT_CATTLE_ID = {
    "810": "1445186816",
    "811": "1445174448",
    "813": "1445174332",
    "814": "1445184485",
    "815": "1384106876",
    "816": "1372243682",
    "817": "1445174745",
    "799": "860053727",
}


WIN_SIZE = 256
IMG_WIDTH = 1280
IMG_HEIGHT = 960
max_timestamp = 2.0
STEP_SECOND = 1.0


SKIP_FOLDER = ["pending", "finished", "failed",
               "blink_noise", "lightning_noise", "dust_noise", "eyelashes_noise",
               "nopl", "pl",
               "bbox_img", "candidate", "final", "plr_result", "final_extended",
               "hard_neg", "raw", "hist_equ", "hist_equ_blur", "positive", "raw", "union"]
