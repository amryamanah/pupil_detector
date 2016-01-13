import os
import random
import cv2
import uuid

win_size = 256
max_x = 1279 - win_size
max_y = 959 - win_size

dataset_path = os.path.join("dataset_oct")

rand_pos_neg_path = os.path.join(dataset_path, "rand_pos_neg_{}".format(win_size))
rand_neg_path = os.path.join(dataset_path, "rand_neg_{}".format(win_size))

os.makedirs(rand_neg_path, exist_ok=True)
os.makedirs(rand_pos_neg_path, exist_ok=True)

kinds = ["pl_{}".format(win_size), "nopl_{}".format(win_size), "negative"]
for kind in kinds:
    for dirpath, dirnames, files in os.walk(os.path.join(dataset_path, kind)):
        for filename in files:
            if filename.endswith(".bmp"):
                img = cv2.imread(os.path.join(dirpath, filename))
                curr_iteration = 0
                print()
                while curr_iteration < 4:
                    x, y = random.randrange(max_x), random.randrange(max_y)
                    cropped = img[y:y+win_size, x:x+win_size]
                    if kind == "pl_{}".format(win_size):
                        res_filename = os.path.join(rand_pos_neg_path, "pl-{}.bmp".format(uuid.uuid4()))
                        print(res_filename)
                        cv2.imwrite(res_filename, cropped)
                    elif kind == "nopl_{}".format(win_size):
                        res_filename = os.path.join(rand_pos_neg_path, "nopl-{}.bmp".format(uuid.uuid4()))
                        cv2.imwrite(res_filename, cropped)
                    elif kind == "negative":
                        res_filename = os.path.join(rand_neg_path, "neg-{}.bmp".format(uuid.uuid4()))
                        cv2.imwrite(res_filename, cropped)
                    curr_iteration += 1

