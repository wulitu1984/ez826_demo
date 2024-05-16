import os
import sys
import cv2
import random

paths = os.listdir("labels")
paths = random.sample(paths, 10)
for path in paths:
    name = os.path.basename(path)
    name = name[:-4]
    print(name)

    image = os.path.join("images", name+".jpg")
    label = os.path.join("labels", name+".txt")
    result = os.path.join("result", name+".txt")


    im = cv2.imread(image)
    with open(label, "r") as f:
        for line in f:
            class_id, x1, y1, x2, y2 = list(map(float, line.split()))
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(im, str(int(class_id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    with open(result, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            class_id, conf, x1, y1, x2, y2 = list(map(float, line.split()))
            if conf > 0.4:
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                cv2.putText(im, str(int(class_id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("", im)
    cv2.waitKey()
