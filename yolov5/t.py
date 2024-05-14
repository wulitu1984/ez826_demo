import os
import sys
import cv2


image = os.path.join("images", sys.argv[1]+".jpg")
label = os.path.join("labels", sys.argv[1]+".txt")
result = os.path.join("result", sys.argv[1]+".txt")


im = cv2.imread(image)
with open(label, "r") as f:
    for line in f:
        class_id, x1, y1, x2, y2 = list(map(float, line.split()))
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(im, str(int(class_id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
with open(result, "r") as f:
    for line in f:
        class_id, conf, x1, y1, x2, y2 = list(map(float, line.split()))
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
        cv2.putText(im, str(int(class_id)), (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
cv2.imshow("", im)
cv2.waitKey()
