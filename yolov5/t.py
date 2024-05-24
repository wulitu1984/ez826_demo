import os
import sys
import cv2
import random

names = [
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"potted plant",
"bed",
"dining table",
"toilet",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
]

paths = os.listdir("result")
paths = random.sample(paths, 10)
for path in paths:
    name = os.path.basename(path)
    name = name[:-4]
    print(name)

    image = os.path.join("images", name+".jpg")
    label = os.path.join("labels", name+".txt")
    result = os.path.join("result", name+".txt")
    result_img = os.path.join("result_img", name+".jpg")


    im = cv2.imread(image)
    with open(label, "r") as f:
        for line in f:
            class_id, x1, y1, x2, y2 = list(map(float, line.split()))
            # cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # cv2.putText(im, str(int(class_id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    with open(result, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            class_id, conf, x1, y1, x2, y2 = list(map(float, line.split()))
            if conf > 0.4:
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                cv2.putText(im, str(names[int(class_id)])+" "+str(round(conf, 3)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("", im)
    cv2.waitKey()
    cv2.imwrite(result_img, im)
