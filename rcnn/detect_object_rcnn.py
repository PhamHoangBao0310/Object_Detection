import config
from model import InferenceModel
from utils import selective_search, non_max_supression
import cv2
import numpy as np
import time


def full_flow(image_path):
  flow_start = time.time()
  print("[INFO] Staring detection")
  image = cv2.imread(image_path, 1)
  # print(image is None)

  start = time.time()
  print("[INFO] Load model")
  model = InferenceModel("E:\\FPTdocument\\Document\\Machine Learning\\AIVN\\object_detection\\Object_Detection\\rcnn\\my_model.h5", "MobileNet")
  print("[INFO] Load model successfully : {}".format(time.time() - start))

  # Perform selective search
  start = time.time()
  print("[INFO] Perform Selective Search")
  rects = selective_search(image)
  print("[INFO] Perform Selective Search successfully : {}".format(time.time() - start))

  # Get max proposal boxes
  start = time.time()
  print("[INFO] Get max proposal boxes")
  proposals = []
  boxes = []

  for (x, y, w, h) in rects[:config.MAX_PROPOSALS_INFER]:
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, config.INPUT_DIMS,
                  interpolation=cv2.INTER_CUBIC)

    proposals.append(roi)
    boxes.append([x, y, x + w, y + h])

  proposals = np.array(proposals, dtype="float32")
  boxes = np.array(boxes, dtype="int32")
  print("[INFO] Get max proposal boxes sucessfully : {}".format(time.time() - start))
  print("[INFO] proposal shape: {}".format(proposals.shape))

  start = time.time()
  print("[INFO] classifying proposals...")
  proba = model.predict(proposals)
  print("[INFO] classifying proposals sucessfully: {}".format(time.time() - start))

  start = time.time()
  print("[INFO] Get more accurate proposals...")
  labels = np.argmax(proba, axis = 1)
  indx = np.squeeze(np.argwhere(labels == 1)) # 1 is racoon
  bbxs = boxes[indx]
  prob = proba[indx]
  idx = np.squeeze(np.argwhere(prob[:,1] > config.MIN_PROBA))
  bbxs = bbxs[idx]
  prob = prob[idx][:,1]
  print("[INFO] Get more accurate proposals sucessfully : {}".format(time.time() - start))

  start = time.time()
  print("[INFO] Perform Non maximum supression")
  bbxs , prob = non_max_supression(bbxs, prob, 0.5)
  print("[INFO] Perform Non maximum supression sucessfully : {}".format(time.time() - start))

  start = time.time()
  print("[INFO] Draw boxes")
  for prob, bb in zip(prob,bbxs):
    (startX, startY, endX, endY) = bb
    cv2.rectangle(image, (startX, startY), (endX, endY), (255,255,0), 1)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    text= "Raccoon: {:.2f}%".format(prob * 100)
    cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
  print("[INFO] Draw boxes sucessfully : {}".format(time.time() - start))

  print("[INFO] Finish detection : {}".format(time.time() - flow_start))
  cv2.imshow("racoon", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()



if __name__ == '__main__':
  full_flow("E:\\FPTdocument\\Document\\Machine Learning\\AIVN\\object_detection\\Object_Detection\\data\\images\\raccoon-2.jpg")
