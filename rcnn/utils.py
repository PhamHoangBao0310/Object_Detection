import cv2
import numpy as np

def compute_iou(boxA, boxB):
  """
    Compute IOU between 2 boxes

    Args:
      boxA, boxB: 2 input boxes

    Returns:
      IOU of 2 boxes
  """
  # Determine Inter area from 2 boxes
  x1 = min(boxA[0], boxB[0])
  y1 = max(boxA[1], boxB[1])
  x2 = min(boxA[2], boxB[2])
  y2 = max(boxA[3], boxB[3])

  # Compute area of inter area
  inter_area = max(0, x2 - x1) * max(0, y2 - y1)

  # Compute box A and box B area
  boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

  # Compute IOU
  iou = inter_area / float(boxA_area + boxB_area - inter_area)

  return iou

def selective_search(image, method = "fast"):
    """
        Selective search with openCV

        Args:
            image: numpy array. Input image.
            method : "fast" or "quality". Method for selective search
        Returns:
            list of bounding box
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if method == "fast":
	    print("[INFO] using *fast* selective search")
	    ss.switchToSelectiveSearchFast()
    else:
	    print("[INFO] using *quality* selective search")
	    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    return rects

def get_area(bounding_box):
  return (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])