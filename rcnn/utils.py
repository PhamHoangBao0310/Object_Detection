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
  # print("boxA : {} and boxB : {}".format(boxA, boxB))
  # Determine Inter area from 2 boxes
  x1 = max(boxA[0], boxB[0])
  y1 = max(boxA[1], boxB[1])
  x2 = min(boxA[2], boxB[2])
  y2 = min(boxA[3], boxB[3])

  # Compute area of inter area
  inter_area = max(0, x2 - x1) * max(0, y2 - y1)
  # print("inter area : {}".format(inter_area))

  # Compute box A and box B area
  boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

  # print("boxA : {} and boxB : {}".format(boxA_area, boxB_area))
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

def non_max_supression(bounding_boxes, probes, overlap_thresh):

  """
    Apply non maximum supression for box and prob

    Args:
      bounding_boxes : list of bounding box
      probes : probability of this bounding box
      overlap_thresh : overlap threshold. Decide to remove this bounding box or not

    Returns:
      List of bounding boxes and its probability
  """

  if len(bounding_boxes) == 0:
    return []

  org_bounding_boxes = bounding_boxes.copy()
  org_probes = probes.copy()
  bounding_boxes = bounding_boxes.astype("float32")

  # Initializing results
  bb_result = []
  prob_result = []

  idxs = np.argsort(probes)

  while len(idxs) > 0:
    last = len(idxs) - 1
    i = idxs[last]
    bb_result.append(bounding_boxes[i])
    prob_result.append(probes[i])
    vfunc_compute_iou = np.vectorize(compute_iou, excluded="boxB", signature="(m),(n)->()")
    iou = vfunc_compute_iou(bounding_boxes, bounding_boxes[i])
    # print(iou)
    overlap_idx = np.squeeze(np.argwhere(iou > overlap_thresh))
    bounding_boxes = np.delete(bounding_boxes, overlap_idx, axis = 0)
    probes = np.delete(probes, overlap_idx, axis = 0)
    idxs = np.argsort(probes)
  return np.asarray(bb_result, dtype="int32"), prob_result




if __name__ == '__main__':
  bbxes = np.array([[213 , 72 ,420 ,263],
                    [234 ,164 ,392 ,246],
                    [202 ,72 ,426 ,280],
                    [202 ,72 ,365 ,280],
                    [  0 ,51 ,253 ,355],
                    [261 ,164 ,402 ,226],
                    [115 ,111 ,507 ,392],
                    [343 ,0 ,555 ,372],
                    [115 ,106 ,483 ,376],
                    [186 ,0 ,650 ,382],
                    [115 ,110 ,483 ,374],
                    [215 ,107 ,427 ,263],
                    [115 ,121 ,483 ,374]])
  prob = np.array([9.9769777e-01,
                  9.9892360e-01,
                  9.9917525e-01,
                  9.9952221e-01,
                  9.9740750e-01,
                  9.9474722e-01,
                  9.9977583e-01,
                  9.9974769e-01,
                  9.9944311e-01,
                  9.9908102e-01,
                  9.9962127e-01,
                  9.9784970e-01,
                  9.9963045e-01])
  bb_result, prob_result = non_max_supression(bbxes, prob, 0.5)
  print(bb_result)
  print(prob_result)
  # sum_func = np.vectorize(sum, excluded="b")
  # print(sum_func(a = [2,3,4,5], b=1))

  # print(compute_iou([213,72,420,263],[115,111,507,392]))
