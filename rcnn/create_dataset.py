import config
import cv2
import xml.etree.ElementTree as ET
import os
import glob
from utils import selective_search, compute_iou
import cv2


def get_bounding_box_from_xml(xml_path):
  """
    Reading xml file and return its bounding boxes

    Args:
      xml_path: str : XML file path
    Returns:
      bounding boxes
  """

  tree = ET.parse(xml_path)
  root = tree.getroot()
  bounding_boxes = []
  for member in root.findall('object'):
    value = [ int(member[4][0].text),
              int(member[4][1].text),
              int(member[4][2].text),
              int(member[4][3].text)
    ]
    bounding_boxes.append(value)
  return bounding_boxes



def create_racoon_dataset():
  for dirPath in (config.POSITVE_PATH, config.NEGATIVE_PATH):
    if not os.path.exists(dirPath):
      os.makedirs(dirPath)
  total_image = 0
  for file_path in glob.glob(config.ORIGINAL_IMAGES_FOLDER + "\\*.jpg"):
    print(file_path)
    img_file_name = file_path.split("\\")[-1]
    xml_file_name = str.replace(img_file_name, "jpg", "xml")
    xml_file_path = config.ANNOTATIONS_FOLDER + "\\" + xml_file_name
    # Extract ground-truth bounding box
    gt_bounding_boxes = get_bounding_box_from_xml(xml_file_path)
    image = cv2.imread(file_path, 1)

    ROI = None
    output_path = None
    for gt_box in gt_bounding_boxes:
      (gtStartX, gtStartY, gtEndX, gtEndY) = gt_box
      ROI =image[gtStartY:gtEndY, gtStartX:gtEndX]
      filename = "{}.png".format(total_image)
      output_path = os.path.sep.join([config.POSITVE_PATH, filename])
      if ROI is not None and output_path is not None:
        ROI = cv2.resize(ROI, config.INPUT_DIMS,
                  interpolation=cv2.INTER_CUBIC)
      cv2.imwrite(output_path, ROI)
      total_image = total_image + 1

def create_dataset():

  # Ensure data directory is exist
  for dirPath in (config.POSITVE_PATH, config.NEGATIVE_PATH):
    if not os.path.exists(dirPath):
      os.makedirs(dirPath)

  total_Positive = 0
  total_Negative = 0

  for file_path in glob.glob(config.ORIGINAL_IMAGES_FOLDER + "\\*.jpg"):
    print(file_path)
    img_file_name = file_path.split("\\")[-1]
    xml_file_name = str.replace(img_file_name, "jpg", "xml")
    xml_file_path = config.ANNOTATIONS_FOLDER + "\\" + xml_file_name
    # Extract ground-truth bounding box
    gt_bounding_boxes = get_bounding_box_from_xml(xml_file_path)

    image = cv2.imread(file_path, 1)
    # Selective search to find proposal region
    proposed_Rects = []
    rects = selective_search(image)
    for x, y, w, h in rects:
      proposed_Rects.append([x, y, x + w, y + h])

    # Define positive ROI and negative ROI
    positiveROIs = 0
    negativeROIs = 0
    max_proposed = min(config.MAX_PROPOSALS, len(proposed_Rects))
    for proposed_Rect in proposed_Rects[:max_proposed]:
      # Unpack proposed rect
      (propStartX, propStartY, propEndX, propEndY) = proposed_Rect
      for gt_box in gt_bounding_boxes:
        # Compute IOU between ground truth and region on proposed
        iou = compute_iou(gt_box, proposed_Rect)
        # if iou > 0.0:
        #   print("{} has IOI > 0".format(proposed_Rect))
        (gtStartX, gtStartY, gtEndX, gtEndY) = gt_box

        roi = None
        output_path = None

        # If proposed image is like ground_truth and number of positive is not exceeds max positive number
        if iou > 0.9 and positiveROIs <= config.MAX_POSITIVE:
          roi = image[propStartY:propEndY, propStartX:propEndX]
          filename = "{}.png".format(total_Positive)
          output_path = os.path.sep.join([config.POSITVE_PATH, filename])

          # increment the positive counters
          positiveROIs += 1
          total_Positive += 1

        # determine if the proposed bounding box falls *within* the ground-truth bounding box
        fullOverlap = propStartX >= gtStartX
        fullOverlap = fullOverlap and propStartY >= gtStartY
        fullOverlap = fullOverlap and propEndX <= gtEndX
        fullOverlap = fullOverlap and propEndY <= gtEndY

        if not fullOverlap and iou < 0.05 and negativeROIs <= config.MAX_NEGATIVE:
          roi = image[propStartY:propEndY, propStartX:propEndX]
          filename = "{}.png".format(total_Negative)
          output_path = os.path.sep.join([config.NEGATIVE_PATH, filename])

          # increment the positive counters
          negativeROIs += 1
          total_Negative += 1

        if roi is not None and output_path is not None:
          roi = cv2.resize(roi, config.INPUT_DIMS,
                          interpolation=cv2.INTER_CUBIC)
          cv2.imwrite(output_path, roi)
    print("Have {} in racoon".format(positiveROIs))
    print("Have {} in no_racoon".format(negativeROIs))
    print("===============Done===============")







if __name__ == '__main__':
  # print(sys.path)
  # print(config.ANNOTATIONS_FOLDER)
  # bbx = get_bounding_box_from_xml("data\\annotations\\raccoon-1.xml")
  # img = cv2.imread("data\\images\\raccoon-1.jpg", 1)
  # # print(img)
  # for bb in bbx:
  #   cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255,255,0), 1)
  # cv2.imshow("racoon", img)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  create_racoon_dataset()

