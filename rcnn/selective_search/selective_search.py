import cv2
import random



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

if __name__ == '__main__':
    
    image = cv2.imread("./selective_search/dog.jpeg", 1)
    rects = selective_search(image, "fast")
    print(len(rects))

    # visuaize image
    for i in range(0, len(rects), 100):
        output = image.copy()
        for x, y, w, h in rects[i: i + 100]:
            color = [random.randint(0, 255) for j in range(3)]
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.imshow("Output", output)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
