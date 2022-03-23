import imutils
import cv2

from pathlib import Path
from tqdm import tqdm


def get_contours(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    if len(c) > 20:
        # Get every fifth elements
        c = c[0::5]

    return c

def main():
    files = Path("datasets-track4/Auto-retail-syndata-release/segmentation_labels").glob("**/*.jpg")

    for i, path in tqdm(enumerate(files)):
        img = cv2.imread(str(path))
        get_contours(img)

if __name__ == "__main__":
    main()
