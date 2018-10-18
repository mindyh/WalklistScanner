import cv2
import imutils
import json
import numpy as np
import os
import shutil
import sys
import pytesseract
from enum import Enum
import re

__DEBUG__ = True
MAX_BARCODES_ON_PAGE = 8
# pixels, as measured from the top of one line of voter info to the top of the next one
DISTANCE_BT_VOTERS = 123  
TEMP_DIR = 'temp/'
DATA_DIR = 'data/'
WALKLIST_DIR = 'walklist/'
ERROR_PAGES_DIR = 'error_pages/'
CLEAN_IMAGE_FILENAME = 'clean_page.jpg'
RESPONSE_CODES_FILENAME = 'response_codes.json'
RESPONSE_CODES_IMAGE_FILENAME = 'response_codes.jpg'
REFPTS_FILENAME = 'ref_bounding_boxes.json'

def add_padding(bounding_box, padding, page_size):
  return ((max(0, bounding_box[0][0] - padding),
           max(0, bounding_box[0][1] - padding)), 
          (min(bounding_box[1][0] + padding, page_size[1]),
           min(bounding_box[1][1] + padding, page_size[0])))


def show_image(image):
  # show the output image
  cv2.namedWindow("Image", 0)
  cv2.resizeWindow("Image", 100, 100)
  cv2.imshow("Image", image)
  cv2.waitKey(0)


def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def load_ref_boxes(list_id):
  boxes = {}
  filepath = '{}{}/{}'.format(DATA_DIR, list_id, REFPTS_FILENAME)

  if is_non_zero_file(filepath):
    with open(filepath, "r+") as f:
      boxes = json.load(f)

  return boxes


def save_ref_boxes(list_id, dict_to_add):
  filepath = '{}{}/{}'.format(DATA_DIR, list_id, REFPTS_FILENAME)
  boxes = load_ref_boxes(list_id)
  with open(filepath, "w+") as f:
    boxes.update(dict_to_add)
    json.dump(boxes, f)


# Region of interest, the cropped image of the text.
def get_roi(image, bounding_box, padding=0.0):
  # dummy until I figure out what this is
  rW = 1
  rH = 1
  (origH, origW) = image.shape[:2]

  startX, startY = bounding_box[0]
  endX, endY = bounding_box[1]

  # scale the bounding box coordinates based on the respective
  # ratios
  startX = int(startX * rW)
  startY = int(startY * rH)
  endX = int(endX * rW)
  endY = int(endY * rH)
 
  # in order to obtain a better OCR of the text we can potentially
  # apply a bit of padding surrounding the bounding box -- here we
  # are computing the deltas in both the x and y directions
  dX = int((endX - startX) * padding)
  dY = int((endY - startY) * padding)
 
  # apply padding to each side of the bounding box, respectively
  startX = max(0, startX - dX)
  startY = max(0, startY - dY)
  endX = min(origW, endX + (dX * 2))
  endY = min(origH, endY + (dY * 2))

  # extract the actual padded ROI
  roi = image[startY:endY, startX:endX]
  return roi


class SegmentationMode(Enum):
  SINGLE_WORD = 8
  BLOCK_OF_TEXT = 6


def run_ocr(image, bounding_box=None, segmentation_mode=SegmentationMode.SINGLE_WORD):
  if bounding_box:
    image = get_roi(image, bounding_box)

  # in order to apply Tesseract v4 to OCR text we must supply
  # (1) a language, (2) an OEM flag of 1, indicating that the we
  # wish to use the LSTM neural net model for OCR, and finally
  # (3) an PSM value, in this case, 6 which implies that we are
  # treating the ROI as a single block of text
  # config = ("-c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' -l eng --oem 0 --psm 6")
  config = ("-l eng --oem 1 --psm %d" % segmentation_mode.value)
  text = pytesseract.image_to_string(image, config=config)
  return text


def get_list_id_from_page(page, bounding_box):
  list_id = get_list_id(get_roi(page, bounding_box))
  return list_id


def get_list_id(image):
  text = run_ocr(image)
  
  list_id = text.split(': ')[1]
  # strip out any non-numeric characters
  list_id = re.sub("[^0-9]", "", list_id)

  # Error check on the list_id.
  print (list_id)
  if len(list_id) != 6:
    print("get_list_id: could not read the list ID.")
    return None

  if __DEBUG__:
    print ("OCR: ", text)
    print ("List ID: '{}'".format(list_id))

  return list_id


refPts = []
is_cropping = False
def markup_page(image):
  def click_and_drag(event, x, y, flags, param):
    # grab references to the global variables
    global refPts, is_cropping
   
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
      refPts = [(x, y)]
      cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:

      # record the ending (x, y) coordinates and indicate that
      # the cropping operation is finished
      refPts.append((x, y))
      cropping = False
   
      # TODO: update rectagle as you are drawing
      # draw a rectangle around the region of interest
      cv2.rectangle(clone, refPts[0], refPts[1], (0, 0, 255), 2)
      cv2.imshow("markup", clone)

  clone = image.copy()
  cv2.namedWindow("markup", cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback("markup", click_and_drag)

  cv2.putText(clone, "Click and drag to draw a box. Press enter when you are done. \
              Press 'r' to reset.", 
              (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
   
  # keep looping until the enter key is pressed
  # TODO: add message telling the user to hit enter to finish
  while True:
    # display the image and wait for a keypress
    cv2.imshow("markup", clone)
    key = cv2.waitKey(0) & 0xFF
   
    # if the 'r' key is pressed, reset to the beginning
    if key == ord("r"):
      clone = image.copy()
   
    # if the enter key is pressed, break from the loop
    elif key == ord("\r"):
      break

  # image = clone.copy()
  cv2.destroyAllWindows()

  return refPts


class ResponseCode:
  def __init__(self, bounding_box, question_number, value, coords=None):
    # Coordinates are calculated as the center of the
    # bounding box around the scan code.
    if not coords:
      self.coords = (int((bounding_box[0][0] + bounding_box[1][0]) / 2.0),
        int((bounding_box[0][1] + bounding_box[1][1]) / 2.0))
    else:
      self.coords = coords
    # The question it belongs to.
    self.bounding_box = bounding_box
    self.question_number = question_number
    self.value = value

  # A dict representation of the ResponseCode.
  def get_dict(self):
    return { "coords": self.coords, "question_number": self.question_number, 
      "value": self.value, "bounding_box": self.bounding_box }


def load_response_codes(list_id):
  response_codes = []
  response_codes_filepath = "{}{}/{}".format(DATA_DIR, list_id, RESPONSE_CODES_FILENAME)
  with open(response_codes_filepath, "r+") as f:
    response_codes_dict = json.load(f)
    for (_, response_dict) in response_codes_dict.items():
      response_codes.append(ResponseCode(response_dict["bounding_box"],
                                         response_dict["question_number"],
                                         response_dict["value"],
                                         response_dict["coords"]))

  return response_codes


def get_page_filename(list_id, page_number):
  return '%s%s/%spage%d.jpg' % (DATA_DIR, list_id, WALKLIST_DIR, page_number)


def get_list_dir(list_id):
  return "{}{}/".format(DATA_DIR, list_id)


def threshold(image, threshold=100, invert=False):
  # Get the Otsu threshold
  # blur = cv2.GaussianBlur(image,(5,5),0)
  # find otsu's threshold value with OpenCV function
  _, image = cv2.threshold(image, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  if invert:
    image = cv2.bitwise_not(image)  # invert the image
  return image

def load_page(image_filepath, rotate_dir=None):
  image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)

  if rotate_dir == "CW":
    image = imutils.rotate_bound(image, 90)
  elif rotate_dir == "CCW":
    image = imutils.rotate_bound(image, 270)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  return image


# from https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
def alignImages(im_to_be_aligned, ref_image):
  MAX_FEATURES = 500
  GOOD_MATCH_PERCENT = 0.15
 
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im_to_be_aligned, None)
  keypoints2, descriptors2 = orb.detectAndCompute(ref_image, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  if __DEBUG__:
    imMatches = cv2.drawMatches(im_to_be_aligned, keypoints1, ref_image, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt


  height, width = ref_image.shape[:2]
  # transform = cv2.estimateRigidTransform(points1, points2, False)
  # aligned_image = cv2.warpAffine(im_to_be_aligned, transform, (width, height),
  #    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
   
  # Find homography
  transform, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  aligned_image = cv2.warpPerspective(im_to_be_aligned, transform, (width, height), 
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))
   
  return aligned_image, transform
