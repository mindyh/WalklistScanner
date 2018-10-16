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
RESPONSE_CODES_FILENAME = 'response_codes.json'
RESPONSE_CODES_IMAGE_PATH = TEMP_DIR + 'response_codes.png'
REFPTS_FILENAME = 'ref_bounding_boxes.json'

def add_padding(bounding_box, padding, page_size):
  return ((max(0, bounding_box[0][0] - padding),
           max(0, bounding_box[0][1] - padding)), 
          (min(bounding_box[1][0] + padding, page_size[1]),
           min(bounding_box[1][1] + padding, page_size[0])))


def show_image(image):
  # show the output image
  cv2.namedWindow("Image", 0)
  # cv2.resizeWindow("Image", 50, 50)
  cv2.imshow("Image", image)
  cv2.waitKey(0)


def load_ref_boxes():
  boxes = {}
  with open(REFPTS_FILENAME, "r+") as f:
    boxes = json.load(f)
  return boxes


# Region of interest, the cropped image of the text.
def get_bounding_box(image, bounding_box):
  # padding = 0.05
  padding = 0.0

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


def run_ocr(image, bounding_box, segmentation_mode=SegmentationMode.SINGLE_WORD):
  roi = get_bounding_box(image, bounding_box)

  # in order to apply Tesseract v4 to OCR text we must supply
  # (1) a language, (2) an OEM flag of 1, indicating that the we
  # wish to use the LSTM neural net model for OCR, and finally
  # (3) an PSM value, in this case, 6 which implies that we are
  # treating the ROI as a single block of text
  # config = ("-c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' -l eng --oem 0 --psm 6")
  config = ("-l eng --oem 1 --psm %d" % segmentation_mode.value)
  text = pytesseract.image_to_string(roi, config=config)
  return text


def get_list_id(image):

  # get bounding box coordinates
  ref_bounding_boxes = load_ref_boxes()

  text = run_ocr(image, ref_bounding_boxes['list_id'])
  
  if "List ID:" not in text:
    print("get_list_id: could not read the list ID.")
    return None

  list_id = text.split(': ')[1]

  # strip out any non-numeric characters
  list_id = re.sub("[^0-9]", "", list_id)

  if __DEBUG__:
    print ("OCR: ", text)
    print ("List ID: '{}'".format(list_id))

  return list_id


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


def load_response_codes():
  response_codes = []
  with open(RESPONSE_CODES_FILENAME, "r+") as f:
    response_codes_dict = json.load(f)
    for (_, response_dict) in response_codes_dict.items():
      response_codes.append(ResponseCode(response_dict["bounding_box"],
                                         response_dict["question_number"],
                                         response_dict["value"],
                                         response_dict["coords"]))

  return response_codes


def get_page_filename(list_id, page_number):
  return '%s%s/%spage%d.jpg' % (DATA_DIR, list_id, WALKLIST_DIR, page_number)


def get_roi(bounding_box, image):
  return image[bounding_box[0][1]:bounding_box[1][1],
               bounding_box[0][0]:bounding_box[1][0]] 

def threshold(image, threshold=100, invert=False):
  # Get the Otsu threshold
  # blur = cv2.GaussianBlur(image,(5,5),0)
  # find otsu's threshold value with OpenCV function
  _, image = cv2.threshold(image, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  show_image(image)
  if invert:
    image = cv2.bitwise_not(image)  # invert the image
  return image

def load_page(list_id, page_number, rotate_dir, image_filepath=None):
  if not image_filepath:
    image_filepath = get_page_filename(list_id, page_number)

  image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)

  if rotate_dir == "CW":
    image = imutils.rotate_bound(image, 90)
  elif rotate_dir == "CCW":
    image = imutils.rotate_bound(image, 270)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

  return image