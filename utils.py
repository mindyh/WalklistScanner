import cv2
import json
import numpy as np
import os
import shutil
import sys
import pytesseract
from enum import Enum


__DEBUG__ = True
MAX_BARCODES_ON_PAGE = 8
# pixels, as measured from the top of one line of voter info to the top of the next one
DISTANCE_BT_VOTERS = 123  
TEMP_DIR = 'temp/'
REFPTS_FILENAME = 'ref_bounding_boxes.json'


def show_image(image):
  # show the output image
  cv2.namedWindow("Image", 0)
  cv2.resizeWindow("Image", 50, 50)
  cv2.imshow("Image", image)
  cv2.waitKey(0)


def load_ref_boxes():
  def box_dict_to_tuples(bounding_box):
    return [(bounding_box["0"][0], bounding_box["0"][1]), 
      (bounding_box["1"][0], bounding_box["1"][1])]

  boxes = {}
  with open(REFPTS_FILENAME, "r+") as f:
      boxes = json.load(f)

  for (name, box) in boxes.items():
    boxes[name] = box_dict_to_tuples(box)

  return boxes


# Region of interest, the cropped image of the text.
def get_roi(image, bounding_box):
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
  roi = get_roi(image, bounding_box)

  # in order to apply Tesseract v4 to OCR text we must supply
  # (1) a language, (2) an OEM flag of 1, indicating that the we
  # wish to use the LSTM neural net model for OCR, and finally
  # (3) an PSM value, in this case, 6 which implies that we are
  # treating the ROI as a single block of text
  # config = ("-c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' -l eng --oem 0 --psm 6")
  config = ("-l eng --oem 1 --psm %d" % segmentation_mode.value)
  text = pytesseract.image_to_string(roi, config=config)
  return text


def get_list_id(image, bounding_box):
  text = run_ocr(image, bounding_box)
  
  if "List ID:" not in text:
    print("get_list_id: could not read the list ID.")
    return None

  list_id = text.split(': ')[1]

  if __DEBUG__:
    print ("OCR: ", text)
    print ("List ID: ", list_id)

  return list_id


class ResponseCode:
  def __init__(self, bounding_box, question_number, value):
    # Coordinates are calculated as the center of the
    # bounding box around the scan code.
    self.coords = ((bounding_box[0][0] + bounding_box[1][0]) / 2.0,
      (bounding_box[0][1] + bounding_box[1][1]) / 2.0)
    # The question it belongs to.
    self.question_number = question_number
    self.value = value

  # A dict representation of the ResponseCode.
  def get_dict(self):
    return { "coords": self.coords, "question_number": self.question_number, 
      "value": self.value }


def load_response_codes():
  pass


def get_page_filename(page_number):
  return '%s/page%d.jpg' % (TEMP_DIR, page_number)


def load_page(page_number):
  image = cv2.imread(get_page_filename(page_number), 0)
  # ret, image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY_INV)
  # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
  #             cv2.THRESH_BINARY, 11, 2)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

  # TODO: deskew image
  # TODO: rectify image

  return image