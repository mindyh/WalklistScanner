import cv2
import imutils
import json
import math
import numpy as np
import os
import shutil
import sys
from typing import Union, Any, List, Optional, Tuple, Dict
import pytesseract
from enum import Enum, IntEnum
import re
import csv
import ast

__DEBUG__ = False
__DEBUG__ = True

TEMP_DIR = 'temp/'
DATA_DIR = 'data/'
WALKLIST_DIR = 'walklist/'
ERROR_PAGES_DIR = 'error_pages/'
CLEAN_IMAGE_FILENAME = 'clean_page.jpg'
RESPONSE_CODES_FILENAME = 'response_codes.json'
RESPONSE_CODES_IMAGE_FILENAME = 'response_codes.jpg'
BOLD_RESPONSE_CODES_IMAGE_FILENAME = 'bold_' + RESPONSE_CODES_IMAGE_FILENAME
REF_BB_FILENAME = 'ref_bounding_boxes.json'
COMMON_REF_FILENAME = 'ref_common.json'

WHITE_VALUE = 255
COLOR_WHITE = (WHITE_VALUE, WHITE_VALUE, WHITE_VALUE)
DPI = 300
MARGIN = 0.25 * DPI
PAGE_HEIGHT = 11 * DPI
PAGE_WIDTH = 8.5 * DPI
MAX_BARCODES_ON_PAGE = 8


class Point:
  def __init__(self, x, y):
    # The question it belongs to.
    self.x = x
    self.y = y

  @classmethod
  def from_tuple(cls, raw_points: tuple):
    return cls(raw_points[0], raw_points[1])

  def calc_distance(self, other_point):
    return math.sqrt((other_point.x - self.x)**2 + (other_point.y - self.y)**2)  

  def to_list(self) -> List:
    return [self.x, self.y]

  def to_tuple(self) -> Tuple:
    return (self.x, self.y)


class Size:
  def __init__(self, w, h):
    # The question it belongs to.
    self.w = w
    self.h = h

  @classmethod
  def from_nparr(cls, shape: tuple):
    return cls(shape[1], shape[0])


class BoundingBox:
  def __init__(self, top_left, bottom_right):
    # The question it belongs to.
    self.raw_bounding_box = [top_left, bottom_right]  # List of 2 tuples
    self.top_left = top_left  # Point class
    self.bottom_right = bottom_right  # Point class
    self.height = self.bottom_right.y - self.top_left.y 
    self.width = self.bottom_right.x - self.top_left.x

  """Loading from a list of tuples."""
  @classmethod
  def from_raw(cls, raw_bounding_box: list):
    return cls(Point.from_tuple(raw_bounding_box[0]), Point.from_tuple(raw_bounding_box[1]))
  
  @classmethod
  def from_points(cls, raw_bounding_box: list):
    return cls(raw_bounding_box[0], raw_bounding_box[1])

  """ Only works from a smaller image to a larger image!
      TODO: support going from a large image to a small image.
      original_origin is what the origin of the BoundingBox is in the coordinate
      system you want to transform to.
  """
  def update_coordinate_system(self, original_origin: Point) -> "BoundingBox":
    self.top_left.x = self.top_left.x + original_origin.x
    self.top_left.y = self.top_left.y + original_origin.y
    self.bottom_right.x = self.bottom_right.x + original_origin.x
    self.bottom_right.y = self.bottom_right.y + original_origin.y
    return self


  def add_padding(self, padding: int, page_size: Size) -> "BoundingBox":
    self.top_left.x = max(0, self.top_left.x - padding)
    self.top_left.y = max(0, self.top_left.y - padding)
    self.bottom_right.x = min(self.bottom_right.x + padding, page_size.w)
    self.bottom_right.y = min(self.bottom_right.y + padding, page_size.h)
    return self


  def to_list(self) -> List:
    return [self.top_left.to_list(), self.bottom_right.to_list()]
             

class ResponseCode:
  def __init__(self, bounding_box: BoundingBox, question_number: int, value: str, coords: Point=None):
    # Coordinates are calculated as the center of the
    # bounding box around the scan code.
    if not coords:
      self.coords = Point(int((bounding_box.top_left.x + bounding_box.bottom_right.x) / 2.0),
                          int((bounding_box.top_left.y + bounding_box.bottom_right.y) / 2.0))
    else:
      self.coords = coords
    # The question it belongs to.
    self.question_number = question_number
    self.value = value
    self.bounding_box = bounding_box


  @classmethod
  def from_raw_bb(cls, raw_bounding_box: list, question_number: int, value: str, coords: Point=None):
    return cls(BoundingBox.from_raw(raw_bounding_box), question_number, value, coords)


  # A dict representation of the ResponseCode.
  def get_dict(self) -> Dict:
    return { "coords": self.coords.to_tuple(), "question_number": self.question_number, 
      "value": self.value, "bounding_box": self.bounding_box.to_list() }


class Rotation(IntEnum):
  CW = 90
  CCW = 270
  NONE = 0


refPts: List[tuple] = []
is_cropping = False

class Image:
  def __init__(self, raw_image: np.array):
    self.raw_image = raw_image
    self.size: Size = Size.from_nparr(raw_image.shape[:2])


  @classmethod
  def from_file(cls, image_filepath: str, rotate_dir: Rotation):
    image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
    image = imutils.rotate_bound(image, rotate_dir)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cls(image)


  def save_to_file(self, filepath: str) -> None:
    cv2.imwrite(filepath, self.raw_image)


  def show(self, message: str= "", title: str="", waitKey: bool=True, resize: bool=True) -> str:
    raw_image = self.raw_image
    cv2.namedWindow(title, 0)
    if resize:
      cv2.resizeWindow(title, (100, 100))
    cv2.putText(raw_image, message, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow(title, raw_image)
    return cv2.waitKey(0) & 0xFF


  def crop(self, top: int=0, bottom: int=0, left: int=0, right: int=0) -> "Image":
    raw_image = self.raw_image
    raw_image = raw_image[top : self.size.h - bottom, left : self.size.w - right]
    return Image(raw_image)


  def threshold(self, threshold: int=100) -> "Image":
    # Get the Otsu threshold
    # blur = cv2.GaussianBlur(image,(5,5),0)
    # find otsu's threshold value with OpenCV function
    image = self.raw_image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 0 , 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return Image(image)


  def numWhitePixels(self) -> int:
    return cv2.countNonZero(self.threshold().grayscale().raw_image)


  def grayscale(self) -> "Image":
    return Image(cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY))


  def invert(self) -> "Image":
    return Image(cv2.bitwise_not(self.raw_image))


  def diff_against(self, ref_image: "Image") -> "Image":
    same_size_image = self.make_same_size_as(ref_image)
    diff = cv2.bitwise_xor(same_size_image.raw_image, ref_image.raw_image)
    return Image(diff)


  def add_border(self, top: int=0, bottom: int=0, left: int=0, right: int=0) -> "Image":
    temp_image = self.raw_image
    temp_image = cv2.copyMakeBorder(temp_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=COLOR_WHITE)
    return Image(temp_image)


  """Pad and crop im_to_change until they're the same size as ref_image."""
  """TODO: refactor to use add_border() and crop()"""
  def make_same_size_as(self: "Image", ref_image: "Image") -> "Image":
    temp_image = self.raw_image

    # Make the height the correct size.
    height_diff = abs(self.size.h - ref_image.size.h)
    if (ref_image.size.h > self.size.h):

      temp_image = cv2.copyMakeBorder(temp_image, height_diff, 0, 0, 0, cv2.BORDER_CONSTANT, value=COLOR_WHITE)
    elif (height_diff != 0):
      temp_image = temp_image[: -height_diff, :]

    # Make the width the correct size.
    width_diff = abs(self.size.w - ref_image.size.w)
    if (ref_image.size.w > self.size.w):
      temp_image = cv2.copyMakeBorder(temp_image, 0, 0, 0, width_diff, cv2.BORDER_CONSTANT, value=COLOR_WHITE)
    elif width_diff != 0:
      temp_image = temp_image[:, : -width_diff]
    
    return Image(temp_image)

  # from https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
  def align_to(self, ref_image: 'Image') -> 'Image':
    raw_im_to_be_aligned = self.make_same_size_as(ref_image).raw_image

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15
   
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(raw_im_to_be_aligned, None)
    keypoints2, descriptors2 = orb.detectAndCompute(ref_image.raw_image, None)
     
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
      imMatches = cv2.drawMatches(raw_im_to_be_aligned, keypoints1, 
                                  ref_image.raw_image, keypoints2, matches, None)
      cv2.imwrite("matches.jpg", imMatches)
     
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
   
    for i, match in enumerate(matches):
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt

    # height, width = ref_image.shape[:2]
    # transform = cv2.estimateRigidTransform(im_to_be_aligned, ref_image, fullAffine=False)
    # # transform = cv2.estimateRigidTransform(points1, points2, fullAffine=False)
    # aligned_image = cv2.warpAffine(im_to_be_aligned, transform, (width, height), flags=cv2.INTER_LINEAR,
    #   borderMode=cv2.BORDER_CONSTANT, borderValue=COLOR_WHITE)
     
    # Find homography
    transform, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
   
    # Use homography
    aligned_image = cv2.warpPerspective(raw_im_to_be_aligned, transform, (self.size.w, self.size.h), 
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=COLOR_WHITE)
     
    return Image(aligned_image)


  def insert_image(self, other_image: "Image", insert_point: Point) -> "Image":
    # Crop the other image to fit if needed.
    available_width = self.size.w - insert_point.x
    if other_image.size.w > available_width:
      other_image = other_image.crop(left=other_image.size.w - available_width)

    available_height = self.size.h - insert_point.y
    if other_image.size.h > available_height:
      other_image = other_image.crop(bottom=other_image.size.h - available_height)

    end_point = Point(insert_point.x + other_image.size.w, insert_point.y + other_image.size.h)

    raw_image = self.raw_image
    raw_image[insert_point.y : end_point.y, insert_point.x : end_point.x] = other_image.raw_image
    return Image(raw_image)


  # Region of interest, the cropped image of the text.
  def get_roi(self, bounding_box: BoundingBox, padding=0.0) -> "Image":
    # dummy until I figure out what this is
    rW = 1
    rH = 1
    (origH, origW) = self.size.h, self.size.w

    startX = bounding_box.top_left.x
    startY = bounding_box.top_left.y
    endX = bounding_box.bottom_right.x
    endY = bounding_box.bottom_right.y

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
    roi = Image(self.raw_image[startY:endY, startX:endX])
    return roi


  class SegmentationMode(Enum):
    SINGLE_WORD = 8
    BLOCK_OF_TEXT = 6


  def find_text(self, bounding_box: BoundingBox=None, segmentation_mode=SegmentationMode.SINGLE_WORD) -> str:
    image = self
    if bounding_box:
      image = self.get_roi(bounding_box)

    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 1, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an PSM value, in this case, 6 which implies that we are
    # treating the ROI as a single block of text
    # config = ("-c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' -l eng --oem 0 --psm 6")
    config = ("-l eng --oem 1 --psm %d" % segmentation_mode.value)
    text = pytesseract.image_to_string(image.raw_image, config=config)
    return text

  def markup(self) -> BoundingBox:
    def click_and_drag(event, x, y, flags, param) -> None:   
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

    clone = self.raw_image.copy()
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
        clone = self.raw_image.copy()
     
      # if the enter key is pressed, break from the loop
      elif key == ord("\r"):
        break

    # image = clone.copy()
    cv2.destroyAllWindows()

    return BoundingBox.from_raw(refPts)


class Page(Image):
  # pixels, as measured from the top of one line of voter info to the top of the next one, 300DPI
  DISTANCE_BT_VOTERS = 220


  def __init__(self, raw_page: np.array, page_number: Optional[int]=None):
    Image.__init__(self, raw_page)
    self.page_number = page_number


  def get_line_bb(self, line_number: int, list_dir: str, padding=0, right_half=False) -> BoundingBox:
    ref_bounding_boxes = load_ref_boxes(list_dir)
    y1 = ref_bounding_boxes["first_barcode"].top_left.y + (self.DISTANCE_BT_VOTERS * (line_number - 1)) - padding
    line_bb =  BoundingBox(Point(0, y1),
                           Point(self.size.w, min(self.size.h, (y1 + self.DISTANCE_BT_VOTERS + (2 * padding)))))
    if right_half:
      # Chop it to the right half, for markup.
      line_bb.top_left.x = int(line_bb.bottom_right.x / 2)

    return line_bb


  # TODO: move load bounding box logic into the function
  def get_list_id(self, bounding_box: BoundingBox) -> Optional[str]:
    return get_list_id(self.get_roi(bounding_box))


def get_list_id(image: Image) -> Optional[str]:
  text = image.find_text()

  id_regex_match = re.search(r'\d{6}', text)
  if id_regex_match:
    list_id = id_regex_match.group(0)
      
    # Error check on the list_id.
    if len(list_id) != 6:
      print("get_list_id: could not read the list ID.")
      return None
  else:
    return None

  return list_id


def make_blank_page() -> Page:
  raw_image = np.ones((PAGE_HEIGHT, PAGE_WIDTH,3), np.uint8) * WHITE_VALUE
  return Page(raw_image)


def is_non_zero_file(fpath) -> bool:
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def load_raw_ref_boxes(list_dir: str) -> dict:
  boxes: dict = {}
  filepath = list_dir + REF_BB_FILENAME

  if is_non_zero_file(filepath):
    with open(filepath, "r+") as f:
      boxes = json.load(f)  
  return boxes


def save_ref_boxes(list_dir: str, dict_to_add: dict) -> None:
  filepath = list_dir + REF_BB_FILENAME
  boxes = load_raw_ref_boxes(list_dir)
  with open(filepath, "w+") as f:
    boxes.update(dict_to_add)
    json.dump(boxes, f)


def load_ref_boxes(list_dir: str) -> Dict[str, BoundingBox]:
  boxes = load_raw_ref_boxes(list_dir)
  for key, box in boxes.items():
    boxes[key] = BoundingBox.from_raw(box)
  return boxes


def load_response_codes(list_id: str) -> List[ResponseCode]:
  response_codes = []
  response_codes_filepath = "{}{}/{}".format(DATA_DIR, list_id, RESPONSE_CODES_FILENAME)
  with open(response_codes_filepath, "r+") as f:
    response_codes_dict = json.load(f)
    for (_, response_dict) in response_codes_dict.items():
      response_codes.append(ResponseCode.from_raw_bb(response_dict["bounding_box"],
                                                     response_dict["question_number"],
                                                     response_dict["value"],
                                                     Point.from_tuple(response_dict["coords"])))

  return response_codes


def get_page_filename(list_id: str, page_number: int) -> str:
  return '%s%s/%spage%d.jpg' % (DATA_DIR, list_id, WALKLIST_DIR, page_number)


def get_list_dir(list_id: str) -> str:
  return "{}{}/".format(DATA_DIR, list_id)


def extractCSVtoDict(filepath: str) -> dict:
  scans = {}
  with open(filepath, 'r') as file:
    for row in csv.DictReader(file):
      questions = {key:value for key, value in row.items() if key is not 'voter_id'}
      scans[row['voter_id']] = {'voter_id': row['voter_id'], 'questions': questions}

  return scans


def convertStringListToList(input_list: List) -> List:
  if isinstance(input_list, str):
    if input_list:
      return ast.literal_eval(input_list)
    else:
      return []
  else:
    return input_list


def map_rotation(rotate_str: str) -> Rotation:
  if rotate_str == "CW":
    return Rotation.CW
  elif rotate_str == "CW":
    return Rotation.CCW
  else:
    return Rotation.NONE