'''usage: python scan.py -w test.pdf'''

# import the necessary packages
import argparse
import copy
import cv2
import json
import numpy as np
from pdf2image import convert_from_path
from pprint import pprint
from pyzbar import pyzbar
import os
import shutil
import sys
from imutils.object_detection import non_max_suppression
import pytesseract


__DEBUG__ = True
REFPTS_FILENAME = 'ref_bounding_boxes.json'
MAX_BARCODES_ON_PAGE = 8
# pixels, as measured from the top of one line of voter info to the top of the next one
DISTANCE_BT_VOTERS = 123   
TEMP_DIR = 'temp/'


def parse_args():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-w", "--walklist", required=True,
    help="path to pdf of walklist")
  return vars(ap.parse_args())

def show_image(image):
  # show the output image
  cv2.namedWindow("Image", 0)
  cv2.resizeWindow("Image", 50, 50)
  cv2.imshow("Image", image)
  cv2.waitKey(0)


def load_ref_boxes():
  points = {}
  with open(REFPTS_FILENAME, "r+") as f:
      points = json.load(f)

  return points


def save_points(refPts):
  points = load_ref_boxes()

  # Add and save new ones
  for ctr in range(len(refPts)):
    if "new_point" not in points:
      points["new_point"] = {}
    points["new_point"][ctr] = refPts[ctr]

  with open(REFPTS_FILENAME, "w+") as f:
    json.dump(points, f)

  print (points)



def get_page_filename(page_number):
  return '%s/page%d.jpg' % (TEMP_DIR, page_number)


# Returns the number of pages in the walklist
def ingest_walklist(filepath):
  # TODO: load as pdf, convert to png
  pages = convert_from_path(filepath, 300)  # 300 dpi, optimal for tesseract
  num_pages = len(pages)
  for page_number in range(num_pages):
      pages[page_number].save(get_page_filename(page_number), 'JPEG')

  return num_pages


def process_page(page_number):
  image = cv2.imread(get_page_filename(page_number), 0)
  # ret, image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY_INV)
  # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
  #             cv2.THRESH_BINARY, 11, 2)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

  # TODO: deskew image
  # TODO: rectify image

  return image


def check_for_errors(args):
  # Check that the pdf passed in exists!
  try:
      fh = open(args["walklist"], 'r')
  except FileNotFoundError:
    print ("Image not found")
    sys.exit()

  # Check that the reference file exists!
  if not (os.path.isfile(REFPTS_FILENAME) and os.path.getsize(REFPTS_FILENAME) > 0):
    print ("Reference points not found")
    sys.exit()

  # TODO: check that the walklist passed in is a PDF

  # Make the temporary directory
  if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
  os.mkdir(TEMP_DIR)


""" Returns (barcode_coordinates, voter_id)"""
def extract_barcode_info(barcode, image):
  voter_id = barcode.data.decode("utf-8")
  barcode_coordinates = barcode.polygon

  # draw image
  if __DEBUG__:
    # extract the the barcode
    (x, y, w, h) = barcode.rect
    print ("rect", barcode.rect)
    print ("poly", barcode.polygon)

    pts = np.array([[[x, y] for (x, y) in barcode.polygon]], np.int32)
    cv2.polylines(image, pts, True, (0, 0, 255), 2)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # the barcode data is a bytes object so if we want to draw it on
    # our output image we need to convert it to a string first

    # draw the barcode data and barcode type on the image
    text = "{}".format(voter_id)
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
      0.5, (0, 0, 255), 2)

    # print the barcode type and data to the terminal
    print("[INFO] Found barcode: {}".format(voter_id))

  return barcode_coordinates, voter_id


refPts = []
is_cropping = False

"""The user draws a rectangle around the response codes, returns each
response code and the global coordinates of it w.r.t. the original image."""
def markup_clean_sheet(image):
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
      cv2.rectangle(image, refPts[0], refPts[1], (0, 0, 255), 2)
      cv2.imshow("markup", image)

  # TODO: Ask the user to draw a rectangle around the response codes

  clone = image.copy()
  cv2.namedWindow("markup", cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback("markup", click_and_drag)

  print ("waiting for user to markup page.")
   
  # keep looping until the enter key is pressed
  # TODO: add message telling the user to hit enter to finish
  while True:
    # display the image and wait for a keypress
    cv2.imshow("markup", image)
    key = cv2.waitKey(1) & 0xFF
   
    # if the 'r' key is pressed, reset to the beginning
    if key == ord("r"):
      image = clone.copy()
   
    # if the enter key is pressed, break from the loop
    elif key == ord("\r"):
      break

  save_points(refPts)

  return refPts


# Region of interest, the cropped image of the text.
def get_roi(image, bounding_box):
  # padding = 0.05
  padding = 0.0

  # dummy until I figure out what this is
  rW = 1
  rH = 1
  (origH, origW) = image.shape[:2]

  startX, startY = bounding_box['0']
  endX, endY = bounding_box['1']

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


def run_ocr(image, bounding_box):
  roi = get_roi(image, bounding_box)

  # in order to apply Tesseract v4 to OCR text we must supply
  # (1) a language, (2) an OEM flag of 1, indicating that the we
  # wish to use the LSTM neural net model for OCR, and finally
  # (3) an PSM value, in this case, 6 which implies that we are
  # treating the ROI as a single block of text
  # config = ("-c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' -l eng --oem 0 --psm 6")
  config = ("-l eng --oem 1 --psm 6")
  text = pytesseract.image_to_string(roi, config=config)
  return text


def get_list_id(image, bounding_box):
  text = run_ocr(image, bounding_box)
  print (text)
  
  if "List ID:" not in text:
    print("get_list_id: could not read the list ID.")
    return None

  list_id = text.split(': ')[1]

  if __DEBUG__:
    print ("OCR: ", text)
    print ("List ID: ", list_id)

  return list_id


def get_scan_codes(image, bounding_box):
  text = run_ocr(image, bounding_box)
  scan_codes = text.split(': ')[1].split()

  if __DEBUG__:
    print ("extracted scan codes: ", scan_codes)

  return scan_codes


def get_voter_id(image, bounding_box):
  text = run_ocr(image, bounding_box)
  print(text)
  voter_id = text.split(' ')[0]

  if __DEBUG__:
    print ("voter_id: ", voter_id)

  return voter_id


def dict_to_tuples(bounding_box):
  return [(bounding_box["0"][0], bounding_box["0"][1]), 
    (bounding_box["1"][0], bounding_box["1"][1])]


def main():
  args = parse_args()
  check_for_errors(args)
  ref_bounding_boxes = load_ref_boxes()

  # load the input image
  num_pages = ingest_walklist(args["walklist"])

  for page_number in range(num_pages):
    page = process_page(page_number)
    # refPts = markup_clean_sheet(page)

    get_list_id(page, ref_bounding_boxes["list_id"])
    get_scan_codes(page, ref_bounding_boxes["first_response_codes"])
    # get_voter_id(page, ref_bounding_boxes["first_voter_id"])

    # find the barcodes in the image and decode each of the barcodes
    barcodes = pyzbar.decode(page)

    # loop over the detected barcodes
    for barcode in barcodes:
      (barcode_coordinates, voter_id) = extract_barcode_info(barcode, page)

    show_image(page)

    # show the output image
    # if __DEBUG__:
    # box = dict_to_tuples(ref_bounding_boxes["first_voter_id"])
    # # print (ref_bounding_boxes["first_response_codes"])
    # cv2.rectangle(page, box[0], box[1], (0, 0, 255), 1)


if __name__ == '__main__':
  main()


