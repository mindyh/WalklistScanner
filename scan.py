'''usage: python scan.py -i test.png'''

# import the necessary packages
import argparse
import cv2
import json
import numpy as np
from pdf2image import convert_from_path
from pprint import pprint
from pyzbar import pyzbar
import os
import shutil
import sys

DEBUG = True
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
  cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
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

  print points


def extract_scan_codes(x, y):
  pass


def get_page_filename(page_number):
  return '%s/page%d.jpg' % (TEMP_DIR, page_number)


# Returns the number of pages in the walklist
def ingest_walklist(filepath):
  # TODO: load as pdf, convert to png
  pages = convert_from_path(filepath)
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
    print "Image not found"
    sys.exit()

  # Check that the reference file exists!
  if not (os.path.isfile(REFPTS_FILENAME) and os.path.getsize(REFPTS_FILENAME) > 0):
    print "Reference points not found"
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
  if DEBUG:
    # extract the the barcode
    (x, y, w, h) = barcode.rect
    print "rect", barcode.rect
    print "poly", barcode.polygon

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
      cv2.imshow("image", image)

  # TODO: Ask the user to draw a rectangle around the response codes
 
  clone = image.copy()
  cv2.namedWindow("image")
  cv2.setMouseCallback("image", click_and_drag)
   
  # keep looping until the enter key is pressed
  # TODO: add message telling the user to hit enter to finish
  while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
   
    # if the 'r' key is pressed, reset to the beginning
    if key == ord("r"):
      image = clone.copy()
   
    # if the enter key is pressed, break from the loop
    elif key == ord("\r"):
      break

  save_points(refPts)

  return refPts
   

def check_list_id(image, bounding_box):
  # Crop the image
  print bounding_box
  crop_img = image[bounding_box["0"][1]:bounding_box["1"][1], 
                   bounding_box["0"][0]:bounding_box["1"][0]]
  show_image(crop_img)

  # run OCR 


def main():
  args = parse_args()
  check_for_errors(args)
  ref_bounding_boxes = load_ref_boxes()

  # load the input image
  num_pages = ingest_walklist(args["walklist"])

  for page_number in range(num_pages):
    page = process_page(page_number)
    # refPts = markup_clean_sheet(page)

    check_list_id(page, ref_bounding_boxes["list_id"])
    extract_scan_codes(page, refPts)

    # find the barcodes in the image and decode each of the barcodes
    # barcodes = pyzbar.decode(page)

    # loop over the detected barcodes
    # for barcode in barcodes:
    #   (barcode_coordinates, voter_id) = extract_barcode_info(barcode, page)

    # show the output image
    show_image(page)


if __name__ == '__main__':
  main()


