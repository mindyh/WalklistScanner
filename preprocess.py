"""This file deals with all the markup and preprocessing needed 
for ingesting a new walklist

usage: python preprocess.py -w test.pdf
"""
import argparse
import copy
import cv2
import json
import numpy as np
from pdf2image import convert_from_path
import os
import shutil
import sys

import utils


def parse_args():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-w", "--walklist", required=True,
    help="path to the PDF of the scanned walklist.")
  ap.add_argument("--skip_markup", action='store_true', 
    help="For dev purposes, if you want to skip marking up the page")
  ap.add_argument("--single_markup", action='store_true', 
    help="For dev purposes, if you want to mark up a single box on the page")
  return vars(ap.parse_args())


def save_points(refPts, point_name="new_point"):
  points = utils.load_ref_boxes()

  # Add and save new ones
  points[point_name] = []
  for point in refPts:
    points[point_name].append(point)

  with open(utils.REFPTS_FILENAME, "w+") as f:
    json.dump(points, f)

  print ("Saved to %s." % utils.REFPTS_FILENAME)


def check_for_errors(args):
  # Check that the pdf passed in exists!
  try:
      fh = open(args["walklist"], 'r')
  except FileNotFoundError:
    print ("Image not found")
    sys.exit()

  # Check that the reference file exists!
  if not (os.path.isfile(utils.REFPTS_FILENAME) and os.path.getsize(utils.REFPTS_FILENAME) > 0):
    print ("Reference points not found")
    sys.exit()

  # TODO: check that the walklist passed in is a PDF

  # Make the temporary directory
  if os.path.exists(utils.TEMP_DIR):
    shutil.rmtree(utils.TEMP_DIR)
  os.mkdir(utils.TEMP_DIR)


# Returns the number of pages in the walklist
def ingest_walklist(filepath):
  # TODO: load as pdf, convert to png
  pages = convert_from_path(filepath, 300)  # 300 dpi, optimal for tesseract
  num_pages = len(pages)
  for page_number in range(num_pages):
      pages[page_number].save(utils.get_page_filename(page_number), 'JPEG')

  print("Done ingesting the PDF.")
  return num_pages


refPts = []
is_cropping = False

"""The user draws a rectangle around the response codes, returns each
response code and the global coordinates of it w.r.t. the original image."""
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
      cv2.rectangle(image, refPts[0], refPts[1], (0, 0, 255), 2)
      cv2.imshow("markup", image)

  clone = image.copy()
  cv2.namedWindow("markup", cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback("markup", click_and_drag)

  cv2.putText(image, "Click and drag to draw a box. Press enter when you are done. \
              Press 'r' to reset.", 
              (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
   
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

  image = clone.copy()
  cv2.destroyAllWindows()

  return refPts


# Returns a list of ResponseCode objects.
def markup_response_codes(image):
  response_codes = []
  # Iterate through and mark each scan code.
  question_number = 1
  # all non-alphanumeric chars
  while True:
    print ("Please mark each response code in survey question %d." % question_number)

    bounding_box = markup_page(image)
    text = utils.run_ocr(image, bounding_box, utils.SegmentationMode.SINGLE_WORD)
     # sometimes OCR picks up stray symbols, get rid of them.
    text = ''.join(ch for ch in text if ch.isalnum())
    response_code = utils.ResponseCode(bounding_box, question_number, text)

    print("Extracted scan code: \"%s\"" % response_code.value) 

    while True:
      print("Is this correct? [y|n]")
      yes_no = input().lower()
      if yes_no == "y":
        break
      else:
        print("Please enter the correct response code: ")
        response_code.value = input()
    
    response_codes.append(response_code)

    print ("Hit enter (no input) to mark another response in the same survey question. \
            Enter 'n' to move to the next survey question. Enter 'q' to finish. [enter|n|q]")
    next_step = input().lower()

    if next_step == "n":
      question_number += 1
    elif next_step == "q":
      break

  return response_codes


def save_response_codes(response_codes):
  response_code_dict = {}
  for ctr in range(len(response_codes)):
    response_code = response_codes[ctr]
    response_code_dict[ctr] = response_code.get_dict()

  with open(utils.RESPONSE_CODES_FILENAME, "w+") as f:
    json.dump(response_code_dict, f)


# Get the box around the all the respond codes.
def get_response_codes_roi(response_codes, page):
  x_min = 99999999  # TODO: switch out for INT_MAX
  x_max = 0
  y_min = 99999999
  y_max = 0

  for response_code in response_codes:
    x_min = min(x_min, response_code.bounding_box[0][0])
    y_min = min(y_min, response_code.bounding_box[0][1])
    x_max = max(x_max, response_code.bounding_box[1][0])
    y_max = max(y_max, response_code.bounding_box[1][1])

  return utils.add_padding(((x_min, y_min), (x_max, y_max)), 20,
                           page.shape[:2])


def main():
  args = parse_args()
  check_for_errors(args)
  ref_bounding_boxes = utils.load_ref_boxes()

  # load the input image
  ingest_walklist(args["walklist"])

  page_number = 1
  page = utils.load_page(page_number)
  if args["single_markup"]:
    box = markup_page(page)
    save_points(box)

  response_codes = []
  if not args["skip_markup"]:
    response_codes = markup_response_codes(page)
    save_response_codes(response_codes)
  else:
    response_codes = utils.load_response_codes()

  response_codes_roi = get_response_codes_roi(response_codes, page)
  save_points(response_codes_roi, "first_response_codes")

  cv2.rectangle(page, response_codes_roi[0], response_codes_roi[1], (0, 0, 255), 2)
  utils.show_image(page)


if __name__ == '__main__':
  main()


