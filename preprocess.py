"""This file deals with all the markup and preprocessing needed 
for ingesting a new walklist"""
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
  return vars(ap.parse_args())


def save_points(refPts):
  points = utils.load_ref_boxes()

  # Add and save new ones
  for ctr in range(len(refPts)):
    if "new_point" not in points:
      points["new_point"] = {}
    points["new_point"][ctr] = refPts[ctr]

  with open(utils.REFPTS_FILENAME, "w+") as f:
    json.dump(points, f)

  print (points)


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


# TODO: pull this out into its own pre-processing script.
# Returns a list of ResponseCode objects.
def markup_response_codes(image):
  response_codes = []
  # Iterate through and mark each scan code.
  question_number = 1
  # all non-alphanumeric chars
  while True:
    print ("Please mark each response code in survey question %d." % question_number)

    bounding_box = markup_page(image)
    text = run_ocr(image, bounding_box, SegmentationMode.SINGLE_WORD)
     # sometimes OCR picks up stray symbols, get rid of them.
    text = ''.join(ch for ch in text if ch.isalnum())
    response_code = ResponseCode(bounding_box, question_number, text)

    print("Extracted scan code: \"%s\"" % response_code.value) 

    while True:
      print("Is this correct? [y|n]")
      yes_no = input().lower()
      if yes_no == "y":
        break
      else:
        print("Please enter the correct response code: ")
        response_code.value = input().lower()
    
    response_codes.append(response_code)

    print ("Hit enter (no input) to mark another response in the same survey question. \
            Enter 'n' to move to the next survey question. Enter 'q' to finish. [enter|n|q]")
    next_step = input().lower()

    if next_step == "n":
      question_number += 1
    elif next_step == "q":
      break

  return response_codes


def save_response_codes(response_codes, list_id):
  response_code_dict = {}
  for response_code in response_codes:
    if response_code.question_number not in response_code_dict:
      response_code_dict[response_code.question_number] = []
    response_code_dict[response_code.question_number].append(response_code.get_dict())

  with open("%s_response_codes.json" % list_id, "w+") as f:
    json.dump(response_code_dict, f)


def main():
  args = parse_args()
  check_for_errors(args)
  ref_bounding_boxes = utils.load_ref_boxes()

  # load the input image
  ingest_walklist(args["walklist"])

  page_number = 1
  page = utils.load_page(page_number)

  list_id = utils.get_list_id(page, ref_bounding_boxes["list_id"])
  response_codes = markup_response_codes(page)
  save_response_codes(response_codes, list_id)

if __name__ == '__main__':
  main()


