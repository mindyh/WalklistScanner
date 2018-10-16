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
  ap.add_argument("-c","--clean", default=None,
    help="path to the PDF of the clean, unmarked page to use as a reference.")
  ap.add_argument("--skip_markup", action='store_true', 
    help="For dev purposes, if you want to skip marking up the page")
  ap.add_argument("--single_markup", action='store_true', 
    help="For dev purposes, if you want to mark up a single box on the page")
  ap.add_argument("--rotate_dir", default=None, 
    help="CW or CCW, rotate the page 90 degrees in that direction.")
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
  # Check that the walklist pdf passed in exists!
  try:
      fh = open(args["walklist"], 'r')
  except FileNotFoundError:
    print ("Walklist file not found")
    sys.exit()

  # Check that the clean pdf passed in exists!
  if args["clean"]:
    try:
        fh = open(args["clean"], 'r')
    except FileNotFoundError:
      print ("Clean file not found")
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

  # Make the data directory
  if not os.path.exists(utils.DATA_DIR):
    os.mkdir(utils.DATA_DIR)


# Returns the number of pages in the walklist
def ingest_walklist(filepath, rotate_dir, list_id, is_clean_file):

  # convert PDF pages to images
  pages = convert_from_path(filepath, 300)  # 300 dpi, optimal for tesseract

  # If list_id isn't passed in, get list ID from the first page
  if not list_id:
    # save page to temp, load with opencv
    temp_filepath = '{}/page_for_id.jpg'.format(utils.TEMP_DIR)
    pages[0].save(temp_filepath, 'JPEG')
    image = utils.load_page(None, None, rotate_dir, temp_filepath)

    reference_image = utils.load_page(None, None, None, utils.REF_IMAGE_PATH)
    aligned_image, h = utils.alignImages(image, reference_image)
    list_id = utils.get_list_id(aligned_image)

    # Make the list id directory
    list_dir = '{}{}'.format(utils.DATA_DIR, list_id)
    if os.path.exists(list_dir):
      shutil.rmtree(list_dir)
    os.mkdir(list_dir)

    # Make the walklist directory
    walklist_dir = '{}/{}'.format(list_dir, utils.WALKLIST_DIR)
    if os.path.exists(walklist_dir):
      shutil.rmtree(walklist_dir)
    os.mkdir(walklist_dir)

  # if this is the clean file, only save the first page
  if is_clean_file:
    num_pages = 1
    clean_filepath = '{}{}/clean_page.jpg'.format(utils.DATA_DIR, list_id)
    pages[0].save(clean_filepath, 'JPEG')

    print("Done ingesting the clean PDF.")

  else:
    num_pages = len(pages)
    for page_number in range(num_pages):
        pages[page_number].save(utils.get_page_filename(list_id, page_number), 'JPEG')

    print("Done ingesting the walklist PDF.")
  return num_pages, list_id


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

  # ingest the walklist and clean walklist
  num_pages, list_id = ingest_walklist(args["walklist"], args["rotate_dir"], None, False)
  if args["clean"]:
    ingest_walklist(args["clean"], args["rotate_dir"], list_id, True)
  else:
    ingest_walklist(args["walklist"], args["rotate_dir"], list_id, True)

  # TODO: rewrite the rest of this function with clean_page

  page_number = 1
  page = utils.load_page(list_id, page_number, args["rotate_dir"])
  if args["single_markup"]:
    box = markup_page(page)
    save_points(box)

  response_codes = []
  if not args["skip_markup"]:
    response_codes = markup_response_codes(page)
  else:
    response_codes = utils.load_response_codes()

  response_codes_roi = get_response_codes_roi(response_codes, page)

  # Normalize the response code coords
  if not args["skip_markup"]:
    for code in response_codes:
      code.coords = (code.coords[0] - response_codes_roi[0][0], code.coords[1] - response_codes_roi[0][1])

  save_response_codes(response_codes) 
  save_points(response_codes_roi, "first_response_codes")

  response_codes_image = page[response_codes_roi[0][1]:response_codes_roi[1][1],
                              response_codes_roi[0][0]:response_codes_roi[1][0]]
  cv2.imwrite(utils.RESPONSE_CODES_IMAGE_PATH, response_codes_image)
  print ("Saved out reference response codes.")
  print ("Done, now run scan.py")

if __name__ == '__main__':
  main()


