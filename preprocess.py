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
  ap.add_argument("-c","--clean", required=True,
    help="path to the PDF of the clean, unmarked page to use as a reference.")
  ap.add_argument("-n","--clean_page_number", type=int, default=0,
    help="The page number of the clean page to markup as part of the ingestion.")
  ap.add_argument("--skip_markup", action='store_true', 
    help="For dev purposes, if you want to skip marking up the page")
  ap.add_argument("--rotate_dir", default=None, 
    help="CW or CCW, rotate the page 90 degrees in that direction.")
  return vars(ap.parse_args())


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

  # TODO: check that the walklist passed in is a PDF

  # Make the temporary directory
  if os.path.exists(utils.TEMP_DIR):
    shutil.rmtree(utils.TEMP_DIR)
  os.mkdir(utils.TEMP_DIR)

  # Make the data directory
  if not os.path.exists(utils.DATA_DIR):
    os.mkdir(utils.DATA_DIR)


# Returns a list of ResponseCode objects.
def markup_response_codes(image):
  response_codes = []
  # Iterate through and mark each scan code.
  question_number = 1
  # all non-alphanumeric chars
  while True:
    print ("Please mark each response code in survey question %d." % question_number)

    bounding_box = utils.markup_page(image)
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


def save_response_codes(list_id, response_codes):
  # Save out the ResponseCodes objects to dict.
  response_code_dict = {}
  for ctr in range(len(response_codes)):
    response_code = response_codes[ctr]
    response_code_dict[ctr] = response_code.get_dict()

  with open('{}{}/{}'.format(utils.DATA_DIR, list_id, utils.RESPONSE_CODES_FILENAME), "w+") as f:
    json.dump(response_code_dict, f)


# Get the box around the all the respond codes.
def get_response_codes_bounding_box(response_codes, page):
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


# Returns the number of pages in the walklist
DPI = 300   # 300 dpi, optimal for tesseract
def ingest_walklist(list_id, filepath, rotate_dir):
  # convert PDF pages to images
  pages = convert_from_path(filepath, DPI)  
  num_pages = len(pages)
  for page_number in range(num_pages):
      pages[page_number].save(utils.get_page_filename(list_id, page_number), 'JPEG')

  print("Done ingesting the walklist PDF.")
  return num_pages


# Markup a clean page to get the list_id
def ingest_clean_page(filepath, page_number, rotate_dir):
  pages = convert_from_path(filepath, DPI) 
  temp_path = "%s%s" % (utils.TEMP_DIR, utils.CLEAN_IMAGE_FILENAME)
  pages[page_number].save(temp_path, 'JPEG')  # Save first page out to temp so we can read it in again
  page_to_markup = utils.load_page(temp_path, rotate_dir)
  
  print ("Please markup the List Id on the page.")
  bounding_box = utils.markup_page(page_to_markup)
  list_id = utils.get_list_id(utils.get_roi(page_to_markup, bounding_box))

  # Make the list id directory
  list_dir = '{}{}'.format(utils.DATA_DIR, list_id)
  if not os.path.exists(list_dir):
    print("Making the directory %s" % list_dir)
    os.mkdir(list_dir)

  # Make the walklist directory
  walklist_dir = '{}/{}'.format(list_dir, utils.WALKLIST_DIR)
  if os.path.exists(walklist_dir):
    shutil.rmtree(walklist_dir)
  os.mkdir(walklist_dir)
  print("Making the directory %s" % walklist_dir)

  # Save the bounding box out.
  utils.save_ref_boxes(list_id, {"list_id": bounding_box})

  # Save the file out to the correct directory.
  clean_filepath = '{}{}/{}'.format(utils.DATA_DIR, list_id, utils.CLEAN_IMAGE_FILENAME)
  print("Saving image to %s" % clean_filepath)
  cv2.imwrite(clean_filepath, page_to_markup)

  print("Done ingesting the clean PDF.")  
  return list_id, page_to_markup


def main():
  args = parse_args()
  check_for_errors(args)

  list_id, clean_page = ingest_clean_page(args["clean"], args["clean_page_number"], args["rotate_dir"])  
  num_pages = ingest_walklist(list_id, args["walklist"], args["rotate_dir"])

  response_codes = []
  if not args["skip_markup"]:
    response_codes = markup_response_codes(clean_page)
  else:
    response_codes = utils.load_response_codes(list_id)

  bounding_box = get_response_codes_bounding_box(response_codes, clean_page)

  # Normalize the response code coords
  if not args["skip_markup"]:
    for code in response_codes:
      code.coords = (code.coords[0] - bounding_box[0][0], code.coords[1] - bounding_box[0][1])

  # Save out the ResponseCodes themselves.
  save_response_codes(list_id, response_codes)

  # Save out the response code image.
  rc_image_path = '{}{}/{}'.format(utils.DATA_DIR, list_id, utils.RESPONSE_CODES_IMAGE_FILENAME)
  cv2.imwrite(rc_image_path, utils.get_roi(clean_page, bounding_box))

  # Save out the response code bounding box.
  utils.save_ref_boxes(list_id, {"response_codes": bounding_box})

  print ("Saved out reference response codes.")
  print ("Done, now run scan.py")

if __name__ == '__main__':
  main()


