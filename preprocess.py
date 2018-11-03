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
from typing import Union, Any, List, Optional, Tuple

import utils
from utils import ResponseCode, BoundingBox, Point


def parse_args():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-w", "--walklist", required=True,
    help="path to the PDF of the scanned walklist.")
  ap.add_argument("-c","--clean", required=True,
    help="path to the PDF to use as a reference.")
  ap.add_argument("-p","--page_number", type=int, default=1,
    help="The page number of the page with clean response codes to markup as part of the ingestion.")
  ap.add_argument("-l","--line_number", type=int, default=1,
    help="The line number of the clean response codes to markup as part of the ingestion.")
  ap.add_argument("-bp","--bold_page_number", type=int, default=1,
    help="The page number of the clean, bolded response codes as part of the ingestion.")
  ap.add_argument("-bl","--bold_line_number", type=int, default =1,
    help="The line number of clean, bolded response codes.")
  ap.add_argument("--skip_bold", action='store_true',
    help="Skip marking up bold responses.")
  ap.add_argument("--skip_markup", action='store_true', 
    help="For dev purposes, if you want to skip marking up the page")
  ap.add_argument("-r", "--rotate_dir", default=None, 
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
def markup_response_codes(page, list_id: str, line_number: int):
  # Get line in the page.
  line_bb = utils.get_line_bb(page, line_number, utils.get_list_dir(list_id), padding=100, right_half=True)
  markup_roi = page[line_bb.top_left.y : line_bb.bottom_right.y, line_bb.top_left.x:]

  # Iterate through and mark each scan code.
  response_codes = []
  question_number = 1
  while True:
    print ("Please mark each response code in survey question %d." % question_number)

    bounding_box = utils.markup_image(markup_roi)
    text = utils.run_ocr(markup_roi, bounding_box, utils.SegmentationMode.SINGLE_WORD)
     # sometimes OCR picks up stray symbols, get rid of them.
    text = ''.join(ch for ch in text if ch.isalnum())

    bounding_box = bounding_box.update_coordinate_system(line_bb.top_left)
    roi = utils.get_roi(page, bounding_box)
    response_code = ResponseCode(bounding_box, question_number, text)

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

    print (r"""Hit enter (no input) to mark another response in the same survey question.
            Enter 'n' to move to the next survey question. Enter 'q' to finish. [enter|n|q]""")
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
    response_code_dict[ctr] = response_codes[ctr].get_dict()

  with open(utils.get_list_dir(list_id) + utils.RESPONSE_CODES_FILENAME, "w+") as f:
    json.dump(response_code_dict, f)


# Get the box around the all the respond codes.
def get_response_codes_bounding_box(response_codes, page):
  x_min = 99999999  # TODO: switch out for INT_MAX
  x_max = 0
  y_min = 99999999
  y_max = 0

  for response_code in response_codes:
    x_min = min(x_min, response_code.bounding_box.top_left.x)
    y_min = min(y_min, response_code.bounding_box.top_left.y)
    x_max = max(x_max, response_code.bounding_box.bottom_right.x)
    y_max = max(y_max, response_code.bounding_box.bottom_right.y)

  ret_bb = BoundingBox(Point(x_min, y_min), Point(x_max, y_max))
  return ret_bb.add_padding(20, page.shape[:2])


# Returns the number of pages in the walklist
DPI = 300   # 300 dpi, optimal for tesseract
def ingest_walklist(list_id, filepath):
  # convert PDF pages to images
  pages = convert_from_path(filepath, DPI)  
  num_pages = len(pages)
  for page_number in range(num_pages):
    print("Ingesting page {} of {}".format(page_number, num_pages))
    pages[page_number].save(utils.get_page_filename(list_id, page_number), 'JPEG')

  print("Done ingesting the walklist PDF.")
  return num_pages


def get_temp_page(pages, page_number: int, rotate_dir: str, temp_filename: str) -> np.array:
  temp_path = "%s%s" % (utils.TEMP_DIR, temp_filename)
  pages[page_number - 1].save(temp_path, 'JPEG')  # Save specified page out to temp so we can read it in again
  return utils.load_image(temp_path, rotate_dir)


# Markup a clean page to get the list_id
def ingest_clean_page(filepath, page_number, rotate_dir, bold_page_number, bold_line_number):

  print("Converting PDF...")
  pages = convert_from_path(filepath, DPI) 
  page_to_markup = get_temp_page(pages, page_number, rotate_dir, utils.CLEAN_IMAGE_FILENAME)
  
  print ("Please markup the List Id on the page.")
  list_id_bb = utils.markup_image(page_to_markup)
  list_id = utils.get_list_id(utils.get_roi(page_to_markup, list_id_bb))
  list_dir = utils.get_list_dir(list_id)

  print ("Please markup the FIRST barcode on the page.")
  first_barcode_bb = utils.markup_image(page_to_markup)

  # Make the list id directory
  if not os.path.exists(list_dir):
    print("Making the directory %s" % list_dir)
    os.mkdir(list_dir)

  # Make the walklist directory
  walklist_dir = '{}{}'.format(list_dir, utils.WALKLIST_DIR)
  if os.path.exists(walklist_dir):
    shutil.rmtree(walklist_dir)
  os.mkdir(walklist_dir)
  print("Making the directory %s" % walklist_dir)

  # Save the bounding box out.
  utils.save_ref_boxes(list_dir, {"list_id": list_id_bb.to_list(), "first_barcode": first_barcode_bb.to_list()})

  # Save the file out to the correct directory.
  clean_filepath = '{}{}/{}'.format(utils.DATA_DIR, list_id, utils.CLEAN_IMAGE_FILENAME)
  print("Saving image to %s" % clean_filepath)
  cv2.imwrite(clean_filepath, page_to_markup)

  # Markup the bold responses.
  if bold_page_number and bold_line_number:
    bold_page_to_markup = get_temp_page(pages, bold_page_number, rotate_dir, "bold_" + utils.CLEAN_IMAGE_FILENAME)

    print ("Please markup the bold response codes on the page.")
    line_bb = utils.get_line_bb(bold_page_to_markup, bold_line_number, list_dir, padding=100, right_half=True)
    bold_responses_bb = utils.markup_image(utils.get_roi(bold_page_to_markup, line_bb))
    bold_responses_bb = bold_responses_bb.update_coordinate_system(line_bb.top_left)

    # Save the bounding box out.
    utils.save_ref_boxes(list_dir, {"bold_response_codes": bold_responses_bb.to_list()})
    # Save the image out.
    bold_rc_image_path = list_dir + utils.BOLD_RESPONSE_CODES_IMAGE_FILENAME
    cv2.imwrite(bold_rc_image_path, utils.get_roi(bold_page_to_markup, bold_responses_bb))

  # Save the file out to the correct directory.
  clean_filepath = list_dir + utils.CLEAN_IMAGE_FILENAME
  print("Saving image to %s" % clean_filepath)
  cv2.imwrite(clean_filepath, page_to_markup)

  print("Done ingesting the clean PDF.")  
  return list_id, page_to_markup


def main():
  args = parse_args()
  check_for_errors(args)

  if args["skip_bold"]:
     args["bold_page_number"] = None
     args["bold_line_number"] = None

  list_id, clean_page = ingest_clean_page(args["clean"], args["page_number"], args["rotate_dir"],
                                          args["bold_page_number"], args["bold_line_number"])  
  num_pages = ingest_walklist(list_id, args["walklist"])
  list_dir = utils.get_list_dir(list_id)

  response_codes = []
  if not args["skip_markup"]:
    response_codes = markup_response_codes(clean_page, list_id, args["line_number"])
  else:
    response_codes = utils.load_response_codes(list_id)

  bounding_box = get_response_codes_bounding_box(response_codes, clean_page)

  # Normalize the response code coords
  if not args["skip_markup"]:
    for code in response_codes:
      code.coords = (code.coords.x - bounding_box.top_left.x, code.coords.y - bounding_box.top_left.y)
    # Save out the ResponseCodes themselves.
    save_response_codes(list_id, response_codes)

    # Save out the response code image.
    rc_image_path = utils.get_list_dir(list_id) + utils.RESPONSE_CODES_IMAGE_FILENAME
    rc_roi = utils.get_roi(clean_page, bounding_box)
    cv2.imwrite(rc_image_path, rc_roi)

    # Save out the response code bounding box.
    utils.save_ref_boxes(list_dir, {"response_codes": bounding_box.to_list()})

  if not args["skip_bold"]:
    # Save out the bold response code image.
    list_dir = utils.get_list_dir(list_id)
    rc_roi = utils.load_image(list_dir + utils.RESPONSE_CODES_IMAGE_FILENAME)
    bold_rc_roi = utils.load_image(list_dir + utils.BOLD_RESPONSE_CODES_IMAGE_FILENAME)
    aligned_bold_rc_roi, _ = utils.alignImages(bold_rc_roi, rc_roi)

    if utils.__DEBUG__:
      utils.show_image(aligned_bold_rc_roi)

    bold_rc_image_path = list_dir + utils.BOLD_RESPONSE_CODES_IMAGE_FILENAME
    cv2.imwrite(bold_rc_image_path, aligned_bold_rc_roi)

  print ("Saved out reference response codes.")
  print ("Done, now run scan.py")

if __name__ == '__main__':
  main()


