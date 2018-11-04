'''usage: python scan.py'''

# import the necessary packages
import argparse
import csv
import cv2
import copy
import json
import numpy as np
from pyzbar import pyzbar
import os
import shutil
import sys
from enum import Enum
import math
import re
import img2pdf
import pprint
from typing import Union, Any, List, Optional, Tuple, Set, Dict, Sequence
import ast

import utils
from utils import BoundingBox, Point, ResponseCode, Image, Page, Size, Rotation
import test

pp = pprint.PrettyPrinter(width=41)

def parse_args():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-l", "--list_id", required=True,
    help="ID of the list to scan")
  ap.add_argument("-r", "--rotate_dir", default=None, 
    help="CW or CCW, rotate the page 90 degrees")
  ap.add_argument("--start_page", type=int, default=0, 
    help="The page number to start from.")
  ap.add_argument("--manual_review", action='store_true', 
    help="Prompt the user to approve/reject each scan")
  ap.add_argument("-t", "--test_file", default=None, 
    help="Path to a benchmark file for testing")
  return vars(ap.parse_args())


def check_files_exist(list_id: str) -> bool:
  # check if there is at least one walklist file
  walklist_dir_path = '{}{}/{}'.format(utils.DATA_DIR, list_id, utils.WALKLIST_DIR)
  has_walklists = False
  if not os.path.exists(walklist_dir_path):
    print ("No {} directory found!".format(walklist_dir_path))
    sys.exit()
  else:
    for file in os.listdir(walklist_dir_path):
      if file.endswith(".jpg"):
        has_walklists = True
        break
    if not has_walklists:
      print ("No walklists found in {}".format(walklist_dir_path))
      sys.exit()

  # check if there is a response codes files
  response_codes_json_path = '{}{}/{}'.format(utils.DATA_DIR, list_id, utils.RESPONSE_CODES_FILENAME)
  try:
      fh = open(response_codes_json_path, 'r')
  except FileNotFoundError:
    print ("Response Codes JSON file not found at {}".format(response_codes_json_path))
    sys.exit()

  # no errors
  return True

""" Returns (barcode_coords, voter_id)"""
def extract_barcode_info(barcode, image: Image) -> Optional[Tuple[BoundingBox, str]]:
  data = barcode.data.decode("utf-8")

  voter_id = re.sub(r'\W+', '', data) # remove any non-alphanumeric characters

  # check if it's a valid voter_id
  id_regex_match = re.match(r'\w{10,}CA', voter_id)
  if id_regex_match:
    voter_id = id_regex_match.group(0)
    if utils.__DEBUG__:
      print('Voter ID: {}'.format(voter_id))
  else:
    print('Invalid voter id {}, skipping.'.format(voter_id))
    return None

  voter_id = voter_id[:-2]  # remove the CA at the end

  (x, y, w, h) = barcode.rect
  barcode_bb = BoundingBox(Point(x, y), Point(x + w, y + h))

  # draw image
  if utils.__DEBUG__:
    markup_image = image.raw_image

    # extract the the barcode
    pts = np.array([[[x, y] for (x, y) in barcode.polygon]], np.int32)
    cv2.polylines(markup_image, pts, True, (0, 0, 255), 2)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # the barcode data is a bytes object so if we want to draw it on
    # our output image we need to convert it to a string first
    # draw the barcode data and barcode type on the image
    text = "{}".format(data)
    cv2.putText(markup_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
      0.5, (0, 0, 255), 2)
    Image(markup_image).show()

    # print the barcode type and data to the terminal
    print("[INFO] Found barcode: {}".format(barcode))

  return barcode_bb, voter_id


def get_response_for_barcode(barcode_coords: BoundingBox, first_response_coords: BoundingBox, 
                             page_size: Size) -> BoundingBox:
  ret_bb = BoundingBox(Point(first_response_coords.top_left.x, barcode_coords.top_left.y),
                             Point(first_response_coords.bottom_right.x, 
                             barcode_coords.bottom_right.y + first_response_coords.height))
  return ret_bb.add_padding(10, page_size)


def get_response_including_barcode(barcode_coords: BoundingBox, first_response_coords: BoundingBox,
                                   page_size: Size) -> BoundingBox:
  ret_bb = BoundingBox(Point(0, barcode_coords.top_left.y),
                             Point(barcode_coords.bottom_right.x, 
                             barcode_coords.bottom_right.y + first_response_coords.height))
  return ret_bb.add_padding(15, page_size)


CONTOUR_LOWER_THRESH = 1900
CONTOUR_UPPER_THRESH = 5000
def get_circle_centers(raw_diff: np.array) -> Tuple[List[Point], bool]:
  # find contours in the thresholded image
  _, contours, hierarchy = cv2.findContours(raw_diff.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
  # loop over the contours, find the good ones and get the center of them
  raw_diff = cv2.cvtColor(raw_diff, cv2.COLOR_GRAY2BGR)
  mask = np.ones(raw_diff.shape[:2], dtype="uint8") * 255
  contour_centers = []
  has_error = False
  for c in contours:
    # Remove the contours too small to be circles
    area = cv2.contourArea(c) 
    hull = cv2.convexHull(c)

    # compute the center of the contour
    M = cv2.moments(hull)
    center = Point(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    if area < CONTOUR_LOWER_THRESH:
      if utils.__DEBUG__:
        cv2.drawContours(mask, [c], -1, 0, -1)
      continue
    elif area > CONTOUR_UPPER_THRESH:
      has_error = True
    else:
      cv2.circle(raw_diff, center.to_tuple(), 7, (0,0,255),-1)
      contour_centers.append(center)
      
    # show the image
    if utils.__DEBUG__:
      cv2.drawContours(raw_diff, [c], -1, (0,255,0), 3)
      raw_diff = cv2.bitwise_and(raw_diff, raw_diff, mask=mask)
      cv2.drawContours(raw_diff, [hull], -1, (0,255,0), 3)
      Image(raw_diff).show()

  return contour_centers, has_error


# Find all contour centers that are close enough to a response code to be considered
# "selected"
def centers_to_responses(centers: List[Point],
                         response_codes: List[ResponseCode], 
                         aligned_response_codes: Image) -> List[ResponseCode]:
  DISTANCE_THRESH = 35
  SPLIT_DIST_THRESH = 5

  selected_responses = []
  for center in centers:
    (closest_code, closest_dist) = None, 9999999
    (second_code, second_dist) = None, 9999999
    for code in response_codes:
      if utils.__DEBUG__:
        cv2.circle(aligned_response_codes.raw_image, code.coords.to_tuple(), 4, (255, 0, 0), thickness=3)

      dist = center.calc_distance(code.coords)

      if dist < DISTANCE_THRESH:
        if dist < closest_dist:
          closest_code = code
          closest_dist = dist
        elif dist < DISTANCE_THRESH:
          second_code = code
          second_dist = dist

    # The case of only appending the closest point.
    if (closest_code and not second_code) or \
       (closest_code and second_code and (second_dist - closest_dist > SPLIT_DIST_THRESH)):
      selected_responses.append(closest_code)
    # The split is too close to call, add both closest responses.
    elif (closest_code and second_code and (second_dist - closest_dist <= SPLIT_DIST_THRESH)): 
      selected_responses.append(closest_code)
      selected_responses.append(second_code)

  if utils.__DEBUG__:
    aligned_response_codes.show()
  return selected_responses


# Figure out if this current response code is bolded or not and return the appropriate
# diffed image and the aligned response codes.
def get_diffed_response_codes(cur_rc: Image, list_dir: str) -> Tuple[Image, Image]:
  cur_rc = cur_rc.threshold()

  # Load the reference response codes, bold and not bold version.
  ref_rc = Image.from_file(list_dir + utils.RESPONSE_CODES_IMAGE_FILENAME, Rotation.NONE)
  ref_rc = ref_rc.threshold()

  bold_ref_rc = Image.from_file(list_dir + utils.BOLD_RESPONSE_CODES_IMAGE_FILENAME, Rotation.NONE)
  bold_ref_rc = bold_ref_rc.threshold(bold_ref_rc)

  # Align and diff against both reference images.
  aligned_rc = cur_rc.align_to(ref_rc)
  bold_aligned_rc = cur_rc.align_to(bold_ref_rc)

  diff = aligned_rc.diff_against(ref_rc)
  bold_diff = bold_aligned_rc.diff_against(bold_ref_rc)

  # Count how many white pixels are in each diff.
  white_pixels = diff.numWhitePixels()
  bold_white_pixels = bold_diff.numWhitePixels()

  # The one with the least white pixels should be the correct image.
  if white_pixels < bold_white_pixels:
    return diff, aligned_rc
  else:
    return bold_diff, bold_aligned_rc


# Returns a list of circled response codes.
def get_circled_responses(response_bounding_box: BoundingBox, 
                          response_codes: List[ResponseCode],
                          page, list_dir) -> Tuple[List[ResponseCode], bool]:
  cur_response_codes = page.get_roi(response_bounding_box)
  diff, aligned_response_codes = get_diffed_response_codes(cur_response_codes, list_dir)

  # crop pixels to account for the alignment algo introducing whitespace
  raw_diff = diff.raw_image
  raw_diff = raw_diff[20:, 0:-10]
  raw_diff = cv2.medianBlur(raw_diff, 5)
  raw_diff = Image(raw_diff).threshold().grayscale().raw_image
  
  raw_diff = cv2.dilate(raw_diff, np.ones((5,5),np.uint8), iterations = 2)
  raw_diff = cv2.medianBlur(raw_diff, 5)
  raw_diff = cv2.medianBlur(raw_diff, 5)
  
  if utils.__DEBUG__:
    Image(raw_diff).show()

  contour_centers, has_error = get_circle_centers(raw_diff)

  circled_responses: List = []
  if not has_error:
    circled_responses = centers_to_responses(contour_centers, response_codes, 
                                             aligned_response_codes)
  print ("Error? ", has_error)

  return circled_responses, has_error


def manual_review(response_bounding_box: BoundingBox, 
                  page: Image, circled_responses: List[ResponseCode],
                  voter_id, response_codes: List[ResponseCode]) -> Tuple[bool, List[ResponseCode]]:
  user_verdict = None
  top_margin = 50

  # init response_image
  responses_image = page.get_roi(response_bounding_box).add_border(top=top_margin)

  # get list of responses
  response_pairs = []
  if circled_responses:
    for resp in circled_responses:
      response_pairs.append('Q{}: {}'.format(resp.question_number, resp.value))

      # add dots in the center of each highlighted response code
      cv2.circle(responses_image.raw_image, (resp.coords.x, resp.coords.y + top_margin), 6, (0,0,255),-1)

    # convert to a string
    response_string = ", ".join(response_pairs)
  else:
    response_string = 'None'
  
  # annotate the response image
  responses_question = "Responses: {}. Is this correct? (y|n|c|m|h|g|l)".format(response_string)

  # keep looping until the y or n key is pressed
  while True:
    # display the image and wait for a keypress
    key = responses_image.show(title="manual review", message=responses_question, resize=False)
   
    # if the 'y' key is pressed, user approves
    if key == ord("y"):
      user_verdict = True
      print('{}: Correct scan!'.format(voter_id))
      break
   
    # if the 'n' key is pressed, user rejects
    elif key == ord("n"):
      user_verdict = False
      print('{}: Incorrect scan :('.format(voter_id))
      circled_responses = get_correct_responses(response_codes)
      break

    # if the 'c' key is pressed, user rejects, correct answer is none
    elif key == ord("c"):
      user_verdict = False
      print('{}: Incorrect scan :('.format(voter_id))
      circled_responses = get_correct_responses(response_codes, 'c')
      break

    # if the 'm' key is pressed, user rejects, correct answer is MAT
    elif key == ord("m"):
      user_verdict = False
      print('{}: Incorrect scan :('.format(voter_id))
      circled_responses = get_correct_responses(response_codes, 'm')
      break

    # if the 'h' key is pressed, user rejects, correct answer is NH
    elif key == ord("h"):
      user_verdict = False
      print('{}: Incorrect scan :('.format(voter_id))
      circled_responses = get_correct_responses(response_codes, 'h')
      break

    # if the 'g' key is pressed, user rejects, correct answer is GTD
    elif key == ord("g"):
      user_verdict = False
      print('{}: Incorrect scan :('.format(voter_id))
      circled_responses = get_correct_responses(response_codes, 'g')
      break

    # if the 'l' key is pressed, user rejects, correct answer is MAT + NH
    elif key == ord("l"):
      user_verdict = False
      print('{}: Incorrect scan :('.format(voter_id))
      circled_responses = get_correct_responses(response_codes, 'l')
      break

  # close window  
  cv2.destroyAllWindows()

  # create empty list if None
  if circled_responses is None:
    circled_responses = []

  print('correct_responses:')
  print([code.value for code in circled_responses])

  return user_verdict, circled_responses


def get_correct_responses(response_codes: List[ResponseCode], shortcut_key=None) -> List[ResponseCode]:
  print('=======================================================')
  print("Please enter a comma-separated list of the correct responses. Enter 'n' if none.")

  # build dict of question: values pairs (for printing nicely)
  # and dict of answers_num: response pairs (for returning)
  response_codes_by_question: Dict = {}
  answers_to_code_objs: Dict = {}
  for resp in response_codes:
    # put in response_codes_by_question
    if not resp.question_number in response_codes_by_question:
      response_codes_by_question[resp.question_number] = []

    unique_answer = "{}{}".format(resp.question_number, resp.value)

    response_codes_by_question[resp.question_number].append(unique_answer)

    # put in answers_to_code_objs
    answers_to_code_objs[unique_answer.lower()] = resp

  # print out to terminal
  for question in response_codes_by_question:
    print('QUESTION {}'.format(question))
    print('   '.join(response_codes_by_question[question]))

  # populate correct responses
  correct_responses = []

  # build shortcuts dict
  shortcuts = {'c': 'n', 'm':['3mat'], 'h': ['3nh'], 'l': ['3mat', '3nh'], 'g': ['3gtd']}

  # check if a shortcut key was entered
  if shortcut_key:
    input_answers = shortcuts[shortcut_key]
  else:
   # get user input
    input_string = input('Enter correct responses: ').lower()
    input_answers = input_string.split(',')
  
  # convert input to answers
  if input_answers == 'n':
    return [] # return empty if no answers

  print('input answers: {}'.format(input_answers))

  for answer in input_answers:
    response_code = answers_to_code_objs[answer.strip()]
    correct_responses.append(response_code)

  return correct_responses


# TODO: complete this function once have multi-response checking
def error_check_responses(responses: List[ResponseCode]) -> bool:
  return False


def create_error_image(page: Page, barcode_coords: BoundingBox, first_response_coords: BoundingBox) -> Image:
  full_response_bounding_box = get_response_including_barcode(barcode_coords, 
      first_response_coords, page.size)
  error_image = page.get_roi(full_response_bounding_box)
  return error_image


def save_responses(responses: List[ResponseCode], voter_id: str, dict_writer) -> None:
  question_to_responses: Dict = {}
  for response in responses:
    key = "question_%s" % response.question_number
    if key not in question_to_responses:
      question_to_responses[key] = []
    question_to_responses[key].append(response.value)

  question_to_responses['voter_id'] = voter_id
  dict_writer.writerow(question_to_responses)


def generate_error_pages(error_images: List[Image], skipped_pages: List[Page], list_id: str) -> None:
  page = utils.make_blank_page()
  page.crop(top=int(2*utils.MARGIN), right=int(2*utils.MARGIN))  # to account for printing

  # calculate the number of error images that can fit on a page
  num_images_per_page = math.floor(page.size.h / error_images[0].size.h)

  # init error pages array
  error_pages = skipped_pages
  images_on_page = 0

  for i, error_image in enumerate(error_images):
    # Create new pages as necessary
    if i % num_images_per_page == 0 and i > 0:
      # save the previous page to the error_pages array
      page.add_border(top=int(2*utils.MARGIN), right=int(2*utils.MARGIN))
      error_pages.append(page)
      images_on_page = 0

    # add images to the page
    insert_point = Point(0, images_on_page * error_image.size.h)
    page = Page(page.insert_image(error_image, insert_point))

    # increment the images on page counter
    images_on_page += 1 

  # add the last page to the array
  page.add_border(top=int(2*utils.MARGIN), right=int(2*utils.MARGIN))
  error_pages.append(page)

  # save out a pdf
  save_error_pages(error_pages, list_id)


def save_error_pages(error_pages: List[Page], list_id: str) -> None:
  # create error dir
  error_dir_path = '{}{}/{}'.format(utils.DATA_DIR, list_id, utils.ERROR_PAGES_DIR)
  if not os.path.exists(error_dir_path):
    os.mkdir(error_dir_path)

  # save out error pages as images
  for i, page in enumerate(error_pages):
    filename = '{}{}_error_page_{}.jpg'.format(error_dir_path, list_id, i)
    cv2.imwrite(filename, page)

  # get list of images filepaths to pass into img2pdf
  error_images = ['{}{}'.format(error_dir_path, i) for i in os.listdir(error_dir_path) if i.endswith(".jpg")]
  
  # convert images to a combined pdf
  error_pdf_filename = '{}{}_errors.pdf'.format(error_dir_path, list_id)
  with open(error_pdf_filename, "wb") as f:
    f.write(img2pdf.convert(error_images))


def scan_page(list_id: str, rotate_dir: Rotation, args, page_number: int, ref_page: Page, ref_bounding_boxes: Dict[str, BoundingBox], list_dir: str,
              results_scans, results_stats, results_errors, previous_scans: dict, backup_writer):
  page = Page.from_file(utils.get_page_filename(list_id, page_number), rotate_dir)
  response_codes = utils.load_response_codes(list_id)

  # align page
  aligned_page = page.align_to(ref_page)
  if utils.__DEBUG__:
    aligned_page.show(title="aligned page")

  # confirm page has the correct list_id
  page_list_id = page.get_list_id(ref_bounding_boxes["list_id"])
  if page_list_id != list_id:
    valid_id, page = handle_missing_page_id(aligned_page, page, list_id, ref_bounding_boxes["list_id"], page_number)
    if not valid_id:
      print('Error: Page {} has ID {}, but active ID is {}. Page {} has been skipped.'.format(page_number+1, page_list_id, list_id, page_number+1))
      results_errors['skipped_pages'].append({page_number: page})
      return results_scans, results_stats, results_errors

  # find the barcodes in the image and decode each of the barcodes
  # Barcode scanner needs the unthresholded image.
  barcodes = pyzbar.decode(page.raw_image)
  if len(barcodes) == 0:
    print('Error: Cannot find barcodes. Page {} has been skipped.'.format(page_number+1))
    results_errors['skipped_pages'].append({page_number: page})
    return results_scans, results_stats, results_errors

  # loop over the detected barcodes
  voter_ids: Set = set()
  for barcode in barcodes:
    results_scans, results_stats, results_errors = scan_barcode(barcode, page, ref_bounding_boxes, list_dir, response_codes, args, results_scans, results_stats, results_errors, previous_scans, backup_writer, voter_ids)
  check_num_barcodes(page, list_dir, len(voter_ids), results_stats)

  if utils.__DEBUG__:
    page.show()

  return results_scans, results_stats, results_errors


def check_num_barcodes(page: Page, list_dir: str, num_scanned_barcodes: int, results_stats) -> None:
  # Manually loop and count barcodes
  num_actual_barcodes = 0

  for line_number in range(1, utils.MAX_BARCODES_ON_PAGE + 1):
    line_bb = page.get_line_bb(line_number, list_dir)
    # extract the barcode portion
    line_bb.top_left.x = line_bb.bottom_right.x - 700
    barcode_roi = page.get_roi(line_bb).invert()

    BARCODE_EXISTS_THRESHOLD = 20000  # if a barcode exists in the area it averages 29k black pixels.
    if barcode_roi.numWhitePixels() > BARCODE_EXISTS_THRESHOLD:
      num_actual_barcodes += 1
    else:
      break  # we have likely reached the end of the page.

  if num_actual_barcodes < num_scanned_barcodes:
    print ("Something went wrong with the image alignment! Cannot accurately count missed barcodes.")
  elif num_actual_barcodes > num_scanned_barcodes:
    results_stats["num_missed_barcodes"] += num_actual_barcodes - num_scanned_barcodes


def handle_missing_page_id(aligned_page: Page, original_page: Page, 
                           list_id: str, id_bounding_box: BoundingBox, page_number: int) -> Tuple[bool, Page]:
  # to check if it's a homography issue, see if the list ID is visible on the raw page
  test_id = original_page.get_list_id(id_bounding_box)
  if test_id == list_id:
    print('Homography error on page {}, using uncorrected page instead.'.format(page_number))
    return True, original_page

  # didn't find on raw page, ask the user to confirm the ID
  id_area = original_page.get_roi(id_bounding_box)
  id_area = id_area.add_border(top=30)

  # annotate the response image
  question = "ID: {}? (y|n)".format(list_id)

  # keep looping until the y or n key is pressed
  while True:
    # display the image and wait for a keypress
    key = id_area.show(title="review", message=question, resize=False)
   
    # if the 'y' key is pressed, user approves
    if key == ord("y"):
      cv2.destroyAllWindows()
      return True, original_page
   
    # if the 'n' key is pressed, user rejects
    elif key == ord("n"):
      cv2.destroyAllWindows()
      break

  return False, aligned_page


def scan_barcode(barcode, page, ref_bounding_boxes, list_dir, response_codes, args, results_scans, results_stats, results_errors, previous_scans, backup_writer, voter_ids) -> Tuple[list, dict, dict]:
  barcode_info = extract_barcode_info(barcode, page)

  # skip if not a valid barcode
  if not barcode_info:
    return results_scans, results_stats, results_errors

  barcode_coords, voter_id = barcode_info

  # Check if the barcode has already been read, skip if so.
  if voter_id in voter_ids:
    return results_scans, results_stats, results_errors
  else:
    voter_ids.add(voter_id)

  # increment barcodes counter
  results_stats['num_scanned_barcodes'] += 1

  # use the existing info if already scanned, unless in testing mode
  if voter_id in previous_scans and not args["test_file"]:
    print('Already scanned {}'.format(voter_id))
    results_dict = previous_scans[voter_id]

  # new barcode to scan
  else:
    if utils.__DEBUG__:
      cv2.rectangle(page, barcode_coords.top_left.to_tuple(), barcode_coords.bottom_right.to_tuple(), 
                    (255, 0, 255), 3)
      page.show()

    # Get the corresponding response codes region
    response_bounding_box = get_response_for_barcode(barcode_coords, ref_bounding_boxes["response_codes"], page.size)

    # Figure out which ones are circled
    ref_response_codes = Page.from_file(list_dir + utils.RESPONSE_CODES_IMAGE_FILENAME, Rotation.NONE)
    circled_responses, has_error = get_circled_responses(response_bounding_box, response_codes, page, list_dir)
    has_error = has_error or error_check_responses(circled_responses)

    # if has an error at this point, add to the error tally
    if has_error:
      results_stats['num_error_barcodes'] += 1

    # Do manual review if error or if flagged, unless in testing mode
    if (has_error or args["manual_review"]) and not args["test_file"]:
      verdict_right, circled_responses = manual_review(response_bounding_box, page, circled_responses, voter_id, response_codes)

      # if user verdict is false, add the voter_id to the list of incorrect scans
      if not verdict_right:
        results_stats['incorrect_scans'].append(voter_id)

    # if in testing mode, convert any None circled_responses to an empty list
    if args["test_file"] and circled_responses is None:
      circled_responses = []

    # build results dict
    results_dict = build_results_dict(voter_id, circled_responses)

  # save results
  results_scans.append(results_dict)
  write_to_backup(results_dict, backup_writer)

  return results_scans, results_stats, results_errors


def build_results_dict(voter_id: str, responses: List[ResponseCode]) -> dict:
  results = {}
  results['voter_id'] = voter_id

  question_to_responses: Dict = {}
  for response in responses:
    key = "question_{}".format(response.question_number)
    while key not in question_to_responses:
      question_to_responses[key] = []
    question_to_responses[key].append(response.value)

  results['questions'] = question_to_responses

  return results


def write_to_backup(results_dict, backup_writer) -> None:
  # build backup dict
  backup_dict = {'voter_id': results_dict['voter_id']}
  for question in results_dict['questions']:
    backup_dict[question] = results_dict['questions'][question]

  # add to CSV
  backup_writer.writerow(backup_dict)


def output_results_csv(list_id: str, list_dir: str, results_scans) -> None:
  # get unique ordered list of questions
  questions_and_answers = [scan['questions'] for scan in results_scans]
  questions = list(set([k for d in questions_and_answers for k in d.keys()]))
  questions.sort()
  if 'voter_id' in questions:
    questions.remove('voter_id')

  formatted_results = []
  for scan in results_scans:
    formatted_scan = {}
    formatted_scan['primary_id'] = scan['voter_id']
    for question in questions:
      if question in scan['questions'].keys():
        answer_set = utils.convertStringListToList(scan['questions'][question])

        for i, answer in enumerate(answer_set):
          if i == 0:
            colname = question
          else:
            colname = '{}_response{}'.format(question, i+1)
          formatted_scan[colname] = answer
      
    formatted_results.append(formatted_scan)

  # Prep the output file
  all_colnames: Sequence[str] = sorted(set().union(*(d.keys() for d in formatted_results)))
  with open("{}/results_{}.csv".format(list_dir, list_id), 'w+') as output_file:
    dict_writer = csv.DictWriter(output_file, all_colnames)
    dict_writer.writeheader()
    dict_writer.writerows(formatted_results)


def show_statistics(results_stats, args) -> None:
  print('======== STATISTICS ========')
  print('Scanned {} barcodes.'.format(results_stats['num_scanned_barcodes']))
  print('Missed {} barcodes.'.format(results_stats['num_missed_barcodes']))
  print('{} ({}%) had system-detected errors.'.format(results_stats['num_error_barcodes'], round((results_stats['num_error_barcodes']/results_stats['num_scanned_barcodes'])*100)))

  if args['manual_review']:
    num_no_system_errors = results_stats['num_scanned_barcodes'] - results_stats['num_error_barcodes']
    error_rate = len(results_stats['incorrect_scans']) / num_no_system_errors
    accuracy_rate = round((1-error_rate)*100)
    print('{} of {} were incorrectly scanned. {}% accuracy rate.'.format(len(results_stats['incorrect_scans']), num_no_system_errors, accuracy_rate))


def prep_backup_csv(list_dir, list_id) -> Tuple[str, list]:

    backup_filename = '{}backup_{}.csv'.format(list_dir, list_id)
    response_codes = utils.load_response_codes(list_id)
    questions = sorted(list(set(['question_{}'.format(code.question_number) for code in response_codes])))
    colnames = ['voter_id'] + questions
    
    return backup_filename, colnames


def load_previous_scans(backup_filename: str, args) -> dict:
  # check if a backup files exists
  if not os.path.exists(backup_filename):
    print('No previous scans, creating backup file')
    return {}

  print('Loading previous scans')
  previous_scans = utils.extractCSVtoDict(backup_filename)

  # copy previous backup
  prev_filename, prev_fileext = os.path.splitext(backup_filename)
  new_filepath = prev_filename + '_prev' + prev_fileext
  shutil.copy(backup_filename, new_filepath)

  return previous_scans


def main() -> None:
  args = parse_args()
  list_id = args["list_id"]
  check_files_exist(list_id)
  list_dir: str = utils.get_list_dir(list_id)
  rotate_dir = utils.map_rotation(args["rotate_dir"])

  ref_bounding_boxes = utils.load_ref_boxes(list_dir)
  ref_page = Page.from_file(list_dir + utils.CLEAN_IMAGE_FILENAME, rotate_dir)

  # init results object
  results_scans: list = []

  # things to track for error reporting
  results_stats = {}
  results_stats['num_scanned_barcodes'] = 0
  results_stats['num_missed_barcodes'] = 0
  results_stats['num_error_barcodes'] = 0
  results_stats['incorrect_scans'] = []

  # stuff to build error PDF for human scanning
  results_errors: dict = {}
  results_errors['errors_for_human'] = []
  results_errors['skipped_pages'] = []

  # write out to CSV backup as process the list
  backup_filename, colnames = prep_backup_csv(list_dir, list_id)
  previous_scans = load_previous_scans(backup_filename, args)

  with open(backup_filename, mode='w') as backup_csv:
    backup_writer = csv.DictWriter(backup_csv, fieldnames=colnames)
    backup_writer.writeheader()

    num_pages = len(os.listdir("{}/{}".format(list_dir, utils.WALKLIST_DIR)))
    for page_number in range(args['start_page'], num_pages):

      print('===Scanning page {} of {} ==='.format(page_number+1, num_pages))

      results_scans, results_stats, results_errors = scan_page(list_id, rotate_dir, args, page_number, ref_page, ref_bounding_boxes, list_dir, results_scans, results_stats, results_errors, previous_scans, backup_writer)

  # output results
  output_results_csv(args['list_id'], list_dir, results_scans)
  # generate_error_pages(results_errors['errors_for_human'], results_errors['skipped_pages'], args['list_id'])

  # show list of skipped pages
  print('Skipped {} pages:'.format(len(results_errors['skipped_pages'])))
  for page in results_errors['skipped_pages']:
    print(page.keys())

  # run test suite if set
  if args["test_file"]:
    test.run_test_suite(args['test_file'], results_scans)

  else:
    # print statistics
    show_statistics(results_stats, args)

if __name__ == '__main__':
  main()


