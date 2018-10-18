'''usage: python scan.py'''

# import the necessary packages
import argparse
import csv
import cv2
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

import utils


def parse_args():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-l", "--list_id", required=True,
    help="ID of the list to scan")
  ap.add_argument("--rotate_dir", default=None, 
    help="CW or CCW, rotate the page 90 degrees")
  ap.add_argument("--start_page", type=int, default=0, 
    help="The page number to start from.")
  return vars(ap.parse_args())


def check_files_exist(list_id):

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
def extract_barcode_info(barcode, image):
  data = barcode.data.decode("utf-8")

  voter_id = re.sub(r'\W+', '', data) # remove any non-alphanumeric characters
  voter_id = voter_id[:-2]  # remove the CA at the end

  (x, y, w, h) = barcode.rect
  barcode_coords = ((x, y), (x + w, y + h))

  # draw image
  if utils.__DEBUG__:

    # make a deep copy of the image to avoid annotating the original
    markup_image = image.copy()

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

    # print the barcode type and data to the terminal
    print("[INFO] Found barcode: {}".format(barcode))

  return barcode_coords, voter_id


def get_voter_id(image, bounding_box):
  text = run_ocr(image, bounding_box)
  print(text)
  voter_id = text.split(' ')[0]

  if utils.__DEBUG__:
    print ("voter_id: ", voter_id)

  return voter_id


def get_response_for_barcode(barcode_coords, first_response_coords):
  padding = 0 # meant this for the next function, but seems to work better!
  response_h = first_response_coords[1][1] - first_response_coords[0][1]
  return ((first_response_coords[0][0] - padding, barcode_coords[0][1] - padding),
          (first_response_coords[1][0] + padding, barcode_coords[0][1] + response_h + padding))


def get_response_including_barcode(barcode_coords, first_response_coords):
  padding = 15
  response_height = first_response_coords[1][1] - first_response_coords[0][1]
  return ((0, barcode_coords[0][1] - padding),
          (barcode_coords[1][0] + padding, barcode_coords[0][1] + response_height + padding))


def calculateDistance(x1, y1, x2, y2):  
  return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  


CONTOUR_LOWER_THRESH = 1900
CONTOUR_UPPER_THRESH = 5000
def get_circle_centers(diff):
  # find contours in the thresholded image
  _, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 
  # loop over the contours, find the good ones and get the center of them
  diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
  mask = np.ones(diff.shape[:2], dtype="uint8") * 255
  contour_centers = []
  has_error = False
  for c in contours:
    # Remove the contours too small to be circles
    area = cv2.contourArea(c) 
    hull = cv2.convexHull(c)

    # compute the center of the contour
    M = cv2.moments(hull)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    if area < CONTOUR_LOWER_THRESH:
      if utils.__DEBUG__:
        cv2.drawContours(mask, [c], -1, 0, -1)
      continue
    elif area > CONTOUR_UPPER_THRESH:
      has_error = True
    else:
      cv2.circle(diff, center, 7, (0,0,255),-1)
      contour_centers.append(center)
      
    # show the image
    if utils.__DEBUG__:
      cv2.drawContours(diff, [c], -1, (0,255,0), 3)
      diff = cv2.bitwise_and(diff, diff, mask=mask)
      cv2.drawContours(diff, [hull], -1, (0,255,0), 3)
      utils.show_image(diff)

  return contour_centers, has_error


# Find all contour centers that are close enough to a response code to be considered
# "selected"
def centers_to_responses(centers, response_codes, aligned_response_codes):
  DISTANCE_THRESH = 35
  SPLIT_DIST_THRESH = 5

  aligned_response_codes = cv2.cvtColor(aligned_response_codes, cv2.COLOR_GRAY2BGR)
  selected_responses = []
  for center in centers:
    (closest_code, closest_dist) = None, 9999999
    (second_code, second_dist) = None, 9999999
    for code in response_codes:
      if utils.__DEBUG__:
        cv2.circle(aligned_response_codes, tuple(code.coords), 4, (255, 0, 0), thickness=3)

      dist = calculateDistance(center[0], center[1], code.coords[0], code.coords[1])

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

  utils.show_image(aligned_response_codes)
  return selected_responses


# Returns a list of circled response codes.
def get_circled_responses(response_bounding_box, response_codes, page, ref_response_codes):
  # carve out the roi
  cur_response_codes = utils.get_roi(page, list(response_bounding_box))
  cur_response_codes = cv2.cvtColor(cur_response_codes, cv2.COLOR_BGR2GRAY)
  cur_response_codes = utils.threshold(cur_response_codes)
  utils.show_image(cur_response_codes)

  ref_response_codes = cv2.cvtColor(ref_response_codes, cv2.COLOR_BGR2GRAY)
  ref_response_codes = utils.threshold(ref_response_codes)
  utils.show_image(ref_response_codes)

  aligned_response_codes, _ = utils.alignImages(cur_response_codes, ref_response_codes)
  diff = cv2.bitwise_xor(aligned_response_codes, ref_response_codes)

  # crop pixels to account for the alignment algo introducing whitespace
  diff = diff[20:, 0:-10]

  diff = cv2.medianBlur(diff, 5)
  diff = utils.threshold(diff)
  
  kernel = np.ones((5,5),np.uint8)
  diff = cv2.dilate(diff,kernel,iterations = 2)
  diff = cv2.medianBlur(diff, 5)
  diff = cv2.medianBlur(diff, 5)
  
  if utils.__DEBUG__:
    utils.show_image(diff)

  contour_centers, has_error = get_circle_centers(diff)

  circled_responses = None
  if not has_error:
    circled_responses = centers_to_responses(contour_centers, response_codes, 
                                             aligned_response_codes)
  print ("Error? ", has_error)

  return circled_responses, has_error


def error_check_responses(responses):
  print('responses:')
  print(responses)
  return False


def create_error_image(page, barcode_coords, first_response_coords):
  full_response_bounding_box = get_response_including_barcode(barcode_coords, first_response_coords)
  error_image = utils.get_roi(page, list(full_response_bounding_box))
  return(error_image)


def save_responses(responses, voter_id, dict_writer):
  question_to_responses = {}
  for response in responses:
    key = "question_%s" % response.question_number
    if key not in question_to_responses:
      question_to_responses[key] = []
    question_to_responses[key].append(response.value)

  question_to_responses['voter_id'] = voter_id
  dict_writer.writerow(question_to_responses)


def generate_error_pages(error_images, skipped_pages, list_id):
  # set page dimensions
  width_inches = 11
  height_inches = 8.5
  dpi = 300
  margin_inches = .25
  width_pixels = int(width_inches * dpi)
  height_pixels = int(height_inches * dpi)
  margin_pixels = int(margin_inches * dpi)

  # calculate the number of error images that can fit on a page
  error_image_height, error_image_width, channels = error_images[0].shape
  num_images_per_page = math.floor((height_pixels - 2*margin_pixels) / error_image_height)

  # init error pages array
  error_pages = skipped_pages
  images_on_page = 0
  page = None

  for i, error_image in enumerate(error_images):

    # Create new pages as necessary
    if i % num_images_per_page == 0:

      # save the previous page to the error_pages array
      if page:
        error_pages.append(page)

      # generate a new page and reset the images on page counter
      page = np.ones((height_pixels,width_pixels,3), np.uint8)*255
      images_on_page = 0

    # add images to the page
    error_image_height, error_image_width = error_image.shape[:2]
    start_x = margin_pixels
    end_x = start_x + error_image_width
    start_y = images_on_page * error_image_height + margin_pixels
    end_y = start_y + error_image_height
    page[start_y:end_y,start_x:end_x] = error_image

    # increment the images on page counter
    images_on_page += 1 

  # add the last page to the array
  error_pages.append(page)

  # save out a pdf
  save_error_pages(error_pages, list_id)


def save_error_pages(error_pages, list_id):
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


def main():
  args = parse_args()
  check_files_exist(args['list_id'])
  list_dir = utils.get_list_dir(args["list_id"])

  ref_bounding_boxes = utils.load_ref_boxes(args["list_id"])
  ref_page = utils.load_page(list_dir + utils.CLEAN_IMAGE_FILENAME)

  # Prep the output file
  outfile = open("{}/results_{}.csv".format(list_dir, args['list_id']), mode='w+')
  dict_writer = csv.DictWriter(outfile, fieldnames=["voter_id", "question_1", "question_2", "question_3", "question_4"])
  dict_writer.writeheader()

  unreadable_responses_for_human = []
  skipped_pages = []
  num_pages = len(os.listdir("{}/{}".format(list_dir, utils.WALKLIST_DIR)))
  for page_number in range(args['start_page'], num_pages):
    page = utils.load_page(utils.get_page_filename(args['list_id'], page_number), args["rotate_dir"])
    response_codes = utils.load_response_codes(args['list_id'])

    # align page
    raw_page = page.copy()
    page, transform = utils.alignImages(page, ref_page)
    utils.show_image(page)

    # confirm page has the correct list_id
    page_list_id = utils.get_list_id_from_page(page, ref_bounding_boxes["list_id"])
    if page_list_id != args['list_id']:
      print('Error: Page {} has ID {}, but active ID is {}. Page {} has been skipped.'.format(page_number, page_list_id, args['list_id'], page_number))
      skipped_pages.append(raw_page)
      continue

    # find the barcodes in the image and decode each of the barcodes
    # Barcode scanner needs the unthresholded image.
    barcodes = pyzbar.decode(page)
    if len(barcodes) == 0:
      print('Error: Cannot find barcodes. Page {} has been skipped.'.format(page_number))
      skipped_pages.append(raw_page)
      continue

    # loop over the detected barcodes
    for barcode in barcodes:
      (barcode_coords, voter_id) = extract_barcode_info(barcode, page)

      if utils.__DEBUG__:
        cv2.rectangle(page, barcode_coords[0], barcode_coords[1], (255, 0, 255), 3)
        utils.show_image(page)

      # Get the corresponding response codes region
      response_bounding_box = get_response_for_barcode(barcode_coords, 
                                ref_bounding_boxes["response_codes"])

      # Figure out which ones are circled
      ref_response_codes = utils.load_page(list_dir + utils.RESPONSE_CODES_IMAGE_FILENAME)
      circled_responses, has_error = get_circled_responses(response_bounding_box, response_codes, page, ref_response_codes)
      has_error = has_error or error_check_responses(circled_responses)

      if has_error:
        error_image = create_error_image(page, barcode_coords, ref_bounding_boxes["response_codes"])
        unreadable_responses_for_human.append(error_image)
      else:
        save_responses(circled_responses, voter_id, dict_writer)

    utils.show_image(page)
  outfile.close()

  generate_error_pages(unreadable_responses_for_human, skipped_pages, args['list_id'])


if __name__ == '__main__':
  main()


