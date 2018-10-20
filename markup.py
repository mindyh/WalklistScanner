"""One-off script for marking up pages."""
import utils
import argparse
from pdf2image import convert_from_path
import json


def parse_args():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image",
    help="path to the image to markup.")
  ap.add_argument("--rotate_dir", default=None, 
    help="CW or CCW, rotate the page 90 degrees in that direction.")
  return vars(ap.parse_args())


def save_points(refPts, point_name="new_point"):
  points = {}

  # Add and save new ones
  points[point_name] = []
  for point in refPts:
    points[point_name].append(point)

  with open(utils.REFPTS_FILENAME, "w+") as f:
    json.dump(points, f)

  print ("Saved to %s." % utils.REFPTS_FILENAME)


def main():
  args = parse_args()

  box = utils.markup_image(utils.load_page(args["image"], args["rotate_dir"]))
  save_points(box)

if __name__ == '__main__':
  main()


