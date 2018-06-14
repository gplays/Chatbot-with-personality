#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser(description='Process input and output dir')
parser.add_argument('-i', help='input directory')
parser.add_argument('-o', help='output directory')

args = parser.parse_args()


from utils.data_utils import build_data
build_data('Data', 'nmt_data')
