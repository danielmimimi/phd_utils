import argparse
from dataset_handler.annotation_handler_hslu import AnnotationHandlerHslu

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Path to config file', default='D:\generated')
args = parser.parse_args()

AnnotationHandlerHslu()