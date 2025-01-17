# import the necessary packages
from pyzbar import pyzbar
import argparse
import cv2
import os  


ROOT = os.path.abspath('./')


def read_barcode(image):
#     image = cv2.imread(img)

    print("Reading barcodes")
    # find the barcodes in the image and decode each of the barcodes
    barcodes = pyzbar.decode(image)
    print(barcodes)
    barcode_dict={}

    # loop over the detected barcodes
    for barcode in barcodes:
            # extract the bounding box location of the barcode and draw the
            # bounding box surrounding the barcode on the image
        #     (x, y, w, h) = barcode.rect
        #     print(x,y,w,h)
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
     
            # the barcode data is a bytes object so if we want to draw it on
            # our output image we need to convert it to a string first
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
     
            # draw the barcode data and barcode type on the image
        #     text = "{} ({})".format(barcodeData, barcodeType)
        #     cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (0, 0, 255), 2)
     
            # print the barcode type and data to the terminal and aappending data to dictionary
            barcode_dict[barcodeData] = barcodeType
            print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    
    return barcode_dict     #returning data dictionary
