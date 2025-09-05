import cv2
import numpy as np

# Load image
img = cv2.imread('input.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Invert colors (since text is bright)
inv = cv2.bitwise_not(gray)

# CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl1 = clahe.apply(inv)

# Sharpening kernel
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
sharp = cv2.filter2D(cl1, -1, kernel_sharpening)

# Morphological closing (dilation + erosion) with small kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
closing = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, kernel)

# Resize image (3x for better OCR detail)
resized = cv2.resize(closing, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

cv2.imwrite('preprocessed_enhanced.jpg', resized)


import easyocr

reader = easyocr.Reader(['hi'], gpu=True)
image = 'preprocessed_enhanced.jpg'
results = reader.readtext(image,
                          contrast_ths=0.4,
                          adjust_contrast=0.8,
                          text_threshold=0.35,
                          low_text=0.25,
                          decoder='wordbeamsearch',
                          paragraph=True)

import cv2
import numpy as np

img = resized
import matplotlib.pyplot as plt
"""
[[[[798, 102], [1665, 102], [1665, 396], [798, 396]], 'छड चुंबक'],
 [[[2373, 665], [3840, 665], [3840, 1571], [2373, 1571]], '६6  =/ &'],
 [[[433, 1671], [1015, 1671], [1015, 2071], [433, 2071]], '९५'],
 [[[3232, 1999], [3805, 1999], [3805, 2159], [3232, 2159]], '९&']]

"""

plt.imshow(cv2.rectangle(img, (798, 102),(1665,396), (0,255,0), 2)) # bounding box 
