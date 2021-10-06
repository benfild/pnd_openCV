import cv2
import pytesseract
from scipy import ndimage

# read the image file and convert in grayscale Image
img = cv2.imread('carsImages/WhatsApp Image 2020-10-15 at 11.23.42.jpeg')
imgGs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGS = cv2.GaussianBlur(imgGs, (5, 5), 0)

# Edges detection
imgEdged = cv2.Canny(imgGs, 63, 200)
cv2.imshow("Edged Image", imgEdged)
cv2.waitKey(0)


# find contours based on edges
contours, new = cv2.findContours(imgEdged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

# initialize licence plate contour and x, y coordinates
contour_with_LP = None
license_plate = None
x = None
y = None
w = None
h = None