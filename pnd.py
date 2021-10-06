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