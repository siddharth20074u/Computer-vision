import cv2
import numpy as np

input = cv2.imread('/users/siddharthsmac/downloads/elephant.jpg')

cv2.imshow('Hello World', input)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(input.shape)

print('Height of image:', int(input.shape[0]), 'pixels')
print('Width of image:', int(input.shape[1]), 'pixels')

cv2.imwrite('output.png', input)
