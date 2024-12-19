import pygame
import sys
import numpy as np
import cv2
from tensorflow import keras

# Loading the pre-trained model
MODEL = keras.models.load_model('Handwritten_digit_Classification_model.keras')

width = 640
height = 480 

pygame.init() 
DISPLAYSURF = pygame.display.set_mode((width, height))
pygame.display.set_caption("Digit Board")

# Constants
BOUNDRYINC = 5
IMAGESAVE = False
PREDICT = True
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
FONT = pygame.font.Font(None, 18)


LABELS = {
    0: "Zero", 1: "One",
    2: "Two", 3: "Three",
    4: "Four", 5: "Five",
    6: "Six", 7: "Seven",
    8: "Eight", 9: "Nine"
}

# Variables
iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == pygame.MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == pygame.MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x = max(number_xcord[0] - BOUNDRYINC, 0)
            rect_max_x = min(width, number_xcord[-1] + BOUNDRYINC)
            rect_min_y = max(number_ycord[0] - BOUNDRYINC, 0)
            rect_max_y = min(height, number_ycord[-1] + BOUNDRYINC)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite(f"image_{image_cnt}.png", img_arr)
                image_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28))
                image = image.reshape(1, 28, 28, 1) / 255.0 
                label = LABELS[np.argmax(MODEL.predict(image))]

                textSurface = FONT.render(label, True, RED)
                DISPLAYSURF.blit(textSurface, (rect_min_x, rect_max_y))

        if event.type == pygame.KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
