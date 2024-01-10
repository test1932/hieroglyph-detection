import pygame
import cv2
import sys
import os
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml.detectStructure import *

# Notes start of 1-2 = 18, end = 389

# b - bracket
# h - hieroglyph
# m - multipleLetters
# l - lowercase
# u - uppercase
# s - skip
# p - non-bracket punctuation
# n - number

HEIGHT = 800
WIDTH = 600

def getCrop(image, cropBounds, contours, move = True):
    x1,y1,x2,y2 = cropBounds
    maxheight, maxwidth = 60, 50
    
    arr = image[int(y1):int(y2) + 1, int(x1):int(x2) + 1]
    height, width = arr.shape
    
    retImage = np.zeros((maxheight,maxwidth))
    if width > maxwidth or height > maxheight:
        return retImage
    
    minX = int(cv2.boundingRect(min(contours, key = lambda x:cv2.boundingRect(x)[0]))[0])
    minY = int(cv2.boundingRect(min(contours, key = lambda x:cv2.boundingRect(x)[1]))[1])
    if move:
        for cont in contours:
            for point in cont:
                point[:,0] -= minX
                point[:,1] -= minY
    cv2.drawContours(retImage, contours, -1, color = (255,255,255), thickness = cv2.FILLED)
    arr = np.pad(arr, ((0,maxheight - height),(0, maxwidth - width)), mode='constant', constant_values=0)
    arr = arr.astype(np.uint8)
    retImage = retImage.astype(np.uint8)
    return cv2.bitwise_and(arr, retImage)

def runPygame(directory, datasetFile, extremes):
    pygame.init()
    fileNames = list(os.listdir(directory))
    breakVar = True
    
    datasetFile = open(datasetFile, "a")
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    shapes, curImage, image = getBoundsSequential(fileNames, directory, extremes)
    curPos = 0#len(shapes) - 1
    
    loadedImage = pygame.image.load(f'{directory}\\{curImage}')
    # imageWidth, imageWidth = 
    # loadedImage = pygame.transform.scale(loadedImage, (WIDTH * 5, HEIGHT * 5))
    
    while breakVar:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                breakVar = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    breakVar = False
                if event.key == pygame.K_0:
                    curPos += 10
                if event.key in [pygame.K_p, pygame.K_h, pygame.K_l, pygame.K_u, pygame.K_m, pygame.K_n, pygame.K_b]:
                    # edit
                    imageCrop = getCrop(image, shapes[curPos][0].bounds, shapes[curPos][1])
                    cv2.imshow("a", imageCrop)
                    datasetFile.write(f'\n{event.unicode},{list(map(int,imageCrop.flatten().tolist()))}')
                if event.key in [pygame.K_p, pygame.K_h, pygame.K_l, pygame.K_u, pygame.K_m, pygame.K_n, pygame.K_b, pygame.K_s]:
                    curPos += 1
                if curPos >= len(shapes):
                    shapes, curImage, image = getBoundsSequential(fileNames, directory, extremes)
                    loadedImage = pygame.image.load(f'{directory}\\{curImage}')
                    # loadedImage = pygame.transform.scale(loadedImage, (WIDTH * 5, HEIGHT * 5))
                    curPos = 0#len(shapes) - 1
                if event.key == pygame.K_x:
                    datasetFile.write(",x")
        
        screen.fill((255,255,255))
        width = shapes[curPos][0].bounds[2] - shapes[curPos][0].bounds[0]
        height = shapes[curPos][0].bounds[3] - shapes[curPos][0].bounds[1]
        try:
            screen.blit(loadedImage, (200,200), (shapes[curPos][0].bounds[0] - 100, shapes[curPos][0].bounds[1] - 100,\
                width + 200, height + 200))
            pygame.draw.rect(screen, (255,0,0), (300 - 5, 300 - 5, width + 10, height + 10), 2)
        except:
            screen.blit(loadedImage, (200,200), (shapes[curPos][0].bounds[0] - 50, shapes[curPos][0].bounds[1] - 50,\
                width + 100, height + 100))
            pygame.draw.rect(screen, (255,0,0), (250 - 5, 250 - 5, width + 10, height + 10), 2)
        pygame.display.flip()
    
    datasetFile.close()
    
def getBoundsSequential(fileNames, directory, ends):
    lines, image = analyseImage(f'{directory}\\{(x := fileNames[random.randint(ends[0], ends[1])])}', thresh = 180, brackets = False)
    bounds = []
    for line in lines:
        for shape in sorted(line.shapes, key = lambda x:x[0].bounds[0]):
            # if shape[0].bounds[3] - shape[0].bounds[1] < 25:
            #     continue
            bounds.append(shape)
    if len(bounds) == 0:
        bounds, x, image = getBoundsSequential(fileNames, directory, ends)
    return bounds, x, image
        
def getBounds(fileNames, directory, ends):
    lines, image = analyseImage(f'{directory}\\{(x := fileNames[random.randint(ends[0], ends[1])])}', thresh = 180, brackets = False)
    bounds = []
    for _ in range(500):
        if len(lines) == 0:
            break
        lineNo = random.randint(0, len(lines) - 1)
        if len(lines[lineNo].shapes) == 0:
            continue
        charNo = random.randint(0, len(lines[lineNo].shapes) - 1)
        if lines[lineNo].shapes[charNo][0].bounds[3] - lines[lineNo].shapes[charNo][0].bounds[1] < 25:
            continue
        bounds.append((lines[lineNo].shapes[charNo][0], lines[lineNo].shapes[charNo][1]))
    
    if len(bounds) == 0:
        bounds, x, image = getBounds(fileNames, directory, ends)
    return bounds, x, image

def main(directory, datasetFile, extremes):
    runPygame(directory, datasetFile, extremes)

if __name__ == '__main__':
    folder = sys.argv[1]
    dataset = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    assert start <= end
    main(folder, dataset, (start, end))