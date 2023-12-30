import os
import sys
import cv2
import pygame
import shapely
import numpy as np

pygame.init()
HEIGHT = 1000

#TODO 
# something to do with differences between lines on the page
# - if no small differences, ignore page
# save image of bracket and compare each character
# wipe area hieros

# ahhh 12, 19,21,32,33,34,36,255

# get bounds of contour 'cont'
def boundingBox(cont):
    minX, maxX, minY, maxY = None, 0, None, 0
    for [[x,y]] in cont:
        if minX == None or x < minX:
            minX = x
        elif x > maxX:
            maxX = x
        if minY == None or y < minY:
            minY = y
        elif y > maxY:
            maxY = y
    return shapely.Polygon([(minX,minY),(minX,maxY),(maxX,maxY),(maxX,minY)])

def cropImage(image, shape):
    x,y,xw,yh = shape.bounds
    return image[int(y):int(yh),int(x):int(xw)]

def getImage(filePath):
    frame = cv2.imread(filePath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    conts, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite("thing.png", thresh)
    conts = list(filter(lambda x: cv2.contourArea(x) > 5, conts))
    return thresh, frame, conts

def findLines(image):
    lines = []
    stage = 0 # 0 = in some zeros, 1 = in line not yet bottom, 2 = found bottom, going to end
    lastRow = 0
    height = 0
    lastLineRow = None
    prevBlackPos = [None,None]
    
    for i in range(len(image)):
        row = sum(image[i])
        whereBlack = np.where(image[i] != 0)
        
        wasBlack = False
        if whereBlack[0].size != 0:
            wasBlack = True
            firstBlack = whereBlack[0][0]
            lastBlack = whereBlack[0][-1]
        
        # print(row, stage)
        if row == 0 and stage != 1:
            lastRow = row
            stage = 0
            height = 0
            continue
        if stage == 2 and ((row > lastLineRow * 3 or row > lastRow * 3)): # line length changed
            stage = 1
            continue
        if stage == 0:
            stage = 1
            lastRow = row
            continue
        
        if row < 0.5 * lastRow and height > 15 and (stage != 2 or lastRow > 10000):
            height = 0
            lastLineRow = row
            # print("line")
            lines.append(i)
            stage = 2
        
        lastRow = row
        if wasBlack:
            prevBlackPos = [firstBlack, lastBlack]
        height += 1
    return lines

def lineIndex(lines, poly):
    low, high = 0, len(lines) - 1
    bound = poly.bounds

    while low <= high:
        mid = (low + high) // 2

        if lines[mid] == bound[1]:
            return mid
        elif lines[mid] < bound[1]:
            low = mid + 1
        else:
            high = mid - 1
            
    return low
    
def isMaybeAbove(aboveLineY, lineY, bound):
    topCond = abs(aboveLineY - bound[1]) < 0.4 * abs(lineY - bound[1])
    meanY = (bound[3] + bound[1]) / 2
    meanCond = abs(aboveLineY - meanY) < 0.5 * abs(lineY - meanY)
    return topCond and meanCond

# checks if there is a relatively tall character above the bound
def isCharAbove(bound, lineBounds, aboveBoundary = True):
    boundBounds = bound.bounds
    width = boundBounds[2] - boundBounds[0]
    meanX = (boundBounds[0] + boundBounds[2]) / 2
    for (aboveBound, contours) in lineBounds:
        aboveBoundBounds = aboveBound.bounds
        height = aboveBoundBounds[3] - aboveBoundBounds[1]
        if aboveBoundBounds[0] <= meanX <= aboveBoundBounds[2] and ((width < 5 or height > 20) or aboveBoundary):
            return True
    return False

def toLines(bounds, lines):
    bounds.sort(key = lambda x: x[0].bounds[1])
    lineBounds = [[] for i in lines]
    for (bound, contours) in bounds:
        low = lineIndex(lines, bound)
        if low == 0 or low == len(lines):
            lineBounds[0].append((bound, contours))
            continue
        
        # if above threshold
        boundary = isMaybeAbove(lines[low - 1], lines[low], bound.bounds)
        boundBounds = bound.bounds
        
        if not boundary:
            lineBounds[low].append((bound, contours))
            continue
        
        #on boundary for being line above
        if isCharAbove(bound, lineBounds[low - 1], ((boundBounds[1] - lines[low - 1]) < 5)):
            lineBounds[low - 1].append((bound, contours))
        else:
            lineBounds[low].append((bound, contours))
        
    return lineBounds

def doesCombine(first, second):
    smaller, larger = (first, second) if first[2] - first[0] < \
        second[2] - second[0] else (second, first)
    return larger[0] < (smaller[0] + smaller[2]) / 2 < larger[2]

def combineOnLine(bounds):
    i = 0
    while i < len(bounds):
        j = 0
        while j < len(bounds):
            if i == j:
                j += 1
                continue
            first = bounds[i]
            second = bounds[j]
            if doesCombine(first[0].bounds,second[0].bounds):
                bounds.remove(first)
                bounds.remove(second)
                bounds.append((first[0].union(second[0]),first[1] + second[1]))
                if j > i:
                    i -= 1
                else:
                    i -= 2
                break
            j += 1
        i += 1
    return list(filter(lambda x: x[0].area > 10, bounds))

def combineHieros(isHeiros, shapes):
    for i in range(len(isHeiros)):
        next = isHeiros[i + 1] if i != len(isHeiros) - 1 else 0
        prev = isHeiros[i - 1] if i != 0 else 0
        if ((next in [0.9, 1] or prev in [1]) and isHeiros[i] in [0.5, 0.9]) or\
                (next == 0.5 and isHeiros[i] == 0.9):
            isHeiros[i] = 1
        # case where two maybes next to each other
        elif next == 0.5 and isHeiros[i] == 0.5:
            if shapes[i + 1][0].distance(shapes[i][0]) < 20:
                isHeiros[i] = 1
                

def removeLongChains(isHeiros):
    if len(isHeiros) < 3:
        return isHeiros
    start = 0
    end = 3
    while end <= len(isHeiros):
        if all(list(map(lambda x: x == 0.1 or x == 0.2, isHeiros[start:end]))):
            i = start
            while isHeiros[i] == 0.1 or isHeiros[i] == 0.2:
                isHeiros[i] = 0
                i += 1
        start += 1
        end += 1
    return isHeiros
            
def expandHieroBlock(shapes, isHeiros):
    copy = isHeiros[:]
    
    changes = True
    while changes:
        changes = False
        for i in range(len(shapes)):
            next = isHeiros[i + 1] if i != len(isHeiros) - 1 else 0
            prev = isHeiros[i - 1] if i != 0 else 0
            
            distanceToNext = shapes[i + 1][0].distance(shapes[i][0]) if next != 0 else 50
            distanceToPrev = shapes[i][0].distance(shapes[i - 1][0]) if prev != 0 else 50
            if isHeiros[i] == 0:
                if next in [0.1, 0.2, 0.9, 1] and distanceToNext < 8:
                    isHeiros[i] = 0.1
                    changes = True
                elif prev in [0.1, 0.2, 0.9, 1] and distanceToPrev < 8:
                    isHeiros[i] = 0.1
                    changes = True
            elif isHeiros[i] == 0.5:
                if next in [0.1, 0.2, 0.9, 1] and distanceToNext < 12:
                    isHeiros[i] = 0.2
                    changes = True
                elif prev in [0.1, 0.2, 0.9, 1] and distanceToPrev < 12:
                    isHeiros[i] = 0.2
                    changes = True
    isHeiros = removeLongChains(isHeiros)
    return isHeiros != copy

def getHieroglyphs(shapes, lineY):
    # 0   = not a hieroglyph                        - not a hieroglyph if not removed
    # 0.1 = got chained from a 0                    - hieroglyph if not removed
    # 0.2 = got chained from a 0.5                  - hieroglyph if not removed
    # 0.5 = might be a hieroglyph if not isolated   - not a hieroglyph if not removed
    # 0.9 = almost certainly an isolated hieroglyph - hieroglyph if not removed
    isHieros = []
    shapes.sort(key = lambda x: x[0].bounds[0])
    for i in range(len(shapes)):
        bounds = shapes[i][0].bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        if height > 30:
            if height > 50:
                isHieros.append(0)
            elif width > 10:
                isHieros.append(0.9)
            else:
                isHieros.append(0.5)
        elif bounds[3] < lineY - 5:
            isHieros.append(0.5)
        elif height > 25 and width > 20:
            isHieros.append(0.5)
        else:
            isHieros.append(0)
    
    changes = True
    while changes:
        combineHieros(isHieros, shapes)
        changes = expandHieroBlock(shapes, isHieros)
    return [shapes[i] for i in range(len(isHieros)) if isHieros[i] in [0.1,0.2,0.9,1]]
    # return [shapes[i] for i in range(len(isHieros)) if isHieros[i] in [0.1,0.2,0.5,0.9,1]]

def display(fileName, lines, characters):
    image = pygame.image.load(fileName)
    
    ratio = image.get_height() / HEIGHT
    scaled = pygame.transform.scale(image, (image.get_width() / ratio, HEIGHT))
    screen = pygame.display.set_mode((image.get_width() / ratio, HEIGHT))
    running = True
    
    colours = [
        (255,0,0),
        (128,128,0),
        (0,255,0),
        (0,255,255),
        (0,0,255),
        (128,128,128)
    ]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.blit(scaled,(0,0))
        
        for i in range(len(characters)):
            for shape, contours in characters[i]:
                (x1,y1,x2,y2) = shape.bounds
                pygame.draw.rect(screen,colours[i%len(colours)],(x1/ratio,y1/ratio,(x2-x1)//ratio + 1,(y2-y1)//ratio + 1))
                
        for i in lines:
            pygame.draw.line(screen, (0,255,0), (0,i/ratio), (scaled.get_width(),i/ratio))
        
        pygame.display.flip()
        
def whiteOutHieros(image, hieros):
    for line in hieros:
        for poly, contours in line:
            cv2.drawContours(image, contours, -1, color = (255,255,255), thickness = cv2.FILLED)
            cv2.drawContours(image, contours, -1, color = (255,255,255), thickness = 3)
    cv2.imwrite("outthing.png", image)
    
def filterSmallPunctuation(shapes):
    withoutSmallPunctuation = []
    for shape in shapes:
        area = shape[0].area
        bounds = shape[0].bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if (not (height < 3 * width and area < 50)):
            withoutSmallPunctuation.append(shape)
    return withoutSmallPunctuation

def removeBrackets(image, allbounds):
    # remove bracket - flipped, largeBracket - flipped, squareBracket - flipped
    bracketImage = cv2.cvtColor(cv2.imread("bracket.png"), cv2.COLOR_BGR2GRAY)
    largeBracketImage = cv2.cvtColor(cv2.imread("largeBracket.png"), cv2.COLOR_BGR2GRAY)
    squareBracket = cv2.cvtColor(cv2.imread("squareBracket.png"), cv2.COLOR_BGR2GRAY)
    bracketImgs = [bracketImage, squareBracket, largeBracketImage]
    
    newShapes = []
    brackets = []
    
    for i,(shape, contour) in enumerate(allbounds):
        bounds = shape.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if height < 25 or width > 10:
            newShapes.append((shape, contour))
            continue
        
        characterImage = cropImage(image, shape).astype(int)
        height, width = characterImage.shape
        
        a = False
        
        # print(i)
        # cv2.imshow("a",characterImage)
        # cv2.waitKey(0)
        
        # if i == 1459:
        #     cv2.imwrite("largeBracket.png", characterImage)
        for img in bracketImgs:
            brH, brW = img.shape
            if abs(brH - height) > 3 or abs(brW - width) > 3:
                continue
            
            resizedbracket = cv2.resize(img, (width, height)).astype(int)
            flipped_resizedbracket = cv2.flip(resizedbracket, 1).astype(int)
            
            cost1 = sum(sum(abs(resizedbracket - characterImage)))
            cost2 = sum(sum(abs(flipped_resizedbracket - characterImage)))
            # print(flipped_resizedbracket - characterImage)
            if cost1 / (height * width) < 75 or cost2 / (height * width) < 75:
                a = True
                break
            
        if not a:
            newShapes.append((shape, contour))
        else:
            brackets.append((shape, contour))
    return newShapes, brackets

def main(fileName):
    print(f'now analysing {fileName}')
    image, colour, contours = getImage(fileName)
    lines = findLines(image)
    bounds = list(map(boundingBox, contours))
    bounds = list(zip(bounds, [[contour] for contour in contours]))
    
    bounds, brackets = removeBrackets(image, bounds)
    
    bounds = toLines(bounds, lines)[1:]
    bounds = list(map(combineOnLine, bounds))
    hieroglyphs = [getHieroglyphs(bounds[i], lines[i + 1]) for i in range(len(bounds))]
    hieroglyphs = [filterSmallPunctuation(line) for line in hieroglyphs]
    whiteOutHieros(colour, hieroglyphs)
    # hieroglyphs = [brackets]
    # hieroglyphs = bounds
    # hieroglyphs = []
    display(fileName, lines, hieroglyphs)

if __name__ == '__main__':
    if sys.argv[1] == 'd':
        for fileName in os.listdir(sys.argv[2]):
            main(f'{sys.argv[2]}\\{fileName}')
    else:
        main(sys.argv[1])