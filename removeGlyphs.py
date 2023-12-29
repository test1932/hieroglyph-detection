import os
import sys
import cv2
import pygame

pygame.init()
HEIGHT = 1000

#TODO 
# copy the black tiny bits and put them on the final image
# shapely multipolygon distance
# reduce chaining distance

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
    return ((minX,minY),(maxX,maxY))

def getImage(filePath):
    frame = cv2.imread(filePath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    conts, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite("thing.png", thresh)
    conts = list(filter(lambda x: cv2.contourArea(x) > 10, conts))
    return thresh, conts

def findLines(image):
    lines = []
    stage = 0 # 0 = in some zeros, 1 = in line not yet bottom, 2 = found bottom, going to end
    lastRow = 0
    height = 0
    lastLineRow = None
    
    for i in range(len(image)):
        row = sum(image[i])
        # print(row, stage)
        if row == 0 and stage != 1:
            lastRow = row
            stage = 0
            height = 0
            continue
        if stage == 2 and (row > lastLineRow * 3 or row > lastRow * 3):
            stage = 1
            continue
        if stage == 2:
            lastRow = row
            continue
        if stage == 0:
            stage = 1
            lastRow = row
            continue
        
        if row < 0.5 * lastRow and height > 10:
            height = 0
            lastLineRow = row
            # print("line")
            lines.append(i)
            stage = 2
        
        lastRow = row
        height += 1
    return lines

def lineIndex(lines, bound):
    low, high = 0, len(lines) - 1

    while low <= high:
        mid = (low + high) // 2

        if lines[mid] == bound[0][1]:
            return mid
        elif lines[mid] < bound[0][1]:
            low = mid + 1
        else:
            high = mid - 1
            
    if low > 0 and (low == len(lines) or isAbove(lines[low - 1], lines[low], bound)):
        return low - 1
    else:
        return low
    
def isAbove(aboveLineY, lineY, bound):
    topCond = abs(aboveLineY - bound[0][1]) < 0.4 * abs(lineY - bound[0][1])
    meanY = (bound[1][1] + bound[0][1]) / 2
    meanCond = abs(aboveLineY - meanY) < 0.5 * abs(lineY - meanY)
    return topCond and meanCond

def toLines(bounds, lines):
    lineBounds = [[] for i in lines]
    for bound in bounds:
        lineBounds[lineIndex(lines, bound)].append(bound)
    return lineBounds

def doesCombine(first, second):
    smaller, larger = (first, second) if first[1][0] - first[0][0] < \
        second[1][0] - second[0][0] else (second, first)
    return larger[0][0] < (smaller[0][0] + smaller[1][0]) / 2 < larger[1][0]

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
            if doesCombine(first,second):
                bounds.remove(first)
                bounds.remove(second)
                x1 = min(first[0][0], second[0][0])
                x2 = max(first[1][0], second[1][0])
                y1 = min(first[0][1], second[0][1])
                y2 = max(first[1][1], second[1][1])
                bounds.append(((x1,y1),(x2,y2)))
                if j > i:
                    i -= 1
                else:
                    i -= 2
                break
            j += 1
        i += 1
    return list(filter(lambda x: (x[1][1] - x[0][1]) * (x[1][0] - x[0][0]) > 10, bounds))

def combineHieros(isHeiros):
    for i in range(len(isHeiros)):
        next = isHeiros[i + 1] if i != len(isHeiros) - 1 else 0
        prev = isHeiros[i - 1] if i != 0 else 0
        if (next in [0.5, 0.9, 1] or prev in [1]) and isHeiros[i] in [0.5, 0.9]:
            isHeiros[i] = 1
            
def expandHieroBlock(bounds, isHeiros):
    changes = True
    anyChanges = False
    while changes:
        changes = False
        for i in range(len(bounds)):
            next = isHeiros[i + 1] if i != len(isHeiros) - 1 else 0
            prev = isHeiros[i - 1] if i != 0 else 0
            distanceToNext = bounds[i + 1][0][0] - bounds[i][1][0] if next != 0 else 50
            distanceToPrev = bounds[i][0][0] - bounds[i - 1][1][0] if prev != 0 else 50
            if isHeiros[i] == 0:
                if next in [0.9, 1] and distanceToNext < 10:
                    isHeiros[i] = 1
                    isHeiros[i + 1] = 1
                    changes = True
                    anyChanges = True
                elif prev in [0.9, 1] and distanceToPrev < 10:
                    isHeiros[i] = 1
                    isHeiros[i - 1] = 1
                    changes = True
                    anyChanges = True
    return anyChanges

def getHieroglyphs(bounds):
    isHieros = []
    bounds.sort(key = lambda x: x[0][0])
    for i in range(len(bounds)):
        width = bounds[i][1][0] - bounds[i][0][0]
        height = bounds[i][1][1] - bounds[i][0][1]
        
        if height > 30:
            if width > 10:
                isHieros.append(0.9)
            else:
                isHieros.append(0.5)
        else:
            isHieros.append(0)
    
    changes = True
    while changes:
        combineHieros(isHieros)
        changes = expandHieroBlock(bounds, isHieros)
        
    return [bounds[i] for i in range(len(isHieros)) if isHieros[i] in [0.9,1]]

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
            for (x1,y1),(x2,y2) in characters[i]:
                pygame.draw.rect(screen,colours[i%len(colours)],(x1/ratio,y1/ratio,(x2-x1)//ratio + 1,(y2-y1)//ratio + 1))
                
        for i in lines:
            pygame.draw.line(screen, (0,255,0), (0,i/ratio), (scaled.get_width(),i/ratio))
        
        pygame.display.flip()

def main(fileName):
    print(f'now analysing {fileName}')
    image, contours = getImage(fileName)
    lines = findLines(image)
    bounds = toLines(list(map(boundingBox, contours)), lines)[1:]
    bounds = list(map(combineOnLine, bounds))
    hieroglyphs = [getHieroglyphs(boundsLine) for boundsLine in bounds]
    # hieroglyphs = bounds
    display(fileName, lines, hieroglyphs)

if __name__ == '__main__':
    if sys.argv[1] == 'd':
        for fileName in os.listdir(sys.argv[2]):
            main(f'{sys.argv[2]}\\{fileName}')
    else:
        main(sys.argv[1])