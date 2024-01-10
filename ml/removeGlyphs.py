import os
import sys
import cv2
import pygame
import shapely
import numpy as np
import pickle
import random

from detectStructure import *
from makeDataset import getCrop

pygame.init()
HEIGHT = 1000

#87,113

def display(fileName, lines, additionalCharacters):
    image = pygame.image.load(fileName)
    
    ratio = image.get_height() / HEIGHT
    scaled = pygame.transform.scale(image, (image.get_width() / ratio, HEIGHT))
    screen = pygame.display.set_mode((image.get_width() / ratio, HEIGHT))
    running = True
    
    colours = [tuple([random.randint(0,225) for i in range(3)]) for i in range(50)]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.blit(scaled,(0,0))
        
        for i,line in enumerate(additionalCharacters):
            for shape,contours in line:
                (x1,y1,x2,y2) = shape.bounds
                pygame.draw.rect(screen,colours[i%len(colours)],(x1/ratio,y1/ratio,(x2-x1)//ratio + 1,(y2-y1)//ratio + 1), 2)
                
        for j,line in enumerate(lines):
            for i, (shape, conts) in enumerate(line.shapes):
                # print(len(line.weights), len(line.shapes))
                if line.weights[i] not in [0.1,0.2,0.9,1]:
                    continue
                (x1,y1,x2,y2) = shape.bounds
                pygame.draw.rect(screen,colours[j%len(colours)],(x1/ratio,y1/ratio,(x2-x1)//ratio + 1,(y2-y1)//ratio + 1))
            pygame.draw.line(screen, (0,255,0), (line.edges[0] / ratio,line.bottom/ratio), (line.edges[1] / ratio,line.bottom/ratio))
            if line.top == None:
                continue
            pygame.draw.line(screen, (255,0,0), (line.edges[0] / ratio,line.top/ratio), (line.edges[1] / ratio,line.top/ratio))
        
        pygame.display.flip()
        
def var(things):
    sigmaX = sum(things)
    sumSquares = sum([x**2 for x in things])
    return (sumSquares / len(things)) - (mean(things) ** 2)

def mean(things):
    if len(things) == 0:
        return 0
    return sum(things) / len(things)
        
def newGetLines(bounds):
    bounds = sorted(bounds, key = lambda x: x[0].bounds[0])
    lines = []
    edges = []
    i = 0
    j = 0
    while i < len(bounds):
        print(j, len(bounds));j+=1
        line = [(bounds[i][0], [bounds[i][1]])]
        centres = [(bounds[i][0].bounds[3] + bounds[i][0].bounds[1]) / 2]
        maxX = bounds[i][0].bounds[2]
        minX = bounds[i][0].bounds[0]
        maxHeight, maxDepth = 5, 5
        indexes = [i]
        
        for j,(poly,contours) in enumerate(bounds[1:]):
            if poly.bounds[0] - maxX > 100 or abs(poly.bounds[3] - centres[-1]) > 100:
                continue
            bound = poly.bounds
            centre = mean(centres[-5:])
            diff = -5 if bound[1] < centre < bound[3] else abs(centre - (bound[1] + 0.5 * bound[3]) / 1.5)
            
            v = maxDepth if bound[3] > centre else maxHeight
            # v = 5
            if diff + 10 <= v:
                maxHeight = max(centre - bound[1], maxHeight)
                maxDepth = max(bound[3] - centre, maxDepth)
                indexes.append(j + 1)
                line.append((poly,[contours]))
                centres.append((bound[3] + bound[1]) / 2)
                maxX = max(maxX, bound[2])
            
        for i in indexes[::-1]:
            del bounds[i]
            
        lines.append(line)
        edges.append((minX,maxX))
    return lines, edges

def isSequential(line1, line2):
    if not (line1[0] > line2[1] or line2[0] > line1[1]):
        return False
    if min(abs(line1[0] - line2[1]), abs(line1[1] - line2[0])) < 50:
        return True
    return False

def getXrange(minX, maxX, line):
    things = []
    lastI = 0
    for i, (char, contours) in enumerate(line):
        if char.bounds[0] > minX and char.bounds[2] < maxX:
            things.append(char)
            lastI = i
    return things, lastI

def combines(line1, line2):
    if len(line1) * len(line2) == 0:
        return True
    
    if abs(line1[0][0].bounds[1] - line2[0][0].bounds[1]) > 70:
        return False
    
    if len(line1) > len(line2):
        firstLine, secondLine = line1, line2
    else:
        firstLine, secondLine = line2, line1

    after = isSequential((line1[0][0].bounds[0], line1[-1][0].bounds[2]),\
        (line2[0][0].bounds[0], line2[-1][0].bounds[2]))
    
    c = shapely.MultiPolygon(map(lambda x:x[0], firstLine)).bounds
    count = 0
    minY,maxY = None,None
    
    firstbounds = secondLine[0][0].bounds
    partialLine, i = getXrange(firstbounds[0] - 100, firstbounds[2] + 100, firstLine)
    
    for char,contours in secondLine:
        while len(partialLine) > 0 and partialLine[0].bounds[0] < char.bounds[0] - 100:
            del partialLine[0]
            
        if len(partialLine) == 0:
            partialLine, i = getXrange(char.bounds[0] - 100, char.bounds[2] + 100, firstLine)
        else:
            while i < len(firstLine) and firstLine[i][0].bounds[2] < char.bounds[2] + 100:
                partialLine.append(firstLine[i][0])
                i += 1
                
        if len(partialLine) == 0:
            continue
        
        a = shapely.MultiPolygon(partialLine).bounds
        lineBox1 = shapely.Polygon([(a[0],a[1]),(a[0],a[3]),(a[2],a[3]),(a[2],a[1])])
        
        if minY == None or char.bounds[1] < minY:
            minY = char.bounds[1]
        if maxY == None or char.bounds[3] > maxY:
            maxY = char.bounds[3]
        if char.intersection(lineBox1).area > 0.2 * char.area:
            count += 1
            
    if maxY and minY and max(maxY, c[3]) - min(c[1], minY) > 60:
        return False
    
    return count > 0.5 * len(secondLine)# or (after and maxY and minY and max(maxY, c[3]) - min(c[1], minY) < 40)

def combineLines(lines):
    i = 0
    while i < len(lines):
        j = 0
        while j < len(lines):
            if i == j:
                j += 1
                continue
            if combines(lines[i], lines[j]):
                print("a",i,j)
                # insert at i
                t = min(i,j)
                lines.insert(t, sorted(lines[i] + lines[j], key = lambda x:x[0].bounds[0]))
                del lines[max(i,j) + 1]
                del lines[min(i,j) + 1]
                if j < i:
                    i -= 2
                else:
                    i -= 1
                break
            j += 1
        i += 1
        
def newCombineOnLine(shapes):
    # shapes is list of (polygon, [contours]) pairs
    i = 0
    while i < len(shapes):
        j = 0
        while j < len(shapes):
            if i == j:
                j += 1
                continue
            first = shapes[i]
            second = shapes[j]
            if doesCombine(first[0].bounds, second[0].bounds):
                del shapes[max(i,j)]
                del shapes[min(i,j)]
                shapes.append((first[0].union(second[0]),first[1] + second[1]))
                if j > i:
                    i -= 1
                else:
                    i -= 2
                break
            j += 1
        i += 1

def trace(val):
    print(val)
    return val

def isMap(shapes):
    for shape in shapes:
        if shape.bounds[3] - shape.bounds[1] > 100:
            return True
    return False

def filterNonsenseBounds(shapes):
    return list(filter(lambda x:x.bounds[3] - x.bounds[1] < 50, shapes))
        
def removeWithModel(lines, image, bracketModelFname, label):
    file = open(bracketModelFname, "rb")
    model = pickle.load(file)
    file.close()
    newLines = []
    removed = []
    
    for line in lines:
        modelInputs = np.array([getCrop(image, line[i][0].bounds, line[i][1], move = True).flatten()\
            for i in range(len(line))])
        res = model.predict(modelInputs)
        removedChars = [line[i] for i in range(len(line)) if res[i] == label]
        new = [line[i] for i in range(len(line)) if res[i] != label]
        newLines.append(new)
        removed.append(removedChars)
    return newLines, removed

def containsLongThingHorizontal(contours):
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if height < 10 and width > 20:
            return True
    return False

def combineHieros(stages, shapes):
    changes = False
    for i in range(len(stages)):
        next = stages[i + 1] if i != len(stages) - 1 else 0
        prev = stages[i - 1] if i != 0 else 0
        distanceToNext = shapes[i + 1][0].distance(shapes[i][0]) if next != 0 else 50
        distanceToPrev = shapes[i][0].distance(shapes[i - 1][0]) if prev != 0 else 50
        
        if next in [0.2,0.5,0.9,1] and stages[i] in [0.5, 0.9, 1] and\
                not (next == 1 and stages[i] == 1) and distanceToNext < 12:
            stages[i] = 1
            stages[i + 1] = 1
            changes = True
        elif prev in [0.2,0.5,0.9, 1] and stages[i] in [0.5, 0.9, 1] and\
                not (prev == 1 and stages[i] == 1) and distanceToPrev < 12:
            stages[i] = 1
            stages[i - 1] = 1
            changes = True
    return changes

def removeLongChains(stages):
    if len(stages) < 3:
        return
    start = 0
    end = 3
    while end <= len(stages):
        if all(list(map(lambda x: x == 0.1 or x == 0.2, stages[start:end]))):
            i = start
            while i < len(stages) and (stages[i] == 0.1 or stages[i] == 0.2):
                stages[i] = 0
                i += 1
        start += 1
        end += 1

def expandHieroBlock(shapes, stages, isBrackets):
    copy = stages[:]
    
    changes = True
    while changes:
        changes = False
        for i in range(len(stages)):
            next = stages[i + 1] if i != len(stages) - 1 else 0
            prev = stages[i - 1] if i != 0 else 0
            
            distanceToNext = shapes[i + 1][0].distance(shapes[i][0]) if next != 0 else 50
            distanceToPrev = shapes[i][0].distance(shapes[i - 1][0]) if prev != 0 else 50
            
            if stages[i] == 0:
                if next in [0.1, 0.2, 0.9, 1] and distanceToNext < 8 and not isBrackets[i]:
                    stages[i] = 0.1
                    changes = True
                elif prev in [0.1, 0.2, 0.9, 1] and distanceToPrev < 8 and not isBrackets[i]:
                    stages[i] = 0.1
                    changes = True
            elif stages[i] == 0.5:
                if next in [0.1, 0.2, 0.9, 1] and distanceToNext < 12:
                    stages[i] = 0.2
                    changes = True
                elif prev in [0.1, 0.2, 0.9, 1] and distanceToPrev < 12:
                    stages[i] = 0.2
                    changes = True
    removeLongChains(stages)
    return stages != copy

def handleIsolated(line, stages, isBrackets):
    for i in range(len(stages)):
        if stages[i] != 0.9:
            continue
        
        if len(stages) == 0:
            break
        
        if len(stages) == 1:
            stages[i] = 1
            break
        
        distanceNext = 50 if i == len(stages) - 1 else line[i][0].distance(line[i + 1][0])
        distancePrev = 50 if i == 0 else line[i][0].distance(line[i - 1][0])
        if i == 0:
            next = stages[i + 1]
            if distanceNext > 8 or next != 0 or isBrackets[i + 1] or \
                    line[i + 1][0].bounds[3] - line[i + 1][0].bounds[1] < 10:
                stages[i] = 1
            continue
        if i == len(stages) - 1:
            prev = stages[i - 1]
            if distancePrev > 8 or prev != 0 or isBrackets[i - 1]:
                stages[i] = 1
            continue
        next,prev = stages[i + 1], stages[i - 1]
        if (distancePrev > 8 or prev != 0 or isBrackets[i - 1]) and \
                (distanceNext > 8 or next != 0 or isBrackets[i + 1] or\
                line[i + 1][0].bounds[3] - line[i + 1][0].bounds[1] < 10):
            stages[i] = 1
            continue
        stages[i] = 0.5
        
def fillGaps(stages):
    for i in range(1,len(stages)-1):
        prev = stages[i - 1]
        next = stages[i + 1]
        if prev in [0.1,0.2,0.5,1] and next in [0.1,0.2,0.5,1]:
            stages[i - 1] = 1
            stages[i] = 1
            stages[i + 1] = 1

def getHieros(line, bracketModelFname, hieroModelFname, image):
    # 0   = not a hieroglyph                        - not a hieroglyph if not removed
    # 0.1 = got chained from a 0                    - hieroglyph if not removed
    # 0.2 = got chained from a 0.5                  - hieroglyph if not removed
    # 0.5 = might be a hieroglyph if not isolated   - not a hieroglyph if not removed
    # 0.9 = almost certainly an isolated hieroglyph - hieroglyph if not directly next to non-hieros
    stages = [0 for item in line]
    line.sort(key = lambda x:x[0].bounds[0])
    
    file = open(bracketModelFname, "rb")
    bracketModel = pickle.load(file)
    file.close()
    
    file = open(hieroModelFname, "rb")
    hieroModel = pickle.load(file)
    file.close()
    
    modelInputs = np.array([getCrop(image, line[i][0].bounds, line[i][1], move = True).flatten()\
        for i in range(len(line))])
    isBrackets = [1 if item == 'b' else 0 for item in bracketModel.predict(modelInputs)]
    isHieros = [1 if item == 'h' else 0 for item in hieroModel.predict(modelInputs)]
    
    for i,(shape, contours) in enumerate(line):
        bounds = shape.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        if width > 40 and height < 30:
            stages[i] = 0
        elif height > 30:
            if width > 40:
                stages[i] = 1
            if width > 10 or isHieros[i]:
                stages[i] = 0.9 if not isBrackets[i] else 0.5
            else:
                stages[i] = 0.5 if not isBrackets[i] else 0
        elif containsLongThingHorizontal(contours):
            stages[i] = 0.9
        elif isHieros[i] and not isBrackets[i]:
            stages[i] = 0.5
        else:
            stages[i] = 0
            
    handleIsolated(line,stages,isBrackets)
    
    changes = True
    while changes:
        changes = False
        changes = combineHieros(stages, line)
        changes = expandHieroBlock(line, stages, isBrackets) or changes
        fillGaps(stages)
    
    return [line[i] for i in range(len(line)) if stages[i] in [0.1,0.2,1]]

def removeSmallHieros(line):
    return list(filter(lambda x:(x[0].bounds[3] - x[0].bounds[1]) *\
        (x[0].bounds[2] - x[0].bounds[0]) > 50, line))
    
def contourWhiteOut(image, hieros):
    colour = (255,255,255)
    for i,(_, contours) in enumerate(hieros):
        cv2.drawContours(image, contours, -1, color = colour, thickness = cv2.FILLED)
        cv2.drawContours(image, contours, -1, color = colour, thickness = 3)  
    cv2.imwrite("outthing.png", image)

def multiPolyToPoly(multiPoly):
    polys = list(map(lambda x: x.bounds,multiPoly.geoms))
    if len(polys) == 0:
        return None
    x1 = min(polys, key = lambda x: x[0])[0]
    y1 = min(polys, key = lambda x: x[1])[1]
    x2 = max(polys, key = lambda x: x[2])[2]
    y2 = max(polys, key = lambda x: x[3])[3]
    return shapely.Polygon([(x1,y1),(x1,y2),(x2,y2),(x2,y1)])

def blockWhiteOut(image, hieros):
    colour = (255,255,255)
    for i,(poly,_) in enumerate(hieros):
        if type(poly) == shapely.MultiPolygon:
            poly = multiPolyToPoly(poly)
            if poly == None:
                continue
        bounds = np.array(list(map(list,poly.exterior.coords))[1:], np.int32).reshape((-1,1,2))
        cv2.polylines(image, [bounds], isClosed = True, color = colour, thickness = 3)
        cv2.fillPoly(image, [bounds], color = colour)
    cv2.imwrite("outthing.png", image)

def main(fileName, bracketModelFname, hieroFname):
    print(f'now analysing {fileName}')
    image, colour, contours = getImage(fileName)
    imageCopy = np.copy(image)
    bounds = list(map(boundingBox, contours))
    
    if isMap(bounds):
        display(fileName, [], [])
        return
    
    bounds = filterNonsenseBounds(bounds)
    lines, edges = newGetLines(list(zip(bounds, contours)))
    lines.sort(key = lambda x:min(map(lambda y: y[0].bounds[1],x)))
    
    combineLines(lines)
    mutatingMap(newCombineOnLine, lines)
    lines = [getHieros(line, bracketModelFname, hieroFname, image) for line in lines]
    lines = list(map(removeSmallHieros, lines))
    
    blockWhiteOut(colour, [j for row in lines for j in row])
    
    additional = lines
    lines = []
    display(fileName, lines, additional)

if __name__ == '__main__':
    if sys.argv[1] == 'd':
        count = int(sys.argv[3])
        for fileName in os.listdir(sys.argv[2])[count - 1:]:
            main(f'{sys.argv[2]}\\{fileName}', sys.argv[4], sys.argv[5])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])