import cv2
import shapely
import numpy as np
import pickle

class characterLine:
    def __init__(self, bottom, top, edges, shapes, weights) -> None:
        self.bottom = bottom
        self.top = top
        self.edges = edges
        if len(weights) != len(shapes):
            self.weights = [0]*len(shapes)
        self.shapes = shapes
        self.weights = weights
        
    def addShape(self, shape, weight = 0):
        self.shapes.append(shape)
        self.weights.append(weight)
        
    def removeShape(self, shape):
        position = self.shapes.index(shape)
        del self.shapes[position]
        del self.weights[position]
        
    def sort(self):
        all = list(zip(self.shapes, self.weights))
        all.sort(key = lambda x: x[0][0].bounds[0])
        lists = list(zip(*all))
        if len(lists) == 0:
            return
        self.shapes = list(lists[0])
        self.weights = list(lists[1])
        
    def intersects(self, other):
        boundsSelf = shapely.Polygon([
            (self.edges[0], self.bottom),
            (self.edges[1], self.bottom),
            (self.edges[1], self.top),
            (self.edges[0], self.top)])
        boundsOther = shapely.Polygon([
            (other.edges[0], other.bottom),
            (other.edges[1], other.bottom),
            (other.edges[1], other.top),
            (other.edges[0], other.top)])
        return boundsSelf.intersects(boundsOther)
    
    def intersection(self, poly):
        boundsSelf = shapely.Polygon([
            (self.edges[0], self.bottom),
            (self.edges[1], self.bottom),
            (self.edges[1], self.top),
            (self.edges[0], self.top)])
        return boundsSelf.intersection(poly)
        
    def __eq__(self, __value: object) -> bool:
        return self.bottom == __value.bottom and self.edges == __value.edges
    
def mutatingMap(func, iterable, *args):
    for item in iterable:
        func(item, *args)
        
def getCrop(image, cropBounds, contours):
    x1,y1,x2,y2 = cropBounds
    maxheight, maxwidth = 60, 50
    
    arr = image[int(y1):int(y2) + 1, int(x1):int(x2) + 1]
    height, width = arr.shape
    
    retImage = np.zeros((maxheight,maxwidth))
    if width > maxwidth or height > maxheight:
        return retImage
    
    minX = int(cv2.boundingRect(min(contours, key = lambda x:cv2.boundingRect(x)[0]))[0])
    minY = int(cv2.boundingRect(min(contours, key = lambda x:cv2.boundingRect(x)[1]))[1])
    
    for cont in contours:
        for point in cont:
            point[:,0] -= minX
            point[:,1] -= minY
    cv2.drawContours(retImage, contours, -1, color = (255,255,255), thickness = cv2.FILLED)
    arr = np.pad(arr, ((0,maxheight - height),(0, maxwidth - width)), mode='constant', constant_values=0)
    arr = arr.astype(np.uint8)
    retImage = retImage.astype(np.uint8)
    return cv2.bitwise_and(arr, retImage)
        
def removeBrackets(image, bounds):
    file = open("../ml/bracketModel.p", "rb")
    model = pickle.load(file)
    file.close()
    
    modelInputs = np.array([getCrop(image, bounds[i][0].bounds, bounds[i][1]).flatten() \
        for i in range(len(bounds))])
    res = model.predict(modelInputs)
    
    brackets = [bounds[i] for i in range(len(bounds)) if res[i] == 'b']
    others = [bounds[i] for i in range(len(bounds)) if res[i] != 'b']
    return others, brackets

def analyseImage(fileName, thresh = 180, brackets = True):
    print(f'now analysing {fileName}')
    image, colour, contours = getImage(fileName, thresh=thresh)
    imageCopy = np.copy(image)
    
    bounds = list(map(boundingBox, contours))
    bounds = list(zip(bounds, [[contour] for contour in contours]))
    if brackets:
        bounds, brackets = removeBrackets(image, bounds)
        whiteOutHieros(imageCopy, [brackets], False, False, usesWeight = False)
    
    if any([x[0].bounds[3] - x[0].bounds[1] > 80 and \
            x[0].bounds[2] - x[0].bounds[0] > 50 for x in bounds]):
        return [], image
    
    lines = findLines(imageCopy)
    lines = adjustLines(lines, weighting = 0.5)
    filterTops(lines)
    adjustTops(lines, image)
    chainAcross(lines, image)
    
    # lines.sort(key = lambda x:x.bottom)
    unmapped = toLines(bounds, lines)
    addObviousMisses(unmapped, lines)
    
    mutatingMap(combineOnLine, lines)
    # # less floaty
    postCombinationLineAdjust(lines)
    
    reallyUnmapped, additionalLines = addAdditionalLines(lines, unmapped, image)
    print(len(reallyUnmapped))
    lines = lines + additionalLines
    
    lines.sort(key = lambda x: x.bottom)
    for line in lines:
        line.sort()
    
    mutatingMap(combineOnLine, lines)
    
    for line in lines:
        line.sort()
    
    return lines, image

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

def getAngle(image):
    #https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
    newImage = image.copy() # gray copy
    blur = cv2.GaussianBlur(newImage, (9,9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle > 45:
        angle -= 90
    return -angle

def rotateImage(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation, (width, height))

def getImage(filePath, thresh = 180):
    frame = cv2.imread(filePath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # gray = rotateImage(gray, getAngle(gray))
    
    ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    conts, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite("thing.png", thresh)
    conts = list(filter(lambda x: cv2.contourArea(x) > 2, conts))
    return thresh, frame, conts

def overlap(line1, line2):
    if line1[1] < line2[0] or line2[1] < line1[0]:
        return 0
    return (line1[1] - line1[0] + line2[1] - line2[0]) - (max(line1[0], line2[0]) - min(line1[1], line2[1]))

def getFirstRun(lineWhereNotBlack):
    start = 0
    end = 0
    runs = []
    while end < (x:=lineWhereNotBlack[0].size):
        while end + 1 < x and lineWhereNotBlack[0][end + 1] - lineWhereNotBlack[0][end] < 100:
            end += 1
        runs.append((lineWhereNotBlack[0][start], lineWhereNotBlack[0][end], \
            lineWhereNotBlack[0][end] - lineWhereNotBlack[0][start]))
        start = end + 1
        end += 1
        
    # print(runs)
    
    res = 0
    # while res < len(runs) - 1 and runs[res][2] < 20:
    #     res += 1
    
    return runs[res]

def updateEdges(extremePos, lineWhereNotBlack):
    maxRun = getFirstRun(lineWhereNotBlack)
    start = maxRun[0]
    end = maxRun[1]
    if extremePos[0] == None:
        extremePos[0] = start
        extremePos[1] = end
        return
    if maxRun[2] > (extremePos[1] - extremePos[0]):
        extremePos[0] = start
        extremePos[1] = end

def findLines(image):
    lines = []
    top = None
    
    stage = 0 # 0 = in some zeros, 1 = in line not yet bottom, 2 = found bottom, going to end
    lastRow = 0
    height = 0
    blackHeight = 0
    
    whiteRun = 0
    
    lastLineRow = None
    extremePos = [None,None]
    
    rangeFunc = lambda xs: 0 if xs[0] == None else xs[1] - xs[0]
    
    for i in range(len(image)):
        row = sum(image[i])
        
        # if lastRow == 0:
        #     top = i
        # print(row, stage, whiteRun)
        if row != 0:
            res = np.where(image[i] != 0)
            if stage == 1:
                updateEdges(extremePos, res)

        if row < 1000:
            if stage != 1:
                lastRow = row
                stage = 0
                height = 0
                continue
            else:
                whiteRun += 1
        
        if row >= 1000:
            whiteRun = 0
            
        if whiteRun >= 5:
            stage = 0
            top = i
            continue
        
        elif stage == 2 and ((row > lastLineRow * 2.75 or row > lastRow * 2.75)): # line length changed
            stage = 1
            top = i
            continue
        
        elif stage == 0:
            stage = 1
            top = i
            lastRow = row
            continue
        
        if (row <= 0.5 * lastRow) and height > 18 and blackHeight > 10 and \
                (stage != 2 or lastRow > 10000) and extremePos[0] != None and \
                i - top > 10:
            lines.append(characterLine(i, top, extremePos, [], []))
            extremePos = [None,None]
            height = 0
            blackHeight = 0
            lastLineRow = row
            stage = 2
            # print("line\t\t", top)
        
        lastRow = row
        height += 1
        blackHeight += row != 0
    return lines

def lineIndex(lines, poly):
    low, high = 0, len(lines) - 1
    bound = poly.bounds

    while low <= high:
        mid = (low + high) // 2

        if lines[mid].bottom == bound[1]:
            return mid
        elif lines[mid].bottom < bound[1]:
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
        elif boundBounds[1] < aboveBoundBounds[3]:
            return True
    return False

def isOutwith(bounds, lineEdges):
    return bounds[2] < lineEdges[0] or bounds[0] > lineEdges[1]

def toLines(bounds, lines):
    bounds.sort(key = lambda x: x[0].bounds[1])
    unmapped = []
    
    for (bound, contours) in bounds:
        low = lineIndex(lines, bound)
        contourBounds = bound.bounds
        
        # if outwith both lines bounds
        if (low == len(lines) or isOutwith(contourBounds, lines[low].edges)) and\
                (low == 0 or isOutwith(contourBounds, lines[low - 1].edges)):
            unmapped.append((bound, contours))
            continue
        
        # # if outwith bounds of low
        if low != len(lines) and isOutwith(contourBounds, lines[low].edges) and \
                contourBounds[1] <= lines[low - 1].bottom + 3:
            unmapped.append((bound, contours))
            continue
        
        # if outwith bounds of low - 1
        if low != 0 and isOutwith(contourBounds, lines[low - 1].edges):
            unmapped.append((bound, contours))
            continue
        
        # if at low = first line or intersecting low
        if low != len(lines) and (contourBounds[3] >= lines[low].bottom - 2):
            lines[low].addShape((bound, contours))
            continue
        
        if low == 0:
            if lines[low].bottom - contourBounds[3] < 10:
                lines[low].addShape((bound, contours))
                continue
            else:
                unmapped.append((bound, contours))
                continue
        
        # if low does not exist or top is close enough to low - 1 
        if (low == len(lines) and contourBounds[1] - lines[low - 1].bottom < 10) or \
                contourBounds[1] <= lines[low - 1].bottom + 3:
            lines[low - 1].addShape((bound, contours))
            continue
        
        if low == len(lines):
            unmapped.append((bound, contours))
            continue
        
        # if bound is below top of line
        if lines[low].top != None and bound.bounds[1] >= lines[low].top:
            lines[low].addShape((bound,contours))

        else:
            unmapped.append((bound, contours))
    
    return unmapped

def doesCombine(first, second):
    smaller, larger = (first, second) if first[2] - first[0] < \
        second[2] - second[0] else (second, first)
    return larger[0] < (smaller[0] + smaller[2]) / 2 < larger[2]

def combineOnLine(line):
    i = 0
    while i < len(line.shapes):
        j = 0
        while j < len(line.shapes):
            if i == j:
                j += 1
                continue
            first = line.shapes[i]
            second = line.shapes[j]
            if doesCombine(first[0].bounds,second[0].bounds):
                line.removeShape(first)
                line.removeShape(second)
                line.addShape((first[0].union(second[0]),first[1] + second[1]))
                if j > i:
                    i -= 1
                else:
                    i -= 2
                break
            j += 1
        i += 1
    # return list(filter(lambda x: x[0].area > 10, bounds))
    return line

def multiPolyToPoly(multiPoly):
    polys = list(map(lambda x: x.bounds,multiPoly.geoms))
    if len(polys) == 0:
        return None
    x1 = min(polys, key = lambda x: x[0])[0]
    y1 = min(polys, key = lambda x: x[1])[1]
    x2 = max(polys, key = lambda x: x[2])[2]
    y2 = max(polys, key = lambda x: x[3])[3]
    return shapely.Polygon([(x1,y1),(x1,y2),(x2,y2),(x2,y1)])

def blockWhiteOut(image, hieros, colour, output = True):
    x1, y1, x2, y2 = None, None, None, None
    
    for line in hieros:
        for (poly, contours), weight in line:
            if type(poly) == shapely.MultiPolygon:
                poly = multiPolyToPoly(poly)
                if poly == None:
                    continue
            
            pbounds = poly.bounds
            if weight in [0.1,0.2,0.9,1]:
                x2 = pbounds[2]
                if x1 == None:
                    x1 = pbounds[0]
                if y1 == None or pbounds[1] < y1:
                    y1 = pbounds[1]
                if y2 == None or pbounds[3] > y2:
                    y2 = pbounds[3]
                continue
            
            else:
                if x1 == None:
                    continue
            
            poly = shapely.Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
            bounds = np.array(list(map(list,poly.exterior.coords))[1:], np.int32).reshape((-1,1,2))
            cv2.polylines(image, [bounds], isClosed = True, color = colour, thickness = 3)
            cv2.fillPoly(image, [bounds], color = colour)
            x1,y1,x2,y2 = None, None, None, None
        
def whiteOutHieros(image, hieros, output = True, fillArea = True, usesWeight = True):
    colour = (255,255,255) if output else 0
    if fillArea:
        blockWhiteOut(image, hieros, colour, output = True)
    else:
        contourWhiteOut(image, hieros, colour, output = True, usesWeight = usesWeight)
    if output:
        cv2.imwrite("outthing.png", image)
    
def contourWhiteOut(image, hieros, colour, output = True, usesWeight = True):
    for line in hieros:
        for item in line:
            if usesWeight:
                [(poly, contours), weight] = item
                if weight not in [0.1,0.2,0.9,1]:
                    continue
            else:
                (poly, contours) = item
            cv2.drawContours(image, contours, -1, color = colour, thickness = cv2.FILLED)
            cv2.drawContours(image, contours, -1, color = colour, thickness = 3)  

def adjustLines(lines, weighting = 0.5):
    lineDiffs = sorted([lines[i].bottom - lines[i - 1].bottom for i in range(1,len(lines))])
    if len(lineDiffs) == 0:
        return lines
    approxMedianLineDiff = lineDiffs[len(lineDiffs) // 2]
    
    for i in range(1, len(lines) - 1):
        prev = lines[i - 1].bottom
        next = lines[i + 1].bottom
        if abs((next - prev) // 2 - approxMedianLineDiff) < 4 and\
                (lines[i].bottom - prev) - approxMedianLineDiff > 3:
            lines[i].bottom = (weighting * (prev + (next - prev) // 2) + (1 - weighting) * lines[i].bottom)
    return lines

def postCombinationLineAdjust(lines):
    i = 1
    while i < len(lines):
        noAboveLine = sum([shape[0].bounds[3] < lines[i].bottom for shape in lines[i].shapes])
        if lines[i].bottom < lines[i - 1].bottom:
            lines[i].bottom = 0
            i += 1
            continue
        
        if (len(lines[i].shapes) == 0 or noAboveLine / len(lines[i].shapes) > 0.2):
            lines[i].bottom -= 1
            i -= 1
        i += 1
        
def filterTops(lines):
    for i in range(len(lines)):
        above = lines[i - 1].bottom if i != 0 else 0
        below = lines[i].bottom
        if lines[i].top == None or lines[i].top > below or lines[i].top < above:
            lines[i].top = None
            
def addLine(unmapped, bound, lines, image):
    possiblyOnLine = [bound]
    minX = bound[0].bounds[0]
    maxX = bound[0].bounds[2]
    meanYs = [(bound[0].bounds[1] + bound[0].bounds[3]) / 2]
    i = 1
    while i < len(unmapped):
        poly = unmapped[i][0].bounds
        if poly[0] > maxX + 30:
            i += 1
            continue
        
        if abs((x := (poly[1] + poly[3])/2) - (sum(meanYs) / len(meanYs))) < 20:
            meanYs.append(x)
            maxX = max(maxX, poly[2])
            possiblyOnLine.append(unmapped[i])
        i += 1
    
    top = min(map(lambda x: x[0].bounds[1], possiblyOnLine))
    tempBottom = max(map(lambda x: x[0].bounds[3], possiblyOnLine))
    
    for shape in possiblyOnLine:
        unmapped.remove(shape)
        
    line = characterLine(tempBottom, top, (minX, maxX), possiblyOnLine, [0]*len(possiblyOnLine))
    adjustBottom([line], 0, image)
    chainAcross([line], image)
    
    return line

def removeMalformed(reallyBad, additionalLines, lines):
    i = 0
    while i < len(additionalLines):
        if additionalLines[i].bottom - additionalLines[i].top < 10:
            reallyBad = reallyBad + additionalLines[i].shapes[:]
            del additionalLines[i]
            continue
        for line in lines:
            if additionalLines[i].intersects(line) or \
                    abs(additionalLines[i].top - line.bottom) < 5 or\
                    abs(additionalLines[i].bottom - line.bottom) < 5:
                reallyBad = reallyBad + additionalLines[i].shapes[:]
                del additionalLines[i]
                i -= 1
                break
        i += 1
    return reallyBad
    
def addAdditionalLines(lines, unmapped, image):
    if len(lines) <= 1:
        return unmapped[:], []
    
    lines.sort(key = lambda x: x.bottom)
    unmapped.sort(key = lambda x:x[0].bounds[0])
    
    diffs = [lines[i + 1].bottom - lines[i].bottom for i in range(0, len(lines) - 1)]
    medianDiff = diffs[len(diffs) // 2]
    
    reallyBad = []
    additionalLines = []
    
    i = 0
    while i < len(unmapped):
        bound, contours = unmapped[i]
        applicableLinesIndexes = []
        for j in range(len(lines)):
            if not(lines[j].edges[0] > bound.bounds[2] or lines[j].edges[1] < bound.bounds[0]):
                applicableLinesIndexes.append(j)
        applicableLines = [lines[i] for i in applicableLinesIndexes]
                
        low = lineIndex(applicableLines, bound)
        if low == 0 or low == len(applicableLines):
            difference = 100
        else:
            difference = applicableLines[low].bottom - applicableLines[low - 1].bottom
            
        # print(bound.bounds[0], difference)
        if difference >= 1.5 * medianDiff:
            x = addLine(unmapped, (bound, contours), lines, image)
            additionalLines.append(x)
        else:
            reallyBad.append((bound, contours))
            del unmapped[i]
            
    reallyBad = removeMalformed(reallyBad, additionalLines, lines)
    
    return reallyBad, additionalLines
            
def adjustBottom(lines, i, image):
    # something with edges
    oldVal = lines[i].bottom
    j = lines[i].bottom
    lastRow = sum(image[int(lines[i].bottom) - 1])
    startRow = sum(image[int(lines[i].bottom) - 1])
    while True:
        row = sum(image[int(j)][int(lines[i].edges[0]):int(lines[i].edges[1])])
        if j <= 0:
            lines[i].bottom = oldVal
            return
        if i != 0 and j < lines[i - 1].bottom:
            lines[i].bottom = oldVal
            return
        width = int(lines[i].edges[1]) - int(lines[i].edges[0])
        # print(i, j, row, width, width * 0.2 * 255)
        if (0.5 * row > lastRow or 0.5 * row > startRow) and row > 255 * width * 0.2:
            lines[i].bottom = j
            return
        j -= 1
        
def adjustTops(lines, image):
    for i in range(1, len(lines)):
        if lines[i].top != None and lines[i].top - lines[i - 1].bottom < 10:
            adjustBottom(lines, i - 1, image)
        
        if lines[i].top == None or lines[i].bottom - lines[i].top > 15:
            if sum(image[int(lines[i].top)][int(lines[i].edges[0]):int(lines[i].edges[1])]) < 2000:
                j = int(lines[i].top)
                while sum(image[j][int(lines[i].edges[0]):int(lines[i].edges[1])]) < 2000:
                    j += 1
                    if lines[i].bottom - j <= 15:
                        break
                else:
                    lines[i].top = j
            continue
        
        starts = np.where(image[int(lines[i].top)] != 0)
        if len(starts[0]) == 0:
            lines[i].top = None
            continue
        
        heights = []
        for startPoint in starts[0]:
            j = lines[i].top
            while j > 0 and image[int(j)][int(startPoint)] != 0:
                j -= 1
            heights.append(lines[i].top - j)
        if lines[i].top - max(heights) < lines[i - 1].bottom:
            lines[i].top = None
        else:
            lines[i].top -= max(heights)
            
def expandLine(line, index, image):
    i = line.edges[index]
    zeroRows = 0
    updated = line.edges[index]
    while True:
        column = sum([row[int(i)] for row in image[int(line.top):int(line.bottom)]])
        if column < 1000:
            zeroRows += 1
        else:
            zeroRows = 0
            updated = i
        if zeroRows >= 25 or i < 10 or i > len(image[0]) - 10:
            break
        i += 1 if index == 1 else -1
    return updated
            
def chainAcross(lines, image):
    for line in lines:
        updatedLeft = expandLine(line, 0, image)
        updatedRight = expandLine(line, 1, image)
        line.edges = (updatedLeft - 20, updatedRight + 20)
        
def addObviousMisses(unmapped, lines):
    addMostlyOnLine(unmapped, lines)
    addMuchCloserToLine(unmapped, lines)
    
def addMuchCloserToLine(unmapped, lines):
    k = 0
    while k < len(unmapped):
        shape, contours = unmapped[k]
        applicableLinesIndexes = []
        for j in range(len(lines)):
            if not(lines[j].edges[0] > shape.bounds[2] or lines[j].edges[1] < shape.bounds[0]):
                applicableLinesIndexes.append(j)
        applicableLines = [lines[i] for i in applicableLinesIndexes]
        
        i = 0
        while i < len(applicableLines) and shape.bounds[1] >= applicableLines[i].bottom:
            i += 1
            
        if i >= len(applicableLines):
            k += 1
            continue
        
        if i == 0:
            k += 1
            continue
        
        above = applicableLines[i - 1]
        below = applicableLines[i]
        
        distanceToAbove = shape.bounds[1] - above.bottom
        distanceToBelow = below.top - shape.bounds[3]
        
        # maybe rework to be after new lines are added
        if distanceToAbove < distanceToBelow * 0.1 and distanceToAbove < 10:
            above.addShape((shape, contours))
            del unmapped[k]
            k -= 1
        elif distanceToBelow < (distanceToAbove * 0.1) - 5 and distanceToBelow < 10:
            below.addShape((shape, contours))
            del unmapped[k]
            k -= 1
        k += 1
    
def addMostlyOnLine(unmapped, lines):
    allFound = []
    for i,(poly, conts) in enumerate(unmapped):
        low = lineIndex(lines, poly)
        found = False
        if low != 0:
            lineIntersection = lines[low - 1].intersection(poly)
            if lineIntersection.area > 0.5 * poly.area:
                lines[low - 1].addShape((poly, conts))
                found = True
        if low != len(lines) and not found:
            lineIntersection = lines[low].intersection(poly)
            if lineIntersection.area > 0.5 * poly.area:
                lines[low].addShape((poly, conts))
                found = True
        if found:
            allFound.append(i)
    
    for i in allFound[::-1]:
        del unmapped[i]
    
    for line in lines:
        line.sort()
        
def handleRemainingUnmapped(unmapped, lines):
    k = 0
    lines.sort(key = lambda x: x.bottom)
    while k < len(unmapped):
        shape, contours = unmapped[k]
        applicableLines = []
        for line in lines:
            if not(line.edges[0] > shape.bounds[2] or line.edges[1] < shape.bounds[0]):
                applicableLines.append(line)
        
        if len(applicableLines) == 0:
            k += 1
            continue
                
        low = lineIndex(applicableLines, shape)
        if low == len(applicableLines) or low == 0:
            line = applicableLines[low - 1] if low == len(applicableLines) else applicableLines[low]
            topDist = shape.bounds[1] - line.bottom
            bottomDist = line.top - shape.bounds[3]
            # print(topDist, bottomDist, shape.bounds[0], low)
            if bottomDist < 15 and topDist < 15:
                line.addShape((shape,contours))
                del unmapped[k]
                continue
            else:
                k += 1
                continue
        
        if applicableLines[low].bottom < shape.bounds[3]:
            distance = shape.bounds[1] - applicableLines[low].bottom
            line = applicableLines[low]
        elif applicableLines[low - 1].bottom < shape.bounds[3]:
            distance = shape.bounds[1] - applicableLines[low - 1].bottom
            line = applicableLines[low - 1]
        
        if distance < 20:
            line.addShape((shape, contours))
            del unmapped[k]
            continue
        k += 1