import pygame
import sys
from removeGlyphs import doesCombine
from detectStructure import getImage
import cv2
import numpy as np

pygame.init()

HEIGHT = 1000
WIDTH = 900

COMBINE_LINES = 0
COMBINE_BOUNDS = 1
REMOVE_HIEROS = 2
ADD_HIERO_BOUNDS = 3
MOVE_HIERO = 4
FREEHAND = 5

ONLY_HIEROS = 1
ONLY_NOT_HIEROS = 2

smallFont = pygame.font.Font(pygame.font.get_default_font(), 15)

class button:
    def __init__(self, x, y, width, height, text, onclick):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.onclick = onclick
        
        self.image = pygame.Surface((width, height))
        self.image.fill((100,100,150))
        self.image.blit(smallFont.render(self.text, False, (0,0,0)), (5,5))
        
    def isClicked(self, x, y):
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height


def boundBounds(boundsFirst, boundsSecond):
    return [min(boundsFirst[0], boundsSecond[0]), 
            min(boundsFirst[1], boundsSecond[1]),
            max(boundsFirst[2], boundsSecond[2]),
            max(boundsFirst[3], boundsSecond[3]),]

def simpleCombineOnLine(shapes:list, hieros, lineIndex):
    i = 0
    while i < len(shapes):
        j = 0
        while j < len(shapes):
            if i == j:
                j += 1
                continue
            first = shapes[i]
            second = shapes[j]
            if doesCombine(first, second):
                del shapes[max(i,j)]
                del shapes[min(i,j)]
                
                if [lineIndex,max(i,j)] in hieros:
                    if [lineIndex,min(i,j)] in hieros:
                        hieros.remove([lineIndex,max(i,j)])
                    else:
                        hieros.remove([lineIndex,max(i,j)])
                        hieros.append([lineIndex,min(i,j)])
                
                for hiero in hieros:
                    if hiero[1] >= max(i,j) and hiero[0] == lineIndex:
                        hiero[1] -= 1
                    
                shapes.insert(min(i,j), boundBounds(first,second))
                if j > i:
                    i -= 1
                else:
                    i -= 2
                break
            j += 1
        i += 1

def parseAnalysis(fname):
    data = list(filter(lambda x: x != '', open(fname, "r").read().split("\n\n")))
    hieros = list(map(lambda x: list(map(int, x.split(','))), data[0].split(';')))
    bounds = list(map(lambda x: list(map(float, x.split(','))), data[1].split('\n')))
    lines = list(map(lambda x: list(map(lambda y: list(map(float,y.split(','))), x.split(';'))),  list(filter(lambda x: x != '', data[2].split('\n')))))
    return hieros, bounds, lines

class application:
    def __init__(self, imageFname, boundsFname, idName):
        self.imageFname = imageFname
        self.boundsFname = boundsFname
        self.idName = idName
        self.hierosLineIndexes, self.bounds, self.lines = parseAnalysis(boundsFname)
        self.cvimage, self.colour, self.contours = getImage(imageFname)
        self.image = pygame.image.load(imageFname)
        
        self.baseScale = (self.image.get_height() / HEIGHT)
        self.originalScaled = pygame.transform.scale(self.image, \
            (self.image.get_width() / self.baseScale, HEIGHT))
        self.scale = 1
        self.scaled = self.originalScaled
        
        self.screen = pygame.display.set_mode((WIDTH,HEIGHT))
        self.curMode = None
        self.selection = None
        
        self.offset = [0,0]
        self.newBounds = []
        self.firstOffsetPos = None
        
        self.buttons = [
            button(700,50,150,30,"Combine Lines",self.createSetMode(COMBINE_LINES)),
            button(700,100,150,30,"Combine Bounds",self.createSetMode(COMBINE_BOUNDS)),
            button(700,150,150,30,"Remove Hieros",self.createSetMode(REMOVE_HIEROS)),
            button(700,200,150,30,"Add Hieros",self.createSetMode(ADD_HIERO_BOUNDS)),
            button(700,250,150,30,"Move Hieros",self.createSetMode(MOVE_HIERO)),
            button(700,300,150,30,"Freehand Add",self.createSetMode(FREEHAND)),
            
            button(700,500,150,30,"Write to file",self.writeChanges),
            
            button(700,550,30,30,"-",self.makeZoom(-0.2)),
            button(820,550,30,30,"+",self.makeZoom(0.2))
        ]
        
    def makeZoom(self, increment):
        def zoom():
            self.scale += increment
            self.scaled = pygame.transform.scale(self.originalScaled, \
                (self.scale * self.originalScaled.get_width(), self.scale * self.originalScaled.get_height()))
        return zoom
    
    def writeChanges(self):
        colour = (255,255,255)
        hieros = [self.lines[lineNo][charNo] for lineNo,charNo in hieros] + self.newBounds
        for x1,y1,x2,y2 in hieros:
            cv2.imwrite(f"out/{self.imageName}-{int(x1)}-{int(y1)}-{int(x2)}-{int(y2)}.png", \
                        self.image[int(y1):int(y2),int(x1):int(x2)])
            points = np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]], np.int32)
            cv2.polylines(self.image, [points], isClosed = True, color = colour, thickness = 3)
            cv2.fillPoly(self.image, [points], color = colour)
        cv2.imwrite("outthing.png", self.image)
    
    def createSetMode(self, mode):
        def onclick():
            self.curMode = mode
        return onclick
        
    def main(self):
        loopBreak = True
        clock = pygame.time.Clock()
        
        while loopBreak:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    if self.firstOffsetPos != None:
                        mousePos = pygame.mouse.get_pos()
                        self.offset[0] += mousePos[0] - self.firstOffsetPos[0]
                        self.offset[1] += mousePos[1] - self.firstOffsetPos[1]
                        self.firstOffsetPos = mousePos
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mousePos = pygame.mouse.get_pos()
                    offsetMousePos = [mousePos[0] - self.offset[0], mousePos[1] - self.offset[1]]
                    if mousePos[0] < 650:
                        if self.curMode == None:
                            self.firstOffsetPos = mousePos
                        if self.curMode == FREEHAND:
                            selection = offsetMousePos
                if event.type == pygame.MOUSEBUTTONUP:
                    mousePos = pygame.mouse.get_pos()
                    offsetMousePos = [mousePos[0] - self.offset[0], mousePos[1] - self.offset[1]]
                    
                    if self.curMode == None and self.firstOffsetPos != None:
                        self.firstOffsetPos = None
                        continue
                    
                    for i, thing in enumerate(self.buttons):
                        if thing.isClicked(*mousePos):
                            if i == self.curMode:
                                self.curMode = None
                                break
                            thing.onclick()
                            selection = None
                            break
                        
                    if mousePos[0] < 650:
                        if self.curMode == COMBINE_LINES:
                            selection, hierosLineIndexes = handleCombineLines(selection, offsetMousePos, \
                                self.lines, (self.image.get_width(), self.image.get_height()), self.hierosLineIndexes, self.baseScale, self.scale)
                        elif self.curMode == COMBINE_BOUNDS:
                            selection = handleCombineBounds(selection, offsetMousePos, \
                                self.lines, (self.image.get_width(), self.image.get_height()), self.hierosLineIndexes, self.baseScale, self.scale)
                        elif self.curMode == REMOVE_HIEROS:
                            hit = handleRemoveHieros(offsetMousePos, self.newBounds,
                                self.lines, (self.image.get_width(), self.image.get_height()), self.hierosLineIndexes, self.baseScale, self.scale)
                            if not hit:
                                self.curMode = None
                        elif self.curMode == ADD_HIERO_BOUNDS:
                            hit = handleAddHieros(offsetMousePos, \
                                self.lines, (self.image.get_width(), self.image.get_height()), self.hierosLineIndexes, self.baseScale, self.scale)
                            if not hit:
                                self.curMode = None
                        elif self.curMode == MOVE_HIERO:
                            selection = handleMoveCharacter(selection, offsetMousePos, \
                                self.lines, (self.image.get_width(), self.image.get_height()), self.baseScale, self.scale, self.hierosLineIndexes)
                        elif self.curMode == FREEHAND:
                            selection = handleFreehand(selection, offsetMousePos, (self.image.get_width(), self.image.get_height()),\
                                self.newBounds, self.baseScale, self.scale)
                    
            self.screen.fill((200,200,255))
            self.screen.blit(self.scaled, self.offset)
            
            if self.curMode == REMOVE_HIEROS:
                displayLines(self.screen, self.newBounds, self.lines, self.hierosLineIndexes, self.baseScale, self.offset, self.scale, ONLY_HIEROS)
            elif self.curMode == ADD_HIERO_BOUNDS:
                displayLines(self.screen, self.newBounds, self.lines, self.hierosLineIndexes, self.baseScale, self.offset, self.scale, ONLY_NOT_HIEROS)
            elif self.curMode == MOVE_HIERO:
                if selection != None:
                    displayLines(self.screen, self.newBounds, self.lines, self.hierosLineIndexes, self.baseScale, self.offset, self.scale)
                else:
                    displayLines(self.screen, self.newBounds, self.lines, self.hierosLineIndexes, self.baseScale, self.offset, self.scale, ONLY_HIEROS)
            else:
                displayLines(self.screen, self.newBounds, self.lines, self.hierosLineIndexes, self.baseScale, self.offset, self.scale)
            
            pygame.draw.rect(self.screen, (200,200,200), [650,0,250,1000])
            for thing in self.buttons:
                self.screen.blit(thing.image, (thing.x, thing.y))
                
            if self.curMode != None:
                sel = self.buttons[self.curMode]
                pygame.draw.rect(self.screen, (100,255,50), [sel.x, sel.y, sel.width, sel.height], 3)

            pygame.display.flip()
            clock.tick(60)

def displayLines(screen, newBounds, lines, hieros, baseScale, offset, scale, args = 0):
    colours = [
        (100,200,100),
        (200,100,100),
        (100,100,200),
        (200,200,100),
        (200,100,200),
        (100,200,200),
        (200,200,200)
    ]
    
    if args != ONLY_NOT_HIEROS:
        lines = lines + [newBounds]
    
    for i,line in enumerate(lines):
        for j,bounds in enumerate(line):
            if ([i,j] in hieros and args == ONLY_NOT_HIEROS) or \
                    ([i,j] not in hieros and args == ONLY_HIEROS and i != len(lines) - 1):
                continue
            x1,y1,x2,y2 = list(map(lambda x: (x / baseScale * scale), bounds))
            pygame.draw.rect(screen, colours[i % len(colours)], \
                            (x1 + offset[0],y1 + offset[1], x2 - x1, y2 - y1), 2)

def getSelected(lines, mousePos, baseScale, scale):
    startIndex, endIndex = 0, len(lines) - 1
    selectedLine = None
    selectedBound = None
    found = False
    for i,line in enumerate(lines[startIndex:endIndex + 1]):
        for j,bound in enumerate(line):
            # print(startIndex + i)
            if bound[0] <= mousePos[0] * baseScale / scale <= bound[2] and \
                    bound[1] <= mousePos[1] * baseScale / scale <= bound[3]:
                selectedLine = startIndex + i
                selectedBound = j
                found = True
                break
        if found:
            break
    return selectedLine, selectedBound, found

def handleCombineBounds(selectedIndex, mousePos, lines, imageDims, hieros, baseScale, scale):
    if mousePos[0] > imageDims[0]:
        return None
    
    selectedLine, selectedBound, found = getSelected(lines, mousePos, baseScale, scale)
    
    if (selectedLine, selectedBound) == selectedIndex or not found:
        return None
    
    if selectedIndex != None:
        firstChar = lines[selectedIndex[0]][selectedIndex[1]]
        secondChar = lines[selectedLine][selectedBound]
        newChar = boundBounds(firstChar, secondChar)
        if selectedIndex[0] == selectedLine:
            del lines[selectedLine][max(selectedBound, selectedIndex[1])]
            del lines[selectedLine][min(selectedBound, selectedIndex[1])]
            lines[selectedLine].insert(min(selectedBound, selectedIndex[1]), newChar)
        else:
            del lines[selectedIndex[0]][selectedIndex[1]]
            del lines[selectedLine][selectedBound]
            
            lines[selectedLine].insert(selectedBound, newChar)
            
            if len(lines[selectedIndex[0]]) == 0:
                for i in range(len(hieros)):
                    if hieros[i][0] >= selectedIndex[0]:
                        hieros[i][0] -= 1
                del lines[selectedIndex[0]]
                
        for i in range(len(hieros)):
            if hieros[i][0] == selectedIndex[0] and hieros[i][1] > selectedIndex[1]:
                hieros[i][1] -= 1
                
        return None
    return (selectedLine, selectedBound)

def handleRemoveHieros(mousePos, newBounds, lines, imageDims, hieros, baseScale, scale):
    if mousePos[0] > imageDims[0]:
        return False
    
    selectedLine, selectedBound, found = getSelected(lines, mousePos, baseScale, scale)
    
    if not found:
        print(newBounds[0], mousePos)
        for i,(x1,y1,x2,y2) in enumerate(newBounds):
            if x1 <= mousePos[0] * baseScale / scale <= x2 and y1 <= mousePos[1] * baseScale / scale <= y2:
                del newBounds[i]
                return True
        return False
    
    if [selectedLine, selectedBound] in hieros:
        hieros.remove([selectedLine, selectedBound])
        return True
        
def handleAddHieros(mousePos, lines, imageDims, hieros, baseScale, scale):
    if mousePos[0] > imageDims[0]:
        return False
    
    selectedLine, selectedBound, found = getSelected(lines, mousePos, baseScale, scale)
    
    if not found:
        return False
    
    hieros.append([selectedLine, selectedBound])
    return True
    
def handleMoveCharacter(selectedIndex, mousePos, lines, imageDims, baseScale, scale, hieros):
    if mousePos[0] > imageDims[0]:
        return None
    
    selectedLine, selectedBound, found = getSelected(lines, mousePos, baseScale, scale)
    
    if not found:
        return None
    
    if selectedIndex == None:
        return (selectedLine, selectedBound)
    
    x = lines[selectedIndex[0]][selectedIndex[1]]
    
    lines[selectedLine].append(x)
    lines[selectedLine].sort(key = lambda x:x[0])
    del lines[selectedIndex[0]][selectedIndex[1]]
    
    pos=lines[selectedLine].index(x)
    
    i = 0
    while i < len(hieros):
        if hieros[i][0] == selectedIndex[0]:
            if hieros[i][1] > selectedIndex[1]:
                hieros[i][1] -= 1
            elif hieros[i][1] == selectedIndex[1]:
                hieros[i][0] = selectedLine
                hieros[i][1] = pos
        elif hieros[i][0] == selectedLine and hieros[i][1] >= pos:
            hieros[i][1] += 1
        i += 1
    return None

def handleCombineLines(selectedIndex, mousePos, lines, imageDims, hieros, baseScale, scale):
    if mousePos[0] > imageDims[0]:
        return None, hieros
    
    selected, _, found = getSelected(lines, mousePos, baseScale, scale)
    if selected == selectedIndex or not found:
        return None, hieros
    
    if selectedIndex != None:
        firstLine = [(bound,1) if [selectedIndex,i] in hieros else (bound,0) \
            for i,bound in enumerate(lines[selectedIndex])]
        secondLine = [(bound,1) if [selected,i] in hieros else (bound,0) \
            for i,bound in enumerate(lines[selected])]
        
        for i in range(len(firstLine)):
            if firstLine[i][1] == 1:
                hieros.remove([selectedIndex,i])
                
        for i in range(len(secondLine)):
            if secondLine[i][1] == 1:
                hieros.remove([selected,i])
        
        sortedLine = sorted(firstLine + secondLine, key = lambda x:x[0][0])
        x = min(selected, selectedIndex)
        tempHieros = [[x, j] for j, item in enumerate(sortedLine) if item[1] == 1]
        newLine = [item[0] for item in sortedLine]
        
        del lines[max(selected,selectedIndex)]
        del lines[min(selected,selectedIndex)]
        hieros = hieros + tempHieros
        simpleCombineOnLine(newLine, hieros, x)
        lines.insert(x, newLine)
        
        i = 0
        while i < len(hieros):
            if hieros[i][0] > max(selected,selectedIndex):
                hieros[i][0] -= 1
            i += 1
        return None, hieros
    return selected, hieros

def handleFreehand(selection, mousePos, imageDims, newBounds, baseScale, scale):
    if mousePos[0] > imageDims[0]:
        return None
    
    newRect = [min(selection[0], mousePos[0]), min(selection[1], mousePos[1]), \
        max(selection[0], mousePos[0]), max(selection[1], mousePos[1])]
    newBounds.append(list(map(lambda x: x * baseScale / scale, newRect)))
    return None

if __name__ == '__main__':
    editor = application(sys.argv[1], sys.argv[2], sys.argv[3])
    editor.main()