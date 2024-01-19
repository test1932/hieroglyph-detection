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
        hieros = [self.lines[lineNo][charNo] for lineNo,charNo in self.hierosLineIndexes] + self.newBounds
        for x1,y1,x2,y2 in hieros:
            cv2.imwrite(f"out/{self.imageFname}-{int(x1)}-{int(y1)}-{int(x2)}-{int(y2)}.png", \
                        self.colour[int(y1):int(y2),int(x1):int(x2)])
            points = np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]], np.int32)
            cv2.polylines(self.colour, [points], isClosed = True, color = colour, thickness = 3)
            cv2.fillPoly(self.colour, [points], color = colour)
        cv2.imwrite("outthing.png", self.colour)
    
    def createSetMode(self, mode):
        def onclick():
            self.curMode = mode
        return onclick
        
    def main(self):
        loopBreak = True
        clock = pygame.time.Clock()
        
        while loopBreak:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    loopBreak = False
                    break
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
                            self.selection = offsetMousePos
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
                            self.selection = None
                            break
                        
                    if mousePos[0] < 650:
                        if self.curMode == COMBINE_LINES:
                            self.selection, self.hierosLineIndexes = self.handleCombineLines(offsetMousePos)
                        elif self.curMode == COMBINE_BOUNDS:
                            self.selection = self.handleCombineBounds(offsetMousePos)
                        elif self.curMode == REMOVE_HIEROS:
                            hit = self.handleRemoveHieros(offsetMousePos)
                            if not hit:
                                self.curMode = None
                        elif self.curMode == ADD_HIERO_BOUNDS:
                            hit = self.handleAddHieros(offsetMousePos)
                            if not hit:
                                self.curMode = None
                        elif self.curMode == MOVE_HIERO:
                            self.selection = self.handleMoveCharacter(offsetMousePos)
                        elif self.curMode == FREEHAND:
                            self.selection = self.handleFreehand(offsetMousePos)
                    
            self.screen.fill((200,200,255))
            self.screen.blit(self.scaled, self.offset)
            
            if self.curMode == REMOVE_HIEROS:
                self.displayLines(ONLY_HIEROS)
            elif self.curMode == ADD_HIERO_BOUNDS:
                self.displayLines(ONLY_NOT_HIEROS)
            elif self.curMode == MOVE_HIERO:
                if self.selection != None:
                    self.displayLines()
                else:
                    self.displayLines(ONLY_HIEROS)
            else:
                self.displayLines()
            
            pygame.draw.rect(self.screen, (200,200,200), [650,0,250,1000])
            for thing in self.buttons:
                self.screen.blit(thing.image, (thing.x, thing.y))
                
            if self.curMode != None:
                sel = self.buttons[self.curMode]
                pygame.draw.rect(self.screen, (100,255,50), [sel.x, sel.y, sel.width, sel.height], 3)

            pygame.display.flip()
            clock.tick(60)

    def displayLines(self, args = 0):
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
            lines = self.lines + [self.newBounds]
        else:
            lines = self.lines
        
        for i,line in enumerate(lines):
            for j,bounds in enumerate(line):
                if ([i,j] in self.hierosLineIndexes and args == ONLY_NOT_HIEROS) or \
                        ([i,j] not in self.hierosLineIndexes and args == ONLY_HIEROS and i != len(lines) - 1):
                    continue
                x1,y1,x2,y2 = list(map(lambda x: (x / self.baseScale * self.scale), bounds))
                pygame.draw.rect(self.screen, colours[i % len(colours)], \
                                (x1 + self.offset[0],y1 + self.offset[1], x2 - x1, y2 - y1), 2)
    
    def getSelected(self, mousePos):
        startIndex, endIndex = 0, len(self.lines) - 1
        selectedLine = None
        selectedBound = None
        found = False
        for i,line in enumerate(self.lines[startIndex:endIndex + 1]):
            for j,bound in enumerate(line):
                # print(startIndex + i)
                if bound[0] <= mousePos[0] * self.baseScale / self.scale <= bound[2] and \
                        bound[1] <= mousePos[1] * self.baseScale / self.scale <= bound[3]:
                    selectedLine = startIndex + i
                    selectedBound = j
                    found = True
                    break
            if found:
                break
        return selectedLine, selectedBound, found

    def handleCombineBounds(self, mousePos):
        if mousePos[0] > 650:
            return None
        
        selectedLine, selectedBound, found = self.getSelected(mousePos)
        
        if (selectedLine, selectedBound) == self.selection or not found:
            return None
        
        if self.selection != None:
            firstChar = self.lines[self.selection[0]][self.selection[1]]
            secondChar = self.lines[selectedLine][selectedBound]
            newChar = boundBounds(firstChar, secondChar)
            if self.selection[0] == selectedLine:
                del self.lines[selectedLine][max(selectedBound, self.selection[1])]
                del self.lines[selectedLine][min(selectedBound, self.selection[1])]
                self.lines[selectedLine].insert(min(selectedBound, self.selection[1]), newChar)
            else:
                del self.lines[self.selection[0]][self.selection[1]]
                del self.lines[selectedLine][selectedBound]
                
                self.lines[selectedLine].insert(selectedBound, newChar)
                
                if len(self.lines[self.selection[0]]) == 0:
                    for i in range(len(self.hierosLineIndexes)):
                        if self.hierosLineIndexes[i][0] >= self.selection[0]:
                            self.hierosLineIndexes[i][0] -= 1
                    del self.lines[self.selectedIndex[0]]
                    
            for i in range(len(self.hierosLineIndexes)):
                if self.hierosLineIndexes[i][0] == self.selection[0] and self.hierosLineIndexes[i][1] > self.selection[1]:
                    self.hierosLineIndexes[i][1] -= 1
                    
            return None
        return (selectedLine, selectedBound)

    def handleRemoveHieros(self, mousePos):
        if mousePos[0] > 650:
            return False
        
        selectedLine, selectedBound, found = self.getSelected(mousePos)
        
        if not found:
            for i,(x1,y1,x2,y2) in enumerate(self.newBounds):
                if x1 <= mousePos[0] * self.baseScale / self.scale <= x2 and y1 <= mousePos[1] * self.baseScale / self.scale <= y2:
                    del self.newBounds[i]
                    return True
            return False
        
        if [selectedLine, selectedBound] in self.hierosLineIndexes:
            self.hierosLineIndexes.remove([selectedLine, selectedBound])
            return True
        
    def handleAddHieros(self, mousePos):
        if mousePos[0] > 650:
            return False
        
        selectedLine, selectedBound, found = self.getSelected(mousePos)
        
        if not found:
            return False
        
        self.hierosLineIndexes.append([selectedLine, selectedBound])
        return True
    
    def handleMoveCharacter(self, mousePos):
        if mousePos[0] > 650:
            return None
        
        selectedLine, selectedBound, found = self.getSelected(mousePos)
        
        if not found:
            return None
        
        if self.selection == None:
            return (selectedLine, selectedBound)
        
        x = self.lines[self.selection[0]][self.selection[1]]
        
        self.lines[selectedLine].append(x)
        self.lines[selectedLine].sort(key = lambda x:x[0])
        del self.lines[self.selection[0]][self.selection[1]]
        
        pos = self.lines[selectedLine].index(x)
        
        i = 0
        while i < len(self.hierosLineIndexes):
            if self.hierosLineIndexes[i][0] == self.selection[0]:
                if self.hierosLineIndexes[i][1] > self.selection[1]:
                    self.hierosLineIndexes[i][1] -= 1
                elif self.hierosLineIndexes[i][1] == self.selection[1]:
                    self.hierosLineIndexes[i][0] = selectedLine
                    self.hierosLineIndexes[i][1] = pos
            elif self.hierosLineIndexes[i][0] == selectedLine and self.hierosLineIndexes[i][1] >= pos:
                self.hierosLineIndexes[i][1] += 1
            i += 1
        return None

    def handleCombineLines(self, mousePos):
        if mousePos[0] > 650:
            return None, self.hierosLineIndexes
        
        selected, _, found = self.getSelected(mousePos)
        if selected == self.selection or not found:
            return None, self.hierosLineIndexes
        
        if self.selection != None:
            firstLine = [(bound,1) if [self.selection,i] in self.hierosLineIndexes else (bound,0) \
                for i,bound in enumerate(self.lines[self.selection])]
            secondLine = [(bound,1) if [selected,i] in self.hierosLineIndexes else (bound,0) \
                for i,bound in enumerate(self.lines[selected])]
            
            for i in range(len(firstLine)):
                if firstLine[i][1] == 1:
                    self.hierosLineIndexes.remove([self.selection,i])
                    
            for i in range(len(secondLine)):
                if secondLine[i][1] == 1:
                    self.hierosLineIndexes.remove([selected,i])
            
            sortedLine = sorted(firstLine + secondLine, key = lambda x:x[0][0])
            x = min(selected, self.selection)
            tempHieros = [[x, j] for j, item in enumerate(sortedLine) if item[1] == 1]
            newLine = [item[0] for item in sortedLine]
            
            del self.lines[max(selected,self.selection)]
            del self.lines[min(selected,self.selection)]
            self.hierosLineIndexes = self.hierosLineIndexes + tempHieros
            simpleCombineOnLine(newLine, self.hierosLineIndexes, x)
            self.lines.insert(x, newLine)
            
            i = 0
            while i < len(self.hierosLineIndexes):
                if self.hierosLineIndexes[i][0] > max(selected,self.selection):
                    self.hierosLineIndexes[i][0] -= 1
                i += 1
            return None, self.hierosLineIndexes
        return selected, self.hierosLineIndexes

    def handleFreehand(self, mousePos):
        if mousePos[0] > 650:
            return None
        
        newRect = [min(self.selection[0], mousePos[0]), min(self.selection[1], mousePos[1]), \
            max(self.selection[0], mousePos[0]), max(self.selection[1], mousePos[1])]
        self.newBounds.append(list(map(lambda x: x * self.baseScale / self.scale, newRect)))
        return None

if __name__ == '__main__':
    editor = application(sys.argv[1], sys.argv[2], sys.argv[3])
    editor.main()