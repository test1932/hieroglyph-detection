import json
import re
import sys
import shapely
import numpy as np

# mapping for correcting misclassified characters in page numbers
NUMBER_CORRECTION_MAPPING = {
    'o':'0',
    'q':'0',
    'i':'1',
    'l':'1',
    'z':'2'
}

# load JSON data and break it into component parts
def loadData(fileName):
    file = open(fileName, "r", encoding='utf-8')
    metaDict = json.loads(file.read())
    file.close()
    styles = metaDict['analyzeResult']['styles']
    paragraphs = metaDict['analyzeResult']['paragraphs']
    lines = metaDict['analyzeResult']['pages'][0]['lines']
    words = metaDict['analyzeResult']['pages'][0]['words']
    return paragraphs, lines, words, styles, metaDict['analyzeResult']['pages'][0]

# loaf file of bounds of hieroglyphs
def loadHieroBounds(filename):
    file = open(filename, "r", encoding="utf-8")
    bounds = list(map(lambda x:list(map(float, x.split(','))),filter(lambda x: x,file.read().split("\n"))))
    file.close()
    return bounds

# checks if a word is likely to be a roman numeral
def isSimpleRomanNumeral(text):
    chars = {'(',')',',','.','[',']','M','D','C','L','X','V','I'}
    for char in text.upper():
        if char not in chars:
            return False
    return len(text) > 1

# predicate for checking if a word is likely to be initials
def isInitials(text):
    return '.' in text[1:-1]

# checks if a paragraph is a heading
def isParaCentred(para, minX, maxX):
    paraBounds = para['boundingRegions'][0]['polygon']
    return abs((paraBounds[0] - minX) - (maxX - paraBounds[2])) < (maxX - minX) / 20 and \
        (paraBounds[0] - minX) > (maxX - minX) / 10 and (maxX - paraBounds[2]) > (maxX - minX) / 10

# checks if a paragraph is likely to be a page number, and corrects it if so
def isNumber(para):
    retVal = len(para['content'].split(' ')) == 1 and any(c.isdigit() for c in para['content'])
    if retVal:
        newNumber = []
        for char in para['content']:
            if char.lower() in NUMBER_CORRECTION_MAPPING:
                newNumber.append(NUMBER_CORRECTION_MAPPING[char.lower()])
            else:
                newNumber.append(char)
        para['content'] = "".join(newNumber)
    return retVal

# adds structural paragraph and heading tags to the list of tags as required
def addParagraphs(paragraphs, lines, tags, minX, maxX):
    for para in paragraphs:
        tag = ('<p>&nbsp&nbsp&nbsp&nbsp','</p>')
        paraBounds = para['boundingRegions'][0]['polygon']
        if isParaCentred(para, minX,maxX):
            tag = ('<h2>','</h2>')
        elif isNumber(para) and abs((paraBounds[0] - minX) - (maxX - paraBounds[2])) > (maxX - minX) / 2:
            floatClass = 'leftblock' if paraBounds[0] - minX > maxX - paraBounds[2] else 'rightblock'
            tag = (f'<div class="{floatClass}">','</div>')
        tags.append((f'{tag[0]}',para['spans'][0]['offset']))
        tags.append((f'{tag[1]}',para['spans'][0]['offset'] + para['spans'][0]['length']))

# adds like breaks at the end of each line
def addLineBreaks(lines, tags):
    for line in lines:
        breakPos = line['spans'][0]['offset'] + line['spans'][0]['length']
        endPara = False
        for tag in tags:
            if tag[0] in {"</p>","</h2>"} and tag[1] == breakPos:
                endPara = True
                break
        if not endPara:
            tags.append(('<br/>',breakPos))

# add tags into flat text
def tagsAndTextToHTML(text, tags):
    tags.sort(key=lambda x:x[1])
    htmlStr = ""
    for i,tag in enumerate(tags):
        if i < len(tags) - 1:
            htmlStr = htmlStr + tag[0] + text[tag[1]:tags[i + 1][1]]
            continue
        htmlStr = htmlStr + tag[0] + text[tag[1]:]
        
    return htmlStr

# add tags for smallCaps
def addSmallCapsStyle(allText, words, tags):
    allCaps = []
    for word in words:
        if word['content'].isupper() and len(word['content']) >= 2:
            tags.append(("<span class='smallUpper'>",word['span']['offset']))
            tags.append(("</span>",word['span']['offset'] + word['span']['length']))
            allCaps.append((word['span']['offset'], word['span']['offset'] + word['span']['length']))

    partStr = allText[:allCaps[0][0]] if allCaps else allText
    for i in range(len(allCaps)):
        text = allText[allCaps[i][0]:allCaps[i][1]]
        partStr += text if isSimpleRomanNumeral(text) or isInitials(text) else text[0] + text[1:].lower()
        
        if i == len(allCaps) - 1:
            partStr += allText[allCaps[i][1]:]
        else:
            partStr += allText[allCaps[i][1]:allCaps[i + 1][0]]
    
    return partStr

# get edges of the text in pixels
def getParasExtremes(paras, page):
    minX, maxX = None, None
    for para in paras:
        paraBounds = para['boundingRegions'][0]['polygon']
        if minX == None or paraBounds[0] < minX:
            minX = paraBounds[0]
        if maxX == None or paraBounds[2] > maxX:
            maxX = paraBounds[2]
    if minX == None:
        minX, maxX = 0, page['width']
    return minX, maxX

# add italics to tags
def addItalics(styles, tags):
    ignoreKeys = {'confidence','spans'}
    for i,style in enumerate(styles[1:]):
        prop = list(set(style.keys()).difference(ignoreKeys))[0]
        if style[prop] == 'italic' and prop == 'fontStyle':
            for span in style['spans']:
                tags.append((f'<i>', span['offset']))
                tags.append((f'</i>', span['offset'] + span['length']))

# get line with top closest to the specified value, O(log(n))
def closestLine(lines, target):
    low, high = 0, len(lines) - 1

    while low <= high:
        mid = (low + high) // 2
        curTop = lines[mid]['polygon'][1]

        if curTop == target:
            return mid
        elif curTop < target:
            low = mid + 1
        else:
            high = mid - 1

    if low > 0 and (low == len(lines) or abs(lines[low - 1]['polygon'][1] - target) \
            < abs(lines[low]['polygon'][1] - target)):
        return low - 1
    else:
        return low

# get last word before the specified offset, O(log(n))
def wordBefore(words, target):
    low, high = 0, len(words) - 1

    while low <= high:
        mid = (low + high) // 2
        curStart = words[mid]['span']['offset']

        if curStart == target:
            return mid
        elif curStart < target:
            low = mid + 1
        else:
            high = mid - 1

    if low > 0 and (low == len(words)):
        return low - 1
    else:
        return low
    
# check if word overlaps hieroglyph bounds b
def wordOverlaps(word, b):
    poly1 = shapely.Polygon(np.array(word['polygon']).reshape([-1,2]))
    poly2 = shapely.Polygon([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]])
    return poly1.intersection(poly2).area > 0.5 * poly1.area or poly1.intersection(poly2).area > 0.5 * poly2.area

# remove words significantly overlapping hieroglyph areas
def removeHieros(flatText, words, lines, hierosBounds):
    wordsToRemove = []
    
    for x1Str, y1Str, x2Str, y2Str in hierosBounds:
        x1,y1,x2,y2 = float(x1Str), float(y1Str), float(x2Str), float(y2Str)
        closestLineIndex = closestLine(lines, y1)
        
        linesSubset = lines[closestLineIndex - 1: closestLineIndex + 2]
        start = linesSubset[0]["spans"][0]['offset']
        end = linesSubset[-1]["spans"][0]['offset'] + linesSubset[-1]["spans"][0]['length']
        
        wordIndex = wordBefore(words, start)
        while words[wordIndex]['span']['offset'] + words[wordIndex]['span']['length'] < end:
            if wordOverlaps(words[wordIndex], (x1,y1,x2,y2)):
                wordsToRemove.append(words[wordIndex])
            wordIndex += 1

    partStr = flatText[:wordsToRemove[0]['span']['offset']] if wordsToRemove else flatText
    for i in range(len(wordsToRemove)):
        offset,length = wordsToRemove[i]['span']['offset'], wordsToRemove[i]['span']['length']
        partStr += ' '*length
        
        if i == len(wordsToRemove) - 1:
            partStr += flatText[offset + length:]
        else:
            partStr += flatText[offset + length:wordsToRemove[i + 1]['span']['offset']]
    return partStr

# turn azure output into HTML
def wordsToHTML(paragraphs, lines, words, styles, page, hierosBounds):
    tags = []
    styleBlocks = [
        '.smallUpper {font-variant:small-caps;}',
        '.leftblock {float:left;}',
        '.rightblock {float:right;}',
        'h2 {text-align: center;font-size: 18px;font-weight: normal;}'
    ]

    minX, maxX = getParasExtremes(paragraphs, page)
    addParagraphs(paragraphs, lines, tags, minX, maxX)
    addLineBreaks(lines, tags)
    addItalics(styles,tags)
    
    flatStr = removeHieros("\n".join([para['content'] for para in paragraphs]), words, lines, hierosBounds)
    flatStr = addSmallCapsStyle(flatStr, words, tags)
    
    bodyHTML = f'<div style="display:flex;justify-content: center;";><div">{tagsAndTextToHTML(flatStr, tags)}</div></div>'
    styleBlock = "\n".join(styleBlocks)
    
    return f'<html>\n<head>\n<style>\n{styleBlock}\n</style>\n</head>\n<body>\n{bodyHTML}\n</body>\n</html>'

def main():
    hieroBounds = loadHieroBounds(sys.argv[3])
    paragraphs, lines, words, styles, page = loadData(sys.argv[1])
    print(res:=wordsToHTML(paragraphs, lines, words, styles, page, hieroBounds))
    f = open(f"{sys.argv[2]}", "w", encoding = 'utf-8')
    f.write(res)
    f.close()

if __name__ == '__main__':
    main()