import json
import re
import sys

APPLY_BOLD = False
APPLY_TIMES_NEW_ROMAN = False

# load JSON data and break it into component parts
def loadData(fileName):
    file = open(fileName, encoding='utf-8')
    metaDict = json.loads(file.read())
    file.close()
    styles = metaDict['analyzeResult']['styles']
    paragraphs = metaDict['analyzeResult']['paragraphs']
    return paragraphs, styles

def inParagraph(line, paragraph):
    lineBounds = line['polygon']
    paragraphBounds = paragraph['boundingRegions'][0]['polygon']
    xInRange = lineBounds[0] >= min(paragraphBounds[0], paragraphBounds[6]) and \
        lineBounds[2] <= max(paragraphBounds[2], paragraphBounds[4])
    yInRange = lineBounds[1] >= min(paragraphBounds[1], paragraphBounds[3]) and \
        lineBounds[5] <= max(paragraphBounds[5], paragraphBounds[7])
    return xInRange and yInRange

def camelToSnekish(strCamel):
    #credit to https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    strCamel = re.sub(r'(?<!^)(?=[A-Z])', '-', strCamel).lower()
    return strCamel

def fixFontName(inputStr):
    if inputStr == 'similar-font-family':
        return 'font-family'
    return inputStr

def toCSSprop(propName):
    return fixFontName(camelToSnekish(propName))

def paragraphsToHTML(paragraphs, styles):
    tags = []
    ignoreKeys = {'confidence','spans'}
    styleBlocks = []

    for para in paragraphs:
        tags.append(('<p>&nbsp&nbsp&nbsp&nbsp',para['spans'][0]['offset']))
        tags.append(('</p>',para['spans'][0]['offset'] + para['spans'][0]['length']))
    
    for i,style in enumerate(styles[1:]):
        prop = list(set(style.keys()).difference(ignoreKeys))[0]#property
        
        #italics
        if style[prop] == 'italic' and prop == 'fontStyle':
            for span in style['spans']:
                tags.append((f'<i>', span['offset']))
                tags.append((f'</i>', span['offset'] + span['length']))
            continue
        
        styleString = f"{toCSSprop(prop)} : {style[prop]};"
        
        if style[prop] == 'bold' and prop == 'fontWeight' and style["confidence"] >= 0.995 and \
                APPLY_BOLD:
            for span in style['spans']:
                tags.append((f'<b>', span['offset']))
                tags.append((f'</b>', span['offset'] + span['length']))
            continue
        
        if "times new roman" in style[prop].lower() and style['confidence'] >= 0.5 and \
                prop == "similarFontFamily" and APPLY_TIMES_NEW_ROMAN:
            styleBlocks.append(f'.C{i} {{{styleString}}}')
            for span in style['spans']:
                tags.append((f'<span class = "C{i}">', span['offset']))
                tags.append((f'</span>', span['offset'] + span['length']))
            
    texts = [para['content'] for para in paragraphs]
    flatText = "\n".join(texts)
    tags.sort(key=lambda x:x[1])
    
    htmlStr = ""
    for i,tag in enumerate(tags):
        if i < len(tags) - 1:
            htmlStr = htmlStr + tag[0] + flatText[tag[1]:tags[i + 1][1]]
            # print(tag[0], tag[1], tags[i + 1][1], flatText[tag[1]:tags[i + 1][1]])
            continue
        htmlStr = htmlStr + tag[0] + flatText[tag[1]:]
        
    styleBlock = "\n".join(styleBlocks)
    
    return f'<html>\n<head>\n<style>\n{styleBlock}\n</style>\n</head>\n<body>\n{htmlStr}\n</body>\n</html>'

def main():
    paragraphs, styles = loadData(f"{sys.argv[1]}")
    res = paragraphsToHTML(paragraphs, styles)
    f = open(f"{sys.argv[2]}", "w", encoding = 'utf-8')
    f.write(res)
    f.close()

if __name__ == '__main__':
    main()