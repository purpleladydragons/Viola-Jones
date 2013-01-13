import pickle
import time
import numpy
from PIL import Image
import math
from decimal import *

ISLEARNING = False

getcontext.prec = 6
imagevecs = []
numberofnonfaces = 104 
numberoffaces = 44
numberofimages = numberofnonfaces + numberoffaces

rows = 24
cols = 24

#THIS IS ALL LEARNING----------------
#LEAVE DISABLED UNLESS YOU UPDATE TRAINING SET 



def vectortomatrix(vec,shape):
    mat = numpy.zeros((shape[0],shape[1]))
    for x in range(len(vec)):
        row = int(x/shape[1])
        col = x % shape[1]
        mat[row][col] = vec[x]

    return mat

def vectortotuplematrix(vec,shape):
    mat = numpy.zeros((shape[0],shape[1]),dtype=numpy.ndarray)
    for x in range(len(vec)):
        row = int(x/shape[1])
        col = x % shape[1]
        mat[row][col] = vec[x]
    
    return mat

def inverthaar(haar):
    newhaar = []
    for row in haar:
        newhaar.append([])
    for i in range(len(haar)):
        for j in range(len(haar[0])):
            newhaar[i].append(haar[i][j]*-1)

    return newhaar


def detThreshold(featurelist):
    ceiling = 0
    count = 0
    faces = []
    nonfaces = []
    for imagepair in featurelist:
        if imagepair[1] == 1:
            faces.append(imagepair[0])
            ceiling += imagepair[0]
            count += 1
        else:
            nonfaces.append(imagepair[0])

    ceiling /= count

    optimalvalue = 0
    optimalscore = 0

    for value in drange(0,ceiling,.1):
        facetotal = 0
        for face in faces:
            if face >= value:
                facetotal += 1
        nonfacetotal = 0
        for nonface in nonfaces:
            if nonface >= value:
                nonfacetotal += 1
        if facetotal - nonfacetotal > optimalscore:
            optimalvalue = value
            optimalscore = facetotal - nonfacetotal

    return optimalvalue

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def findBestClassifier():
    besterror = 100
    bestclassifier = 0
    for haar in haars:
        error = 0
        threshold = thresholds[haars.index(haar)]
        for i in range(len(haar[2])):
            imagepair = haar[2][i]
            if imagepair[1] == 1 and imagepair[0] < threshold:
                error += weights[i]
            if imagepair[1] == 0 and imagepair[0] >= threshold:
                error += weights[i]

        if error < besterror:
            besterror = error
            bestclassifier = haar

    return bestclassifier,besterror


def reweight():
    threshold = thresholds[haars.index(bestclassifier)]
    if besterror == 0:
        print 'no error'
        print bestclassifier
        print 'This is the',haars.index(bestclassifier)+1,'th classifier'
    else:
        p = .5 * math.log((1-besterror)/besterror)
        for i in range(len(weights)):
            print i, len(bestclassifier[2])
            if bestclassifier[2][i][1] == 1 and bestclassifier[2][i][0] < threshold:
                c = -1
            elif bestclassifier[2][i][1] == 0 and bestclassifier[2][i][0] >= threshold:
                c = -1
            else:
                c = 1
            
            weights[i] = weights[i] * (math.e ** (-1 * p * c))

        weightsum = sum(weights)
        for i in range(len(weights)):
            weights[i] /= weightsum #this normalizes the weights



def scanimage(image,strongclassifiers):
    """scan across an image using a 24x24 detector which contains classifier"""
    newimage = numpy.zeros((len(image),len(image[0])))
    for i in range(len(newimage)):
        for j in range(len(newimage[0])):
            newimage[i][j] = image[i][j]

    facelocs = []
    classifiers = []
    classifierpairs = []
    for a in range(len(strongclassifiers)):
        strongclassifier = strongclassifiers[a]
        classifiers.append([])
        classifierpairs.append([])
        for b in range(len(strongclassifier)):
            classifierpair = strongclassifier[b]
            classifierpairs[a].append(classifierpair)
            classifier = classifierpair[0]
            classifiers[a].append(classifier)
    detectors = []
    for i in range(len(classifiers)):
        detectors.append([])
        for j in range(len(classifiers[i])):
            classifier = classifiers[i][j]
            detectors[i].append([classifier[0],classifier[1],detThreshold(classifier[2])]) #previously zerodetector
    for i in range(0,len(image) - 23,1): #24x24 - 1 , and an adjustable step
        for j in range(0,len(image[0]) - 23,1):
            for a in range(len(detectors)):
                strongscore = 0
                for k in range(len(detectors[a])):
                    weakdetector = detectors[a][k]
                    threshold = weakdetector[2]
                    score = 0
                    xloc = weakdetector[1][1]
                    yloc = weakdetector[1][0]
                    for y in range(len(weakdetector[0])):
                        for x in range(len(weakdetector[0][0])):
                            score += image[i+y+yloc][j+x+xloc] * weakdetector[0][y][x]
                    if score >= threshold:
                        strongscore += classifierpairs[a][k][1] #a = strong #, k = feature of strong #, 1 is the p in featurepair
                    else:
                        strongscore -= classifierpairs[a][k][1]
                
                if strongscore > 0:
                    if a == len(strongclassifiers) - 1 :
                        facelocs.append((i,j))
                else:
                    break

    facelocs = integrateOverlaps(facelocs)        
    facelocs = integrateOverlaps(facelocs)        
    facelocs = integrateOverlaps(facelocs)        
    facelocs = integrateOverlaps(facelocs)        
    return facelocs

    
def integrateOverlaps(facelocs):
#this probably doesnt work as intended
#if one face overlaps with faceB, but not faceC, then faceB can still overlap with A and C
    newfacelocs = []
    for i in range(len(facelocs)):
        overlapping = [facelocs[i]]
        for j in range(len(facelocs)): #if the first face overlaps with a later one, then that later one will of course overlap with first
            if i == j:
                pass
            else:
                if abs(facelocs[i][0] - facelocs[j][0]) < 24: #but we dont care about that, we just want the one overall
                    if abs(facelocs[i][1] - facelocs[j][1]) < 24:
                        overlapping.append(facelocs[j])
        ys = [point[0] for point in overlapping]
        xs = [point[1] for point in overlapping]

        c1 = (sum(ys)/len(ys), sum(xs)/len(xs)) #top left
        #c2 = (sum(ys)/=len(ys), sum([x+23 for x in xs])/=len(xs)) #top right
        #c3 = (sum([y+23 for y in ys])/=len(ys), sum(xs)/=len(xs)) #bottom left
        #c4 = (sum([y+23 for y in ys])/=len(ys), sum([x+23 for x in xs])/=len(xs)) #bottom right

        newfacelocs.append(c1)                
    return newfacelocs

def matrixtovec(matrix):
    vec = []
    for row in matrix:
        for pixel in row:
            vec.append(pixel)

    return vec


def findfaces(colorimage,saveas='output.jpg'):

    width,height = colorimage.size

    colorimagematrix = numpy.array(colorimage.getdata())
    colorimagematrix = vectortotuplematrix(colorimagematrix,(height,width)) #fix size to be abstract

    greyimagematrix = numpy.array(colorimage.convert('L').getdata())
    greyimagematrix = vectortotuplematrix(greyimagematrix,(height,width))

    facelocs = scanimage(greyimagematrix,strongs)
    for point in facelocs:
        x = point[1]
        y = point[0]
        print y,x

        for i in range(24): #detector size
            colorimagematrix[y+i][x] = (255,0,0)
            colorimagematrix[y+i][x+23] = (255,0,0)
            colorimagematrix[y][x+i] = (255,0,0)
            colorimagematrix[y+23][x+i] = (255,0,0)


    color = matrixtovec(colorimagematrix)

    im = colorimage
    data = list(tuple(pixel) for pixel in color)
    im.putdata(data)
    im.save(saveas)

if ISLEARNING:
    try:
        haars = pickle.load(open('haars','r'))
        loaded = True
        print 'loaded haars each containing',len(haars[0][2]),'images'
    except:
        print 'failed to load haars'
        loaded = False

    if not loaded:
        for x in range(1,numberoffaces+1):
            im = Image.open('/home/austin/code/python/ai/violajones/f'+str(x)+'.jpg').convert('L').resize((24,24),Image.ANTIALIAS)
            imagevecs.append((numpy.array(im.getdata()),1))
        for x in range(1,numberofnonfaces+1):
            im = Image.open('/home/austin/code/python/ai/violajones/'+str(x)+'.jpg').convert('L').resize((24,24),Image.ANTIALIAS)
            imagevecs.append((numpy.array(im.getdata()),0))
    else:
        newfaces = numberoffaces - len([image for image in haars[0][2] if image[1] == 1]) #length of a list of scores for each image
        for x in range(numberoffaces+1-newfaces,numberoffaces+1):
            im = Image.open('/home/austin/code/python/ai/violajones/f'+str(x)+'.jpg').convert('L').resize((24,24),Image.ANTIALIAS)
            imagevecs.append((numpy.array(im.getdata()),1))
        newnonfaces = numberofnonfaces - len([image for image in haars[0][2] if image[1] == 0])
        for x in range(numberofnonfaces+1-newnonfaces,numberofnonfaces+1):
            im = Image.open('/home/austin/code/python/ai/violajones/'+str(x)+'.jpg').convert('L').resize((24,24),Image.ANTIALIAS)
            imagevecs.append((numpy.array(im.getdata()),0))


    images = []

    for x in range(len(imagevecs)):
        images.append( (vectortomatrix(imagevecs[x][0],(rows,cols)) , imagevecs[x][1]))
        #if x < numberoffaces:
         #   images.append( ( vectortomatrix(imagevecs[x],(rows,cols)) , 1 ))
        #else:
         #   images.append( ( vectortomatrix(imagevecs[x],(rows,cols)) , 0 ))

    haar1 = [[1,-1],
             [1,-1],
             [1,-1],
             [1,-1]]

    haar2 = [[1,1,1,1],
             [-1,-1,-1,-1]]

    haar3 = [[-1,1,-1],
             [-1,1,-1]]

    haar4 = inverthaar(haar1)
    haar5 = inverthaar(haar2)
    haar6 = inverthaar(haar3)

    originalhaars = [haar1, haar2,haar3,haar4,haar5,haar6]
    if not loaded:
        haars = []

        for haar in originalhaars:
            for i in range(rows -  (len(haar)-1) ): #for every row except the bottom stuff
                for j in range(cols - (len(haar[0])-1) ): #for every column except the far-right stuff
                    haars.append([haar, (i,j), []])   #for each possible haar location, note which original haar, its loc, and a list of future scores
                
    for haar in haars:
        for image in images:
            score = 0
            for i in range(haar[1][0], haar[1][0]+len(haar[0])):
                for j in range(haar[1][1], haar[1][1]+len(haar[0][0])):
                    score += haar[0][i-haar[1][0]][j-haar[1][1]] * image[0][i][j] 
            haar[2].append((score,image[1]))

    pickle.dump(haars,open('haars','w'))

    weights = [(1.00000/numberofimages) for x in range(numberofimages)]

    thresholds = [detThreshold(haar[2]) for haar in haars]
    bestclassifier,besterror = findBestClassifier()
    weaks = []
    for t in range(100):
        print 'on round', t
        reweight()
        bestclassifier,besterror = findBestClassifier()

        if besterror == 0:
            break

        p = .5 * math.log((1-besterror)/besterror)
        weaks.append((bestclassifier,p,besterror)) 
    
    sorted(weaks,key=lambda tup:tup[2])
    strong1 = weaks[:2]
    strong2 = weaks[2:5]
    strong3 = weaks[5:9]
    strong4 = weaks[9:15]
    strong5 = weaks[15:20]
    strongs = [strong1,strong2,strong3]

    pickle.dump(strongs, open('strongclassifiers','w'))
    print 'done training'
#END LEARNING-------------------------------------- 

strongs = pickle.load(open('strongclassifiers','r'))

family = Image.open('/home/austin/code/python/ai/violajones/family2.jpg')
findfaces(family,'output2.jpg')
print 'ON SMALLER NOW'
#family = Image.open('/home/austin/code/python/ai/violajones/family.jpg').resize((240,150),Image.ANTIALIAS)
#findfaces(family,'other.jpg')
