
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 19:10:13 2020

@author: Dell_1
"""

import nltk
import string

from os import listdir
from os.path import isfile, join

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# All text entries to compare 
BASE_INPUT_DIR = "C:\\Users\\arjav\\Desktop\\Text_Mining\\Project\\testfiles1"


#returns File list
def returnFilePathsList(folderPath):
    fileInfo = []
    FileNamesList = [fileName for fileName in listdir(folderPath) if isfile(join(folderPath, fileName))]
    FilePathsList = [join(folderPath, fileName) for fileName in listdir(folderPath) if isfile(join(folderPath, fileName))]
    fileInfo.append(FileNamesList)
    fileInfo.append(FilePathsList)
    return fileInfo

fileNames, filePaths = returnFilePathsList(BASE_INPUT_DIR)
#print(fileNames, "\n", filePaths)


# Get document contents
def create_docContentDict(filePaths):
    rawContentDict = {}
    for filePath in filePaths:
        with open(filePath, "r") as f:
            fileContent = f.read()
        rawContentDict[filePath] = fileContent
    return rawContentDict
rawContentDict = create_docContentDict(filePaths)
#print(rawContentDict)


def tokenizeContent(contentsRaw):
    tokenized = nltk.tokenize.word_tokenize(contentsRaw)
    return tokenized


def removeStopWordsFromTokenized(contentsTokenized):
    stop_word_set = set(nltk.corpus.stopwords.words("english"))
    filteredContents = [word for word in contentsTokenized if word not in stop_word_set]
    return filteredContents

def performPorterStemmingOnContents(contentsTokenized):
    porterStemmer = nltk.stem.PorterStemmer()
    filteredContents = [porterStemmer.stem(word) for word in contentsTokenized]
    return filteredContents

def stemming(contentsTokenized):
    stemmer= nltk.LancasterStemmer()
    filteredContents = [stemmer.stem(word) for word in contentsTokenized]
    return filteredContents

def removePunctuationFromTokenized(contentsTokenized):
    excludePuncuation = set(string.punctuation)    
    filteredContents = [word for word in contentsTokenized if word not in excludePuncuation]
    return filteredContents

def convertItemsToLower(contentsRaw):
    filteredContents = [term.lower() for term in contentsRaw]
    return filteredContents

# process data using PorterStemmer
def processData(raw_data):
    cleaned = tokenizeContent(raw_data)
    cleaned = removeStopWordsFromTokenized(cleaned)
    cleaned = performPorterStemmingOnContents(cleaned)    
    cleaned = removePunctuationFromTokenized(cleaned)
    cleaned = convertItemsToLower(cleaned)
    return cleaned


# process data using LancasterStemmer
def processData2(raw_data):
    cleaned = tokenizeContent(raw_data)
    cleaned = removeStopWordsFromTokenized(cleaned)
    cleaned = stemming(cleaned)    
    cleaned = removePunctuationFromTokenized(cleaned)
    cleaned = convertItemsToLower(cleaned)
    return cleaned


# print TFIDF values 
def print_TFIDF_values(term, values, fileNames):
    values = values.transpose() 
    numValues = len(values[0])
    print('                ', end="")   #bank space for formatting output
    for n in range(len(fileNames)):
        print('{0:18}'.format(fileNames[n]), end="")   
    print()
    for i in range(len(term)):
        print('{0:8}'.format(term[i]), end='\t|  ')     
        for j in range(numValues):
            print('{0:.12f}'.format(values[i][j]), end='   ') 
        print()

def print_Jaccard_Similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    print("\n===============Jaccard Similarity=============\n", len(intersection)/len(union))
    print('===============================================\n')


def print_CosineSimilarity(tfs, fileNames):
    #print(cosine_similarity(tfs[0], tfs[1]))
    print("\n\n\n========COSINE SIMILARITY=====================\n")
    numFiles = len(fileNames)
    names = []
    print('                   ', end="")    #formatting
    for i in range(numFiles):
        if i == 0:
            for k in range(numFiles):
                print(fileNames[k], end='   ')
            print()

        print(fileNames[i], end='   ')
        for n in range(numFiles):
            matrixValue = cosine_similarity(tfs[i], tfs[n])
            numValue = matrixValue[0][0]
            names.append(fileNames[n])
            print(" {0:.8f}".format(numValue), end='         ')
            #(cosine_similarity(tfs[i], tfs[n]))[0][0]
        print()
    print("\n\n=====================================\n")


def main():
    baseFolderPath = "C:\\Users\\arjav\\Desktop\\Text_Mining\\Project\\testfiles1"

    fileNames, filePathList = returnFilePathsList(baseFolderPath)

    rawContentDict = create_docContentDict(filePathList)

    query_doc1 = ([key for key in rawContentDict.values()][0])    
    compare_doc1 = ([key for key in rawContentDict.values()][1])    

    query_doc = processData2(query_doc1)
    compare_doc = processData2(compare_doc1)

    #Jaccard Similarity
    print_Jaccard_Similarity(query_doc,compare_doc)

    # calculate tfidf
    tfidf = TfidfVectorizer(tokenizer=processData)
    tfs = tfidf.fit_transform(rawContentDict.values())
    tfs_Values = tfs.toarray()
    tfs_Term = tfidf.get_feature_names()
    # print results
    print_TFIDF_values(tfs_Term, tfs_Values, fileNames)
    print_CosineSimilarity(tfs, fileNames)



main()
