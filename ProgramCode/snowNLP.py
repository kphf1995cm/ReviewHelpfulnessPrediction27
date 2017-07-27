from snownlp import SnowNLP
import time
import random
import textProcessing as tp

def storeReviewSenValue(dataSetDir,dataSetName,dataSetFileType,sheetNum,colNum,dstDir):
    start=time.clock()
    dataSetPath = dataSetDir + '/' + dataSetName + dataSetFileType
    dstPath = dstDir + '/' + dataSetName + 'SnowNLPSentiment.txt'
    reviewSet=tp.get_excel_data(dataSetPath,sheetNum,colNum,'data')
    reviewSentiment=[]
    for review in reviewSet:
        s=SnowNLP(review)
        reviewSentiment.append(s.sentiments)
    reviewNum=0
    f=open(dstPath,'w')
    for x in reviewSentiment:
        f.write(str(x)+'\n')
        reviewNum+=1
    f.close()
    end=time.clock()
    return reviewNum,end-start


reviewDataSetDir='D:/ReviewHelpfulnessPrediction\ReviewSet'
reviewDataSetName='HTC_Z710t_review_2013.6.5'
reviewDataSetFileType='.xlsx'
desDir='D:/ReviewHelpfulnessPrediction\ReviewDataFeature'
recordNum,runningTime=storeReviewSenValue(reviewDataSetDir,reviewDataSetName,reviewDataSetFileType,1,5,desDir)
#recordNum,runningTime=storeReviewSenValue("D:\ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature\SenimentReviewSet",'neg_review','.xlsx', 1,1,'D:/ReviewHelpfulnessPrediction\ReviewDataFeature')
print 'handle sentences num:',recordNum,' running time:',runningTime

