##Customer reviews -- Fuji x100f (mirrorless camera)

fileName =r'C:\Users\yinyaling\Desktop\fuji.reviews.csv'

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import data
from urllib.request import urlopen
from bs4 import BeautifulSoup #get the reviews
import pandas as pd  #dataframe and csv
import numpy as np
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


splitter = data.load('tokenizers/punkt/english.pickle')
sid = SentimentIntensityAnalyzer()

#visit the website and get reviews
urlString = r'https://www.bhphotovideo.com/c/product/1311229-REG/fujifilm_16534584_x100f_camera_silver.html'
url = urlopen(urlString)
soup = BeautifulSoup(url, 'html.parser') #pass url to soup

#creat an empty dictionary
commentAndSentiment = []

#get the reviews by find relative info on the page source
reviews= soup.find_all(itemprop='review')
for r in reviews:
    date=r.find(itemprop='datePublished')['content']
    author = r.find(itemprop='author').get_text().replace(',', '').replace('\n','')
    stars = r.find(itemprop='ratingValue').get_text().replace(',', '').replace('\n','')
    comment = r.find(itemprop='description').get_text().replace(',', '').replace('\n','')
    sentences = splitter.tokenize(comment) #split the comment into sentences
    sentimentTotal = 0
    sentenceCount = len(sentences)
    for c in sentences:
        ss = sid.polarity_scores(c)  #give back dictionary
        print(c)
        print(ss)
        sentimentTotal += ss['compound']
    sentimentTuple = (date,stars,author, comment,sentimentTotal/sentenceCount)
    commentAndSentiment.append(sentimentTuple) #add the sentimentTuple to the commentAndSentiment

#print the dictionary
for s in commentAndSentiment:
    print(s)
    
#create dataframe
commentFrame = pd.DataFrame(commentAndSentiment, columns=['date', 'stars','author','comment','sentiment average'])

commentFrame.to_csv(fileName)

#plot average sentiment by month #cast date column to datetime datatype and set it as the index
commentFrame['date'] = pd.to_datetime(commentFrame['date'])  
 
#cast sentiment column as float datatype
commentFrame['sentiment average'] = commentFrame['sentiment average'].astype(float)

 
##scatterplot of the sentiment average by date
x = commentFrame['date'].values
y = commentFrame['sentiment average'].values

#different sentiment have different color
colors = np.random.rand(len(x))

plt.scatter(x, y, c=colors, alpha=0.5)
plt.xlabel('Date')
plt.ylabel('Sentiment Average')
plt.title('Sentiment Average of Fuji x100f')
plt.colorbar()
plt.grid(True)
plt.show()
