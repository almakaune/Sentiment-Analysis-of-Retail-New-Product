# Sentiment-Analysis-of-Retail-New-Product
To develop market intelligence for Aritzia (a large North American women's clothing retailer) through analysis of customer reviews. Topic modelling is a useful method to analyze large text documents with an efficient and quantitative approach. Results have the intent of discovering hidden themes, or overall sentiment of the customer reviews of new product.

## Introduction
Social media is a vital tool for companies to better understand their customers, and be able to further improve their products and or services. As a result of this, text analysis has become an active field of research in computational linguistics and natural language processing. 

A popular problem in the mentioned field is topic modelling, a task which attempts to cluster documents into one or more classes. Latent Dirichlet Allocation, or LDA, is a topic model that generates topics based on word frequency from a set of documents. It considers each document as a collection of topics in a certain proportion.  

Topic modelling is basically a clustering problem. The output of a topic model will be a set of words that have high similarity measure. It is up to the person to infer the overall topic from the set of words.

<img src="https://github.com/almakaune/Sentiment-Analysis-of-Retail-New-Product/blob/main/images/topic-model.png" width = 40%>

## Research Questions

* Where in the industry is topic modelling used?
* What are different models for topic modelling?
* Which model are we selecting and why?
* How to optimize model through measuring coherence?
* What can we do with the results?

## Methodology

### 3.1 Data Collection
A data set of E-commerce reviews has been manually collected through a copy-and-paste style from the Aritzia website, under “New“ product. A .csv dataset was created that consisted of the columns brand, clothing, name, rating and review with a total of 1336 entries collected. 

### 3.2 Preparing text for statistical analysis
The following procedure was performed on the data in preparation for the LDA model. 

Reviews are broken into individual sentences. Apply part-of-speech tagging to retain only words that are adjectives, nouns or adverbs – that is, words that have information about the product of product quality (a process known as tokenization). Stem the words (ie convert to root form). Remove all stop words, such as unnecessary words, or words that may not be unique – eg “bought”, “typically”, “large”, “easy”, “look” ,etc. 

### 3.3 Machine Learning
The LDA model was imported through a Python library. As the primary focus of this study is on the implementation of the model and results, not much will be said about the theory behind the algorithm. Additional resources are provided at the end. 
 
 ## Results
 All experiments in this study were conducted on a laptop, using Spyder and Jupyter notebook. 

First, data was separated into two classes, positive reviews consisting of rating >=4 and negative reviews consisting of a rating <=3. A word cloud of two classes was generated and shown below.

Negative Word Cloud:
<img src="https://github.com/almakaune/Sentiment-Analysis-of-Retail-New-Product/blob/main/images/neg_word_cloud.png" width = 40%>

Positive Word Cloud:
<img src="https://github.com/almakaune/Sentiment-Analysis-of-Retail-New-Product/blob/main/images/pos_word_cloud.png" width = 40%>

A frequency distribution was created for number of reviews by brand, category and rating. The graph shows that the most reviewed brands are Wilfred Free, Wilfred and then TNA. Most reviewed categories include t-shirts, dresses and sweatshirts. Most reviews were rated a 5, which shows that overall the new products have been well received.

Reviews by Brand:
<img src="https://github.com/almakaune/Sentiment-Analysis-of-Retail-New-Product/blob/main/images/reviews_by_brand.png" width = 40%>

Reviews by Category: 
<img src="https://github.com/almakaune/Sentiment-Analysis-of-Retail-New-Product/blob/main/images/reviews_by_category.png" width = 40%>

Reviews by Rating:
<img src="https://github.com/almakaune/Sentiment-Analysis-of-Retail-New-Product/blob/main/images/reviews_by_rating.png" width = 40%>

### LDA Topic Model Results:

The graph below shows that there seems to be 4-5 clusters generated by the LDA model. Frequency graphs of words per cluster are displayed for review and further analysis. It is evident that some of the words are unintelligible – such as 'o', '6', and 'm'. More work to to effectively preprocess the dataset could be done to effectively remove such words. 

<img src="https://github.com/almakaune/Sentiment-Analysis-of-Retail-New-Product/blob/main/images/distance-map.png" width = 40%>

<img src="https://github.com/almakaune/Sentiment-Analysis-of-Retail-New-Product/blob/main/images/Topic-1.png" width = 40%>

## Conclusion

It is clear that the model was able to effectively cluster the reviews into 4 main groups, or topics. It is up to the audience to decide whether these results are able to provide greater context has to how the new products are being received by customers. 

## References
[1] https://link.springer.com/content/pdf/10.1007/s11573-018-0915-7.pdf

[2] Büschken J, Allenby GM (2016) Sentence-based text analysis for customer reviews. https://pdfs.semanticscholar.org/3be0/d1cb7dd15387c947dd919e6421be60d82b47.pdf?_ga=2.189142760.2112220891.1603322114-468455083.1603322114

[3] http://bmvh38.ust.hk/mark/files/staff/Ying/Ying-MS-2013.pdf

[4] https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2


