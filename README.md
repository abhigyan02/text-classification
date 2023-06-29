# Text Classification with Naive Bayes

This project implements text classification using the Naive Bayes algorithm with the scikit-learn library. The code categorizes text documents into four predefined categories: alt.atheism, soc.religion.christian, comp.graphics, and sci.med. It is a useful tool for tasks such as sentiment analysis, topic categorization, and document filtering.

## Dataset

The code utilizes the 20 Newsgroups dataset, which is a collection of newsgroup documents grouped into different categories. The dataset provides a diverse range of text data for training and evaluation purposes.

## Dependencies

To run this code, the following dependencies are required:

- scikit-learn
- numpy

The code fetches the 20 Newsgroups training data, preprocesses it using CountVectorizer and TfidfTransformer, trains a Multinomial Naive Bayes classifier, and classifies a set of new documents.

## Results

The code prints the predicted categories for the new documents. Each prediction is shown alongside the corresponding input document.

![image](https://github.com/abhigyan02/text-classification/assets/75851981/74536fa1-940b-45cd-8b43-edcfaa1e9c6e)


