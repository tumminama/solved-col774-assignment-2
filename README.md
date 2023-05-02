Download Link: https://assignmentchef.com/product/solved-col774-assignment-2
<br>
<ol>

 <li><strong> Text Classification</strong></li>

</ol>

In this problem, we will use the Na¨ıve Bayes algorithm for classification of tweets by different twitter users. The dataset for this problem can be obtained from <a href="http://help.sentiment140.com/for-students">this website</a><a href="http://help.sentiment140.com/for-students">.</a> Given a user’s tweet, task is to predict the sentiment (Positive, Negative or Neutral) of the tweet. Read the website for more details about the dataset. The dataset contains separate training and test files containing 1.6 Million training samples and 498 test samples, respectively.

<ul>

 <li><strong> </strong>Implement the Na¨ıve Bayes algorithm to classify each of the tweets into one of the given categories. Report the accuracy over the training as well as the test set.</li>

</ul>

Notes:

<ul>

 <li>Make sure to use the Laplace smoothing for Na¨ıve Bayes (as discussed in class) to avoid any zero probabilities. Use <em>c </em>= 1.</li>

 <li>You should implement your algorithm using logarithms to avoid underflow issues.</li>

 <li>You should implement Na¨ıve Bayes from the first principles and not use any existing Python modules.</li>

</ul>

In the remaining parts below, we will only worry about test accuracy.

<ul>

 <li><strong> </strong>What is the test set accuracy that you would obtain by randomly guessing one of the categories as the target class for each of the review (random prediction). What accuracy would you obtain if you simply predicted the class which occurs most of the times in the training data (majority prediction)? How much improvement does your algorithm give over the random/majority baseline?</li>

</ul>

1

<ul>

 <li><strong> </strong>Read about the <a href="https://en.wikipedia.org/wiki/Confusion_matrix">confusion matrix</a><a href="https://en.wikipedia.org/wiki/Confusion_matrix">.</a> Draw the confusion matrix for your results in the part (a) above (for the test data only). Which category has the highest value of the diagonal entry? What does that mean? What other observations can you draw from the confusion matrix? Include the confusion matrix in your submission and explain your observations.</li>

 <li><strong> </strong>The dataset provided to you is in the raw format i.e., it has all the words appearing in the original set of tweets. This includes words such as ‘of’, ‘the’, ‘and’ etc. (called stopwords). Presumably, these words may not be relevant for classification. In fact, their presence can sometimes hurt the performance of the classifier by introducing noise in the data. Similarly, the raw data treats different forms of the same word separately, e.g., ‘eating’ and ‘eat’ would be treated as separate words. Merging such variations into a single word is called stemming.

  <ul>

   <li>Read about stopword removal and stemming (for text classification) online.</li>

   <li>Perform stemming and remove the stop-words in the training as well as the test data. You are free to use any libraries for this purpose.</li>

   <li>Remove Twitter username handles from the tweets. You can use nltk tokenizer package for this purpose. You are free to use any other libraries for this purpose also.</li>

   <li>Learn a new model on the transformed data. Again, report the accuracy.</li>

   <li>How does your accuracy change over test set? Comment on your observations.</li>

  </ul></li>

 <li><strong> </strong>Feature engineering is an essential component of Machine Learning. It refers to the process of manipulating existing features/constructing new features in order to help improve the overall accuracy on the prediction task. For example, instead of using each word as a feature, you may treat bi-grams (two consecutive words) as a feature. Come up with at least two alternative features and learn a new model based on those features. Add them on top of your model obtained in part (d) above. Compare with the test set accuracy that you obtained in parts (a) and parts (d). Which features help you improve the overall accuracy? Comment on your observations.</li>

 <li>TFIDF or tf–idf , short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. You can read more details about it from the link <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf">TF-IDF</a><a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf">.</a>

  <ul>

   <li>Your task now is to use TF-IDF features with Gaussian Naive Bayes Model to predict the sentiments of the given tweets. You can use Scikit-Learn’s TFIDFVectorizer and GaussianNB module for this purpose. How does your accuracy change over the test set? Comment on your observation.</li>

   <li>Selecting a smaller set of features is although not strictly necessary but it becomes handy when it is challenging to train a model with too many words or features. Use Scikit-learn’s SelectPercentile to choose features with highest scores. Report your accuracy over the test set. Does selecting smaller set of features improves your time taken to train the model? Compare the time taken in both the cases. You can use python’s time module for this purpose. Comment on your observation.</li>

  </ul></li>

 <li><strong> </strong>Receiver Operating Characteristic (ROC) metric is generally used to evaluate classifier output quality. ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis. ROC curves are typically used in binary classification to study the output of a classifier. In order to extend ROC curve and ROC area to multi-label classification, it is necessary to binarize the output. One ROC curve can be drawn per label, but one can also draw a ROC curve by considering each element of the label indicator matrix as a binary prediction (micro-averaging). Another evaluation measure for multi-label classification is macro-averaging, which gives equal weight to the classification of each label. Read more about ROC curves from the link <a href="https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc">Link-1</a><a href="https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc">.</a> Plot the suitable ROC curve as per the problem requirement and comment on your observation. You can use any available libraries of your choice for the same.</li>

</ul>