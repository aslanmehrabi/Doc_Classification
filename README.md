# Doc_Classification
Document Classification of New York Times Articles

Input data is related to the documents of New York Times (2000 - 2003) 
Each of the documents are related to one the following categories: News, Opinion, Classified, Features
A Classifier is developed here to classify the category of the articles based their context.


Fearure Extraction Methods: 
- Bag of Words
- TF_IDF
- Dimension Reduction

Classification Algorithms:
- SVM (Gaussian, Linear, and Sigmoid Kernel)
- Neural Networks (MLP)

Cross validation is used for tuning the parameters of each method.

Results of the Implementation:


| Classifier \ Feature extraction | tf_idf(1vsAll) | bagging(1vsAll) | Tf_idf (reduced dim) | Bagging(reduced dim)	| tf_idf |	bagging	|
|---------------------------------|----------------|-----------------|----------------------|-----------------------|--------|---------|
| SVM RBF kernel                  |    85.01	    |    83.58        |   	 61.15           |	     55.55	        |  65.95 |   66.99	|
| SVM linear kernel               |    83.21	    |    82.68	      |      23.5	           |       23.5           |	 61.2	 |   66.05	|
| SVM sigmoid kernel              |    85.01	    |    71.6	        |      61.2	           |       56.3	          |  64.7	 |   66.75	|
|NN MLP                          |      -        |      -          |      49.9            |       50.6			      |   -    |     -    |





