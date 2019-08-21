---
layout : posts

---



## What Is Customer Churn?

<figure>
        <center>
            <img src="{{ '/assets/churn_images/customer-churn-rate.jpg'}}">
        </center>
</figure>
    
<p style = "text-align: justify">
Customer attrition, also known as customer churn, customer turnover, or customer defection. Churn is one of the largest problems facing most businesses. Customer churn is when an existing customer, user, player, subscriber or any kind of return client stops doing business or ends the relationship with a company.
</p>

<p style = "text-align: justify">
The healthcare product companies, telephone service companies, beauty product companies, Internet service providers, insurance firms, often use customer attrition analysis and customer attrition rates as one of their key business metrics because the cost of retaining an existing customer is far less than acquiring a new one. Companies from these sectors often have customer service branches which attempt to win back defecting clients, because recovered long-term customers can be worth much more to a company than newly recruited clients.
</p>

<p style = "text-align: justify">
Due to the rapid growth in the mobile phone network and advancement in the telecommunication industries.There is need of surviving service in the competitive environment, and the retention of existing customers has become a huge challenge.
</p>

<p style = "text-align: justify">
The basic layer for predicting future customer churn is data from the past. Data contains customers that already have churned (response) and their characteristics before the churn happened. We will try to predict the response for existing customers. This method belongs to the supervised learning category.
</p>

<p style = "text-align: justify">          
Customer churn is binary classification problem and there are many more classification algorithm in machine learning used to solve churn problem. The logistic regression, decision tree, naive bayes, SVM, and random forest are the most popular binary classification algorithm in machine learning. 
</p>

## How Will We Predict Customer Churn?

<p style = "text-align: justify">
The basic layer for predicting future customer churn is data from the past. We look at data from customers that already have churned and their characteristics before the churn happened. By fitting a statistical model that relates the predictors to the response, we will try to predict the response for existing customers. This method belongs to the supervised learning category. In practice we conduct the following steps to make these precise predictions:
</p>

## Dataset

<p style = "text-align: justify">
One of the most valuable assets of a company has a data. As data is rarely shared publicly, we take an available dataset you can find on Kaggle Telcom Customer Churn Dataset. This dataset included 3034 unique customer records for a telecom company called Telco and each entry had gives information about the customer.

</p>

## Data looks like

<p style = "text-align: justify">
 To better understand data we will load it into pandas and shows how it looks using basic commands of pandas. 
 </p>

 <figure>
  <img src="{{ '/assets/churn_images/data_head.png'}}">
</figure>


## Churn Visualization

<p style = "text-align: justify">
The responce variable is <b>Churn</b> in the given dataset . Using pie chart shows how many percentage of churn in the given dataset. It shows the 14.5% total amount of chustomer churn in the data.
</p>

<img src="{{ '/assets/churn_images/churn_ratio.png'}}" class="center">
   

## Feature Engineering

<p style = "text-align: justify">
 Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. Feature engineering is fundamental to the application of machine learning. It is an important part in machine learning. It is very common to see categorical features in a dataset. However, our machine learning algorithm can only read numerical values. It is essential to encoding categorical features into numerical values.
</p>

<p style = "text-align: justify">
 Dropping irrevalent features, handlling missing values, converting categorical features into numerical features, scalling the dataset , and adding new features are happened in feature engineering.
</p>

1. **Feature Encoding**

<p style = "text-align: justify">
Better encoding of categorical data can mean better model performance. The encoding means converting "Yes" or "No" to 0 or 1 so that algorithm can work with data. There are many types of encoding in machine learning.
</p>
Here we will cover two different ways of encoding categorical features:
   1. LabelEncoder 
   2. OneHotEncoder

<p style = "text-align: justify">
    These two encoders are parts of the scikit Learn library in Python, and they are used to convert categorical data, or text data, into numbers, which our predictive models can better understand.
    If feature having two categories then LabelEncoder works well on that feature, and if feature having more than two categories then OneHotEncoder works well.
</p>

<figure>
        <center>
            <img src="{{ '/assets/churn_images/data_types.png'}}">
        </center>
</figure>

<p style = "text-align: justify">
    The above figure shows State, International_Plan, and Voice_Mail_Plan are categorical features in the dataset. So, this categorical features are converted into numerical using encoder.
</p>
<p style = "text-align: justify">
    Applying OneHot encoding on state feature because it contains more than two categories. Using get_dummies() converted it into numerical. Likewise other categorical features also converted into numerical.
</p>

The below figure shows how encoding performs.

<figure>
    <center>
        <img src="{{ '/assets/churn_images/Encoding_data.png'}}">
    </center>
</figure>

## Feature Scalling or Standardization

<p style = "text-align: justify">
 Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step. Sometimes, it also helps in speeding up the calculations in an algorithm.
</p>

<p style = "text-align: justify">
 This dataset contains features that highly vary in magnitude and range. Normalization should be performed when the scale of a feature is irrelevant or misleading and should normalize when the scale is meaningful.
</p>

<p style = "text-align: justify">
 Feature scalling helps to weight all the features equally and for normalization algorithm use the Euclidean Distance. If a feature in the dataset is big in scale compared to others then in algorithms where Euclidean distance is measured this big scaled feature becomes dominating and needs to be normalized. 
</p>

<img src="{{'/assets/churn_images/scalling.png'}}" class="center">

## Training And Testing Model

<p style = "text-align: justify">
We use sklearn, a Machine Learning library in Python, to create a classifier or model. Creating various models using diffrent algorithms like RandomForest, Logistic Regression, etc. and select best model from them. 
</p>
<p style = "text-align: justify">
We split the dataset to train (80% samples) and test (20% samples). We train the model and make predictions. With classification_report we calculate precision and recall with actual and predicted values. RandomForest model is the best among them on this data with 0.9610 accuracy on training data. 
The GridSearchCV or RandomizedSearchCv method used to find the best parameter for training model.
</p>  

<figure>
    <center>
        <img src="{{ '/assets/churn_images/model.png'}}">
    </center>
</figure>

## Model Performance Matrix

### Confusion Matrix
 
<p style = "text-align: justify">
    A confusion matrix is a table that is used to describe the performance of a classification model (or “classifier”) on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.
</p>

<figure>
        <center>
            <img src="{{ '/assets/churn_images/confusion_matrix.png'}}">
        </center>
</figure>

### Precision Score 

<p style = "text-align: justify">
    The precision can be calculated by dividing the total number of correctly classified positive examples by the total number of predicted positive examples. A model with high precision means few false positives. In other words, not many non-churners were classified as churners.
</p>

###  Recall
<p style = "text-align: justify">
    Recall can be defined as the ratio of the total number of correctly classified positive examples divide to the total number of positive examples. A model with high recall means it correctly classified most churners. 
</p>

### F1 Score
<p style = "text-align: justify">
    The f1_score is calculated using the precision and recall. F-measure which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more.  
</p>

<p style = "text-align: justify">
    The generate_report() is a helper function which shows the above terms.The below figure shows the precision, recall and f1_score on the telecom dataset.  
</p>
<figure>
        <center>
            <img src="{{ '/assets/churn_images/precision_recall.png'}}">
        </center>
</figure>

### AUC-ROC Curve
<p style = "text-align: justify">
    When we need to check or visualize the performance of the multi-class classification problem, we use AUC (Area Under The Curve) ROC (Receiver Operating Characteristics) curve. It is one of the most important evaluation metrics for checking any classification model’s performance.
</p>
<figure>
        <center>
            <img src="{{ '/assets/churn_images/AUC_ROC.png'}}">
        </center>
</figure>
