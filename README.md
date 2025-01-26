# Amharic-Hate-Speech-Detection
Amharic Hate Speech Detection Using Facebook Posts: Achieved 95% training accuracy and 92% testing accuracy.

- * Google Colab Link * - https://colab.research.google.com/drive/1Cnk2ICDjtkBEEK6IvMsJPvBpLaaqQwUp?usp=sharing

## **1\. Introduction**

Hate speech detection is a critical task in ensuring safer online interactions, particularly for underserved languages such as Amharic. With the rise of social media and online platforms, the need for automated systems to detect and mitigate hate speech has become increasingly important. This project focuses on developing a machine learning model to detect hate speech in Amharic text using deep learning techniques, including word embeddings and Long Short-Term Memory (LSTM) networks. The goal is to create a robust and efficient model that can accurately classify Amharic text as hate speech or non-hate speech.

## **2\. Methodology**

### **2.1 Libraries and Dependencies**

The project leverages the following libraries and tools:

* **PyTorch**: For building and training the LSTM model.  
* **scikit-learn**: For metrics and dataset splitting.  
* **NLTK** and **Gensim**: For tokenization and word embedding creation.  
* **ONNX and ONNX Runtime**: For exporting and running the trained model efficiently.  
  **2.2 Data Preparation**

  #### **Dataset**

* The dataset consists of posts stored in `Posts.txt` and corresponding labels stored in `Labels.txt`.  
* Labels are categorized as:  
  * `1`: Hate speech.  
  * `0`: Non-hate speech.  
* Dataset Source: [https://data.mendeley.com/datasets/ymtmxx385m/1](https://data.mendeley.com/datasets/ymtmxx385m/1)

![][image1]

#### **Preprocessing**

* The text data was cleaned by removing unnecessary spaces and special characters.  
* Tokenization was performed to break the text into individual words using NLTK.  
  ![][image2]

  #### **Word Embedding**

* Word embeddings were created using Gensim's Word2Vec with a dimension of 100\.  
* The embedding matrix was saved to ensure consistency across training and inference.  
  ![][image3]

  #### **Vectorization**

* Each post was vectorized into a fixed-length sequence (e.g., 50 tokens) with zero-padding applied to handle varying text lengths.

![][image4]

We have also used the similarity measure to check the data

`word2vec.wv.similarity('ጠቅላይ', 'ሚኒስትሩ')`

Result \- 0.7488441

`word2vec.wv.similarity('ጠቅላይ', 'ወሳኝ')`

Result \- 0.20797253

`word2vec.wv.most_similar('ጠቅላይ')`

![][image5]

`word2vec.wv.most_similar('ህዝብ')`

![][image6]

**2.3 Dataset Splitting**

* The dataset was divided into three subsets:  
  * **Training**: 90% of the data.  
  * **Validation**: 10% of the data.  
  * **Test**: 10% of the data.  
* PyTorch's `DataLoader` was used for efficient batch loading during training and evaluation.  
  Training Data samples  
  ![][image7]  
  Validation Data Samples  
  ![][image8]

  Test Data Samples

  ![][image9]

  **2.4 Model Architecture**

A custom LSTM model was built for sequence classification with the following components:

* **LSTM Layer**: To capture temporal patterns and dependencies in the text.  
* **Batch Normalization**: To facilitate faster convergence and reduce overfitting.  
* **Dropout**: To prevent overfitting by randomly dropping neurons during training.  
* **Sigmoid Activation**: For binary classification (hate speech vs. non-hate speech).  
  ![][image10]  
  **2.5 Training and Validation**

  #### **Hyperparameters**

* **Learning Rate**: 0.01  
* **Batch Size**: 128  
* **Hidden Dimension**: 64  
* **Epochs**: 150  
* **Early Stopping Patience**: 10 epochs

  #### **Loss Function**

* **Binary Cross-Entropy Loss**: Used to measure the difference between the predicted and actual labels.

  #### **Optimizer and Scheduler**

* **Adam Optimizer**: For optimizing model parameters.  
* **Learning Rate Scheduler**: Reduces the learning rate when validation accuracy plateaus to ensure better convergence.

  #### **Training Loop**

* Training and validation accuracy were tracked per epoch to monitor performance.  
* Early stopping was implemented to halt training when validation accuracy did not improve for 10 consecutive epochs, reducing overfitting.

  ### **2.6 Exporting Model**

* The trained model was exported to the ONNX format for efficient deployment and inference in production environments.

## **3\. Results**

### **3.1 Test Set Performance**

* **Validation Accuracy :** The model achieved an accuracy of 93% on the validation set.  
* **Test Accuracy**: The model achieved a high accuracy of 91% on the test set.

![][image11]![][image12]

![][image13]

![][image14]

### **3.3 Model Robustness**

* The LSTM model demonstrated strong performance across both the validation and test datasets, showcasing its robustness in detecting hate speech in Amharic text.

# API LINK \-\> [https://nlp-flask.onrender.com/predict](https://nlp-flask.onrender.com/predict)

Since it was deployed on render free tier, the first request will often take 50 seconds but the rest won’t.  
Request Body format :  
`{`  
`"posts": ["ሚኪዬ ብዙ ጊዜ በአስደማሚ አርቲክሎችህ ተደምሚያለሁ ብስለትህ ስክነትህ የታላቁን የአማራ ህዝብ ስነ ልቦና ማወቅህ ከሌሎች ለየት ያደርጋሀል"]`  
`}`

Response format  
`{`  
    `"predictions": [`  
        `{`  
            `"post": "ሚኪዬ ብዙ ጊዜ በአስደማሚ አርቲክሎችህ ተደምሚያለሁ ብስለትህ ስክነትህ የታላቁን የአማራ ህዝብ ስነ ልቦና ማወቅህ ከሌሎች ለየት ያደርጋሀል",`  
            `"hate_content_probability_percentage": "0.19%",`  
            `"label": "Free"`  
        `},`  
        `{`  
            `"post": "ውድቅ መሆኑን አስታውሰው ተቋማቱ ጥያቄውን ውድቅ ያደረጉበት የተለያዩ መረጃዎች በድርድሩ ወቅት ለኩባንያው እንደሚቀርብ አክለዋል",`  
            `"hate_content_probability_percentage": "99.83%",`  
            `"label": "Hate"`  
        `}`  
    `]`  
`}`



