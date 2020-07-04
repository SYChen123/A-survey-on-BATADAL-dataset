# A survey on BATADAL dataset



## Contents

This survey consists of the following part.

- [A survey on BATADAL dataset](#a-survey-on-batadal-dataset)
  - [Contents](#contents)
  - [Dataset](#dataset)
  - [Summary of researches on BATADAL dataset](#summary-of-researches-on-batadal-dataset)
    - [BATADAL competition](#batadal-competition)
      - [1. Housh et al.](#1-housh-et-al)
      - [2. Abokifa et al.](#2-abokifa-et-al)
      - [3. Giacomoni et al.](#3-giacomoni-et-al)
      - [4. Brentan et al.](#4-brentan-et-al)
      - [5. Chandy et al.](#5-chandy-et-al)
      - [6. Posha et al.](#6-posha-et-al)
      - [7. Aghashahi et al.](#7-aghashahi-et-al)
    - [Other researches on BATADAL dataset](#other-researches-on-batadal-dataset)
      - [1. Abdulaziz Almehmadi.](#1-abdulaziz-almehmadi)
      - [2. Erba et al.](#2-erba-et-al)
      - [3. Taormina et al.](#3-taormina-et-al)
- [My experiment records](#my-experiment-records)
  - [Code](#code)
  - [Results](#results)
    - [ROC and AUC](#roc-and-auc)
    - [Acc, f1 score, precision and recall](#acc-f1-score-precision-and-recall)
    - [detection trajectory](#detection-trajectory)




## Dataset

BATADAL dataset is a multivariate time series dataset for anomaly detection in the context of water distribution system. 

This dataset is proposed in the publication below.

***Taormina R, Galelli S, Tippenhauer N O, et al. Battle of the Attack Detection Algorithms: Disclosing cyber attacks on water distribution networks[J]. Journal of Water Resources Planning and Management, 2018, 144(8).***

Details of this dataset can be found here https://github.com/scy-phy/www.batadal.net and here http://www.batadal.net/




## Summary of researches on BATADAL dataset


After that the water distribution systems(WDS) have upgraded from physical systems into cyber-physical systems, WDS is more vulnerable and susceptible to cyber attacks. More specifically, the element named supervisory control and data acquisition systems(SCADA) makes WDS more vulnerable to attacks. Hence there are more needs for developing powerful and reliable cyber-attack detection techniques or systems. Cyber-attacks detection techniques are either model-driven or data-driven.



### BATADAL competition

Here comes the research teams who took the BATADAL competition and introduction of their papers. The results of these research teams can be found on the official website of BATADAL. http://www.batadal.net/results.html


#### 1. Housh et al.

***Model-based approach for cyber-physical attack detection in water distribution systems***

Model-driven approach. Firstly estimate the demand based on partial SCADA readings. Secondly simulate the hydraulics based on the estimated demand using *EPANET*, a physically based water hydraulics simulation model. Next calculate the errors between SCADA readings and simulated values. Finally apply moving average and decision rules on the errors to find anomalies.


#### 2. Abokifa et al.

***Detection of Cyber Physical Attacks on Water Distribution Systems via Principal Component Analysis and Artificial Neural Networks***

Data-driven method. 3 layers framework. The first layer is to detect obvious anmalies from statistical perspective using mean and standard deviation. The second is multi-layer perceptron(MLP) used to detect contextual anomalies. The input data is noisy so FFT and third degree low pass Butterworth filter are applied to input data before being fed to MLP. The third layer is PCA, tranforming the principal components into 2 subspaces, normal subspace(the first 14 principal components) and anomaly subspace(the rest). Besides directly applying PCA, the authors also implement leave out one(LOO) algorithm which adds instances of interest to the data to see if it changes the direction of princial components.

In 2018, Abokifa et al. made some improvement on his previous aforementioned work. The improvement is illustrated in the paper ***Real-Time Identification of Cyber-Physical Attacks on Water Distribution Systems via Machine Learning Based Anomaly Detection Techniques***. They refine their idea from the following perspectives, 
    a. add one more module named actuator rules verification, which detect anomalies by checking if the rules are satisfied, e.g. the pump should be ON when the water level in the tank is below a minimum value.
    b. apply semi-supervised learning structure, adding the data that is predicted to be normal to the trusted dataset to retrain the model.
    c. Alarm watch window.


#### 3. Giacomoni et al.

***Identification of Cyber Attacks on Water Distribution Systems by Unveiling Low-Dimensionality in the Sensory Data***

Data-driven and Convex optimization-based approach.
The first step of the algorithm is to apply actuator rule verification and data verification(some rules set manually to identify whether the system is under attack). If there is no anomaly reported after the first step, then apply an optimization-based detection algorithm to detect attacks further. 


#### 4. Brentan et al.

***On-line cyber attack detection in water networks through state forecasting and control by pattern recognition***

Data-driven approach. Apply NARX model to predict the future state of the water distribution system. And then for a series error terms, which are obtained by computing the difference between prediction and measured data, use standard deviation of error terms to detect and localize possible anomalies.


#### 5. Chandy et al.

***Detection of cyber-attacks to water systems through machine-learning-based anomaly detection in scada data***

Data-driven approach. It's comprised of 2 parts. The first part is rule-based method, checking if the operation/physical rules are broken. The output is a set of flagged events. It's a bit like the verification step in the algorithm proposed by Giacomni et al. The second part is used to confirm the flagged events. It's composed of a convolutional variational autoencoder. Autoencoder is trained on normal data. Given a new data point, we can determine whether it's an anomaly by the reconstruction probability. 

In 2018, Chandy et al. refined their work by only using variational autoencoder and discarding the rule-based part in their previous work. The refinement is elaborated in the paper ***Cyberattack Detection using Deep Generative Models with Variational Inference***.


#### 6. Posha et al.

***An approach to detect the cyber-physical attack on water distribution system***

Data-driven approach. The method consists of 3 modules. The first one is to check if the data point is consistent with the control rules. The second one is pattern recognition. The model is trained on normal data to define normal behavior patterns. The third one is developed based on the relationships between the components of water distribution systems. This part is for confirming the attack events detected by the aforementioned modules.


#### 7. Aghashahi et al.

***Water distribution systems analysis symposium-battle of the attack detection algorithms (BATADAL)***

Data-driven approach. The algorithm is composed of 3 parts. The first part is to preprocess the data. It's actually to remove the time and flag attributes in the input .csv file. The second part is to train a random forest. Here is how they get input training data ready. Firstly compute average lag-0 covariance matrix A for normal series and average lag-1 covariance matrix B for anomalous series. Then for a time point t, compute lag-0 covariance matrix C for it. Finally the input data for RF is (||C-A||, ||C-B||). The third part is to test the model.



### Other researches on BATADAL dataset

In this section, some other researches using BATADAL dataset is shown below. Although these research teams didn't participate in the competition, their work has something to do with BATADAL dataset.

This section still needs to be completed.


#### 1. Abdulaziz Almehmadi.

***SCADA Networks Anomaly-based Intrusion Detection System***

Treat the intrusion detection task as binary classification. They firstly extract some features from SCADA readings manually and then apply PCA to select some crucial features. Finally train naïve bayes, svm and random forest separately to see which algorithm performs the best. And random forest can mitigate the impact that imbalanced data has so it performs the best compared with naïve bayes and svm.


#### 2. Erba et al.

***Real-time Evasion Attacks with Physical Constraints on Deep Learning-based Anomaly Detectors in Industrial Control Systems***

This paper is different from papers above since it focuses on the techniques to **evade detection** rather than detect anomalies. In the paper, the authors proposed white box and black box attack algorithms to evade detection of the system. It's an very interesting work. As for white box, since the assumption is attackers know how detection system is performed, they modify some variables of the anomalous data in order to craft hard-to-recognize anomalies. As for black box, they use adversarial trained autoencoder to generate anomalies with pertubations. In terms of evaluation, they use BATADAL dataset and WADI dataset.


#### 3. Taormina et al.

***Deep-Learning Approach to the Detection and Localization of Cyber-Physical Attacks on Water Distribution Systems***

Data-driven approach. They apply AutoEncoder to the BATADAL dataset to detect and localize the anomalies. In my following experiment on BATADAL, I use the source code of this paper to see how it performs on BATADAL dataset. 

	


# My experiment records


## Code

The code I use is downloaded from https://github.com/rtaormina/aeed. This is the source code for the following publication.

***Taormina, R. and Galelli, S., 2018. Deep-Learning Approach to the Detection and Localization of Cyber-Physical Attacks on Water Distribution Systems. Journal of Water Resources Planning and Management, 144(10), p.04018065.***

The anomaly detection model in the code is AutoEncoder with 7 layers. By checking if the reconstruction error is bigger than the threshold we set, we can tell whether it's an anomaly.

In training phase, it firstly splits BATADAL_dataset03 into train set and validation set. Next, the model is trained with early stop and learning rate decay.

In testing phase, it uses BATADAL_dataset04 and BATADAL_test_dataset for evaluation. Firstly set threshold as **quantile of average reconstruction error on validation set**. Next, compute accuracy, f1-score, precision, recall on these 2 test sets and draw the detection trajectory. Finally, draw the ROC curve and compute AUC under different window size.



## Results

The model is trained for 21 epochs on dataset03. The loss is 0.0011 on both train and val set.


### ROC and AUC

![roc_auc](./fig/roc_auc.png)

We can see that AUC becomes bigger as window size gets bigger. Window size 12 is the best. But if you care more about reducing the number of false alarms, then window size 3 is better, which can be concluded by the following comparison.


### Acc, f1 score, precision and recall

Window size = 1,
![1](./fig/metrics1.png)

Window size = 3,
![3](./fig/metrics3.png)

Window size = 6,
![6](./fig/metrics6.png)

Window size = 12,
![12](./fig/metrics12.png)


### detection trajectory

Window size = 3,
![trajectory_3](./fig/trajectory3.png)