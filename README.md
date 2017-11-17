# IRIS - Transparency

This project aims to explore 2 hypotheses:
- When applying Quantitative Input Influence (QII) in layer-by-layer fashion, a pattern of value would show up across levels of layers as well as neurons units within 1 layer. The context of experiment is any sophisticated model defined as inheritance of neural network and its variants.

- We are attempting to find another view of Dropout, in hope to figure out a potential explanation for this from game theoretic perspective.

## IRIS DATASET

IRIS dataset is famous for its popularity in AI literature. Though its simplicity, the data is a good starting place to observe the effect of new approach (at least back to 20 years ago) with its visualizability. This dataset consists of 150 samples of 3 classed of ``Iris flowers`` with 50 instances each, together with 4 features: ``Sepal Length``, ``Sepal Width``, ``Petal Length``, and ``Petal Width``. Authors claimed that one class is linearly separable from the others, whereas the latters are not.

To justify the correctness of the claim, I have make several plots cover all possible combinatorial sets of 1, 2 and 3 features. As it shows in Figure 1 and 2, *Iris Setosa* is entirely distinguishable when taking into account of merely ``Petal Length`` feature, but there is no clear cut boundary for the other 2, even in 3-dimensional feature space. It's worth noting that thanks to small number of features and value range, it is feasible to conduct sampling-based experimen on this dataset.

<p align="center">
  <img src="images/IRIS_plot_2D.png">
  <br><br>
  <b>Figure 1: Plot of singular feature and pair-wise features of 3 classes of instances</b><br>
</p>

<p align="center">
  <img src="images/IRIS_plot_3D.png">
  <br><br>
  <b>Figure 2: Plot of all possible combinations when chosing 3 out of 4 features</b><br>
</p>


They are numerical feature, not categorical

## NEURAL NETWORK ARCHITECTURE

For our goal is observing QII value distributions across different layers, the number of hidden layers is set to 3. In order to set up number of neural units, we based on several rules of thumb (which might or might not be helpful, a debatable topic):
1) Number of units in hidden layer must be in range [``input_nodes``, (``input_nodes``+``output_nodes``)*2/3]
2) Number of units is a sequence of non-increasing number. 
3) To handle non-linearly separable data, we integrate a non-linear function, namely, ``Rectified Linear Unit`` (*ReLU*), or ``f(x) = max(0, x)``.
4) We encode output as 1-hot vector, hence choosing *softmax* function as final filter of output, turn them into a pseudo-probability distribution. Therefore, it is natural to set *Cross-entropy* as loss function. The behind intuition is we are trying to minimize *Kullback-Leibler Divergence*, or disrepancy between predicted and expected probability distribution. 

Given above conditions, we construct a neural network has architecture as in Figure 3. It's worth noting that one can achieve 100% accuracy with following structures: ``1000`` > ``800`` > ``600``. However, with a simpler architecture, we can obtain approximated global optimum within 100 epochs.

<p align="center">
  <img src="images/NN_Architecture.png">
  <br><br>
  <b>Figure 3: Architecture of Neural Network used in experiment setting</b><br>
</p>


## TRANSPARENCY TEST


Quantitative Input Influence (*QII*) is a class of sampling method that approximately calculate influence for each feature by manipulating them and observe corresponding in the output. This method is model-agnostic, which does not make any assumption on prediction model, hence treat them as blackbox. 

The idea is borrowed from Cooperative Game, in a setting such that we have a set *N* of *n* agents, and utility function *f(x):N->R* that map any possible coalition to utility value. Depends on assumptions made, given any particular coalition, one can calculate marginal contribution for each agent in that coalition. Shapley method assumes that any agents must make a non-negative contribution to its belonging coalition, hence the gap between coalition with and without its existence must be non-negative as well.

Takes over from this, if one priorly knows a set of impactful features, which all sharing a positive or negative correlation (as long as they are in same direction), hence can use Shapley to calculate marginal contribution made by each feature in that chosen set. Pseudo-code below shows how to calculate influence score of feature *i* in feature set *S* in a particular sample *X*:

```python
QII = 0

for i in numb_samplings:
    sequence = permutation(S)
    feature_i = index(sequence)
    sequence_S = sequence[:feature_i-1]
    sequence_S_i = sequence[:feature_i] 
    QII = QII - (change(sequence_S_i) - change(sequence_S))

QII = QII/numb_samplings
```
The ``change`` function is tentative, which could be any numerical change factor in prediction. The feature set put into ``change`` would have its value drawn from other samples in training set, whereas the rest stays the same.

Now, we push it further by making one assumption that the model is no longer a blackbox, but a neural network model instead. By doing so, the black turns gray/white, hence we could explore how Shapley distributes in higher level of abstraction of feature, cause there is a common belief that the deeper the layer, the more meaningful the features are.

A few side notes:
- How to choose the set of impactful features?
- How to choose which individual data to test on?
- Neural Network is more biased in classification side. Pretraining with AutoEncoder might helps extracting meaningful feature, but leads to more problems as well. Unfortunately, we could not justify the meaningful of features in tabular datatype, cause it's lack of visualizability.


## RESULT
 
A handful and lively report of parameters (Weights/Biases) of the Neural Network could be shown by downloading the repository, opening Commandline and direct to the downloaded folder. Enter follow command and open browser with address `localhost:6006`. Voilà! 
```bash
tensorboard --logdir=log
```

So far, I choose the investigated sample is the first one in training set, and using exhaustive sampling (test on all permutation) as method to calculate Shapley value.

||Node1|Node2|Node3|Node4|Node5|Node6|Node7|Node8|
|-|-:|-:|-:|-:|-:|-:|-:|-:|
|Input Layer|0.094|0.302|0.202|0.058||||
|Hidden Layer 01|0.0|0.0|-0.776|24.344|0.0|0.344|16.224|1.624|
|Hidden Layer 02|0.111|0.501|0.081|0.0|0.0|0.0|
|Hidden Layer 03|0.588|0.0|-0.051|0.092|

## TODO LIST
- Make Neural Network deterministic by setting up pseudo random seed
- Reload the same weights for Neural Network is alternative solution
- Implement Banzhaf
- Implement Dropout (another Neural Network)

## ACKNOWLEDGEMENT

- Thank you Ben Hammer for his wonderful [plot tutorial](https://www.kaggle.com/benhamner/python-data-visualizations) of IRIS. 
- Thank you EdoVaira for his/her useful [tutorial](https://github.com/EdoVaira/Iris-Neural-Network/blob/master/Iris_Network.py) of training 1-hidden layer NN on IRIS dataset.
- OOP-style in designing [tensorflow graph](https://stackoverflow.com/questions/37770911/tensorflow-creating-a-graph-in-a-class-and-running-it-ouside).
- Determining number of [hidden units](http://www.faqs.org/faqs/ai-faq/neural-nets/part3/section-10.html) and their [ratio](https://www.kaggle.com/louisong97/neural-network-approach-to-iris-dataset).