 

## CS772: Deep Learning for Natural Language Processing (DL-NLP)

This week's course content focuses on the task vs. technique matrix for DL-NLP. It covers classical ML, deep learning, perceptron, logistic regression, SVM, graphical models, dense FF with BP and softmax, RNN, LSTM, CNN, morphology, POS, chunking, parsing, NER, MWE, coref, WSD, machine translation, semantic role labeling, sentiment, and question answering.

Books and journals/conferences related to NLP, ML, and DL are also discussed.

The nature of DL-NLP is illustrated with POS tagging and neural parsing data. Classification decisions are also discussed, such as whether to insert brackets at a position, and if so, which type of bracket and label to insert. The steps for the first pass of the representation from two consecutive words are also outlined.


 

Recurrent Neural Network (RcNN) based Parsing:

- RcNN based parsing is a process of obtaining a parse tree of a sentence by combining bi-grams and passing them through a recurrent neural network. 
- The process begins by concatenating the vectors of the words and passing the concatenation through the recurrent neural network. 
- The resulting combination-unit is pre-multiplied by a learnt weight vector, the product added with a bias term, and the result passed through a non-linear function to obtain a score for the unit. 
- The highest scoring combination-unit is retained and a new sequence is obtained by deleting the word-units constituting the combination-unit. 
- The new sequence is treated like in the previous pass, combining bi-grams. 
- Retained combination-units also pass through a feedforward network with softmax final layer, to obtain the labels NP, VP, PP etc. 
- The process stops with the finding of the start symbol S. 
- An example is provided to illustrate the process. 
- The objective function of the neural parsing is also discussed.


 

# Perceptron Model
A perceptron is a computing element with input lines having associated weights and the cell having a threshold value. It is motivated by the biological neuron and its output is denoted as y. The parameter values (weights & thresholds) need to be found.

## Features of Perceptron
The input output behavior is discontinuous and the derivative does not exist at Σwixi= θ. Σwixi-θ is the net input denoted as net and is referred to as a linear threshold element due to linearity of x appearing with power 1.

## Computing Boolean Functions
The parameter values can be found by satisfying the inequalities for each Boolean function. For n=2, all functions except XOR and XNOR are computable. The number of threshold functions (2n2) becomes negligibly small for larger values of #BF.


# Perceptron Training Algorithm
Perceptron Training Algorithm (PTA) is a non-linear relation between y and net. It involves preprocessing steps to modify the computation law and absorb θ as a weight.

## Preprocessing
1. Modify computation law to: y = 1 if ∑wixi > θ, y = 0 if ∑wixi < θ
2. Absorb θ as a weight
3. Negate all zero-class examples

## Example to Demonstrate Preprocessing
Consider an OR perceptron with 1-class <1,1>, <1,0>, <0,1> and 0-class <0,0>. After preprocessing, the augmented x vectors are 1-class <-1,1,1>, <-1,1,0>, <-1,0,1> and 0-class <-1,0,0>. Negating the 0-class gives -<1,0,0>. The vectors are now:

X1 | X2 | X3
---|----|----
-1 | 0  | 1
-1 | 1  | 0
-1 | 1  | 1



 

## Perceptron Training Algorithm (PTA)

The PTA is a training algorithm that converges if the vectors used for testing are from a linearly separable function. The weight vector at the nth step of the algorithm is denoted as wn. The weight vector at the beginning is w0. When a vector Xj fails the test wiXj> 0, the weight vector is updated as wi+1= wi+ Xj.

The expression G(wn) is used to measure the convergence of the algorithm, which is defined as G(wn) = |wn| . |w*| . cosɵ/|wn|, where w* is the weight vector that satisfies w*Xj> 0 for all j. G(wn) is always less than or equal to |w*|, as -1 ≤ cosɵ≤ 1.

The numerator of G(wn) is always positive, as w*.Xi fail is always positive. This is true as long as | Xj| ≥δmin, where δmin is the minimum magnitude.


 

Perceptron Training Algorithm (PTA) Convergence:

• The numerator of G grows with n, while the denominator of G grows as n1/2. 
• This means that the numerator grows faster than the denominator. 
• If PTA does not terminate, G(wn) values will become unbounded, but this is impossible as |G(wn)| ≤ |w*|, which is finite. 
• Hence, PTA has to converge, as proven by Marvin Minsky. 
• PTA will converge if the vectors used for testing are from a linearly separable function, regardless of the initial choice of weights. 

Project Ideas:
• Semantics Extraction using Universal Networking Language
• Part Of Speech, Named Entity Recognition, Word Sense Disambiguation, Co-reference
• Combining Machine Learning with Rule Based Technique
• Sentiment Analysis (Tweets, Blogs, Indian Languages)
• Text Entailment


 

## Perceptron Model
A perceptron is a computing element with input lines having associated weights and the cell having a threshold value. It is motivated by the biological neuron and its output is denoted by y. The statement of convergence of PTA states that it converges if the vectors are from a linearly separable function.

## Sigmoid
The sigmoid neuron takes the input xi and the weights w1 and produces an output oi. The sigmoid function is also known as the logit function and can saturate in case of extreme agitation or emotion. It is used for decision making under sigmoid, where the output is between 0-1 and is looked upon as the probability of Class -1 (C1).

## Softmax
Softmax is used for multiclass classification and turns a vector of K real values into a vector of K real values that sum to 1. It is used to decide for the class which has the highest probability.


 
Softmax and Cross Entropy are intimately connected, with Softmax providing a vector of probabilities. The Winner-take-all strategy can be used with Softmax to make a classification decision. Consider an example where the Softmax vector is <0.09, 0.24, 0.65>, which corresponds to 3 classes (e.g. positive (+), negative (-), and neutral (0)). Training data is provided in the form of input sentences and the corresponding output vector. The difference between the target and obtained values is called the Loss, which can be calculated using either Total Sum Square Loss (TSS) or Cross Entropy. Cross Entropy measures the difference between two probability distributions and is used with Softmax.


 

## Cross Entropy Loss
Cross entropy loss is equivalent to multiplying probabilities and minimizing it is equivalent to maximizing the likelihood of observed data.

## Gradient Descent Approach
Gradient descent approach is used to minimize the loss. It involves the derivative of the input-output function for each neuron.

## Backpropagation Algorithm
Backpropagation algorithm is used to minimize the loss. It is the most important technique used in the course.

## Sigmoid and Softmax Neurons
Sigmoid neuron is represented by the equation: 

$net_i = \sum_{j=0}^m w_j x_j$

Softmax neuron is represented by the equation: 

$net_i = \sum_{c=1}^C w_c x_c$

## Notation
The notation used is: 
- i=1..N, N-o pairs, i runs over the training data
- j=0…m, m components in the input vector, j runs over the input dimension (also weight vector dimension)
- k=1…C, C classes (C components in


 

## Sigmoid and Softmax Neurons

Sigmoid neurons are used for two-class classification, where the output is interpreted as the probability of the class being 1. The target value for each input is either 1 or 0. The likelihood of observation is maximized by minimizing the cross entropy loss.

Softmax neurons are used for multi-class classification, where the output is interpreted as the probability of the class being c for the ith input. The target vector for each input is a vector of length C, where only one of the components is 1 and the rest are 0. The likelihood of observation is maximized by minimizing the cross entropy loss.


 

## Likelihood and Cross Entropy
The likelihood of observations in the case of softmax is given by the equation: 
$$\sum_{i=1}^{N} \log \frac{p(o_i|c)}{p(o_i)} = \log \frac{p(o|c)}{p(o)}$$

Maximizing the likelihood is equivalent to minimizing the cross entropy loss, which is denoted by $-LL$.

## Derivatives
The derivative of the sigmoid function is given by:
$$\frac{\partial}{\partial net_i} \sigma(net_i) = \sigma(net_i)(1-\sigma(net_i))$$

The derivative of the softmax function is given by:

Case 1: When the class $c$ for $O$ and $NET$ are the same:
$$\frac{\partial}{\partial net_i} \sigma(net_i) = \sigma(net_i)(1-\sigma(net_i))$$

Case 2: When the class $c'


 

## Genetic Algorithms
Genetic algorithms use probabilistic operators such as selection, crossover, and mutation.

## Single Sigmoid Neuron and Cross Entropy Loss
Cross entropy loss is derived for a single data point, dropping the upper right suffix. The change in any weight is learning rate * difference between target and observed outputs * input at the connection.

## Total Sum Square (TSS) Loss
TSS loss is used with a single neuron and sigmoid. The change in any weight is -ηδL/δw1, where η is the learning rate, L is the loss (½(t-o)2), t is the target, and o is the observed output.

## Softmax and Cross Entropy
Softmax and cross entropy can be generalized when E is cross entropy loss. The change in any weight is learning rate * difference between target and observed outputs * input at the connection.


 

## Backpropagation
Backpropagation is an algorithm used to train neural networks. It is used to calculate the gradient descent equations for the weights of the network. It is used to update the weights of the network in order to minimize the total sum square (TSS) loss. 

### Output Layer
The output layer consists of multiple neurons, which use a sigmoid activation function and the TSS loss. The target vector is <t1, t0> and the observed vector is <o1, o0>. The TSS loss is calculated as ½[(t1-o1)2+(t0-o0)2].

### Cross Entropy Loss
The cross entropy loss is the sum of the cross entropies over the instances. It is equivalent to multiplying probabilities and minimizing the total cross entropy loss is equivalent to maximizing the likelihood of observed data.

### Hidden Layers
The hidden layers consist of a fully connected feed forward network with no jumping of connections over layers. The gradient descent equations are used to calculate the value of j, which is propagated backwards to find the value of k. This recursion can give rise



 

## Information

- Information is a set of facts or data that can be used to make decisions.
- It serves as the basis for knowledge and understanding, and is the raw material for decision making.
- It is the foundation for knowledge and understanding.
n…o
w1

 

## Sigmoid Neuron
A sigmoid neuron is a single neuron with weights and inputs that are used to calculate an output. The weight change rule is calculated using the equation: $\Delta w_1 = \eta (t-o)x_1net$.

## Softmax Neuron
Softmax neurons are used in the output layer of a neural network and are used to calculate the probability of a given class. The weight change rule is calculated using the equation: $\Delta w_1 = \eta (t-o)x_1net$.

## Total Sum Square (TSS) Loss
TSS loss is used to calculate the weight change rule for a single neuron with a sigmoid activation function. The equation for the weight change rule is $\Delta w_1 = -\eta \frac{\delta L}{\delta w_1}$, where $\eta$ is the learning rate, $L$ is the loss, $t$ is the target, and $o$ is the observed output.

INFORMATION

 

Neural networks require numerical inputs to process information. Representations of text can be formed at different granularities, such as words, n-grams, phrases, and sentences. The general weight updating rule for neural networks is Δw = η(t-o)o(1-o)x1net, where η is the learning rate, t is the target vector, o is the observed vector, and x1net is the input. Word vectors are derived by setting the optimization equation to (1/2)∑∑∑(t-o)2, where t is the target vector and o is the observed vector. The gradient descent equation for deriving the word vector is Δuk = -∂L/∂uk.


 

This lecture discussed the concept of representation and how it is used to learn the nuances of words. Representation is a way of taking a discrete entity to a continuous space, and is used to take advantage of distance measures such as Euclidean and Riemannian distances. The lecture also discussed the importance of recognizing similarity and difference, which is the foundation of intelligence. Natural embedding of words is not intuitive, and the lecture discussed the use of 1-hot representation and collocations as a starting point. The learning objective is to maximize context probability. Finally, the lecture discussed cross-lingual mapping, which involves a strong assumption that embedding is the same across languages.


 

## Lexical and Semantic Relations in WordNet
Lexical and semantic relations are important for understanding language. Synonymy, hypernymy/hyponymy, antonymy, meronymy/holonymy, gradation, entailment, and troponymy are all examples of lexico-semantic relations. WordNet primarily focuses on paradigmatic relations, while ConceptNet focuses on syntagmatic relations. Substitutability is a foundational concept in linguistics and NLP, which states that words in paradigmatic relations can substitute each other in a sentential context.

## Learning Objective
The learning objective of NLP is to maximize the probability of getting the context words given the target (skip gram) and the probability of getting the target given the context words (CBOW).


 

Modelling P(context word|input word):
• We want to model the probability of a context word given an input word, e.g. P('bark'|'dog').
• To do this, we take the weight vector from the 'dog' neuron to the projection layer (udog) and the weight vector from the 'bark' neuron to the projection layer (vbark).
• These weight vectors give the initial estimates of the word vectors of 'dog' and 'bark'.
• The weights and word vectors are fixed by back propagation.
• To model the probability, we compute the dot product of udog and vbark, exponentiate it, and take the softmax over all dot products over the whole vocabulary.

Modelling p(wt+j|wt):
• Input to projection is a weight matrix W of size V x d, where V is the vocab size and d is the dimension.
• Each row gives the weight vector of dim d representing that word.
• From the whole projection layer, a weight vector of dim d is sent to each neuron in each compartment, where the compartment represents a context word.
• Each


 

## O-Occurence Matrix
O-occurence matrix is a fundamental concept in Natural Language Processing (NLP). It is also known as Lexical Semantic Association (LSA). The matrix is very sparse, with many 0s in each row. To reduce the dimensionality of the matrix, Principal Component Analysis (PCA) or Singular Value Decomposition (SVD) can be applied. This compression helps to capture better semantics.

## Linguistic Foundation of Word Representation by Vectors
The linguistic foundation of word representation by vectors is based on Harris' Distributional Hypothesis, which states that words with similar distributional properties have similar meanings. This hypothesis was further developed by Firth in the 1950s, who proposed that a word is known by the company it keeps. This means that differences in meaning can be modeled rather than the proper meaning itself.

## Skip Gram
Skip gram is a deep learning network that predicts context from a word. For example, given the input-output pair of "dog-cat" and "dog-lamp", the output vectors associated with "dog" will be more similar to the output vector of "cat" than "l


 

## Representation of Words

Words can be represented in different ways, such as 1-hot representation, co-occurrences, and learning the representation. 1-hot representation arranges the words in lexicographic order and sets the ith bit to 1 for the word in the ith position, with all other bits being 0. However, this does not capture many nuances, such as semantic similarity. Co-occurrences also do not fully capture all the facets, but are a good starting point.

## Embedding

Embedding is a way of taking a discrete entity to a continuous space. This allows us to take advantage of distance measures, such as Euclidean distance and Riemannian distance, to compute similarity. Recognizing similarity and difference is the foundation of intelligence, and is used in many areas, such as Pattern Recognition and Natural Language Processing.

INFORMATION:

 

## Lexical and Semantic Relations in WordNet
WordNet is a lexical database that contains lexico-semantic relations such as synonymy, antonymy, hypernymy, meronymy, troponymy, etc. It also contains syntagmatic and paradigmatic relations, which are primarily paradigmatic.

## Cross-Lingual Mapping
Cross-lingual mapping involves strong assumption that embedding spaces across languages are isomorphic, which is not true for distant languages. Without this assumption, unsupervised NMT is not possible.

## Substitutability
Words in paradigmatic relations can substitute each other in the sentential context. For example, "The cat is drinking milk" can be substituted with "The animal is drinking milk". Substitutability is a foundational concept in linguistics and NLP.


 

## Learning Objective
The goal is to maximize the probability of getting the context words given the target (skip gram) and the probability of getting the target given the context words (CBOW).

## Modelling P(context word|input word)
To model the probability, the dot product of the weight vector from the 'dog' neuron to the projection layer (udog) and the weight vector to the 'bark' neuron from the projection layer (vbark) is computed. The dot product is then exponentiated and softmaxed over the whole vocabulary.

## Exercise
The probability of 'bark' given 'dog' cannot be modelled as the ratio of counts of <bark, dog> and <dog> in the corpus. This way of modelling probability through dot product of weight vectors of input and output words, exponentiation and softmaxing works because it allows for more accurate estimates of the probability.

## Possible Project Ideas
Semantics Extraction using Universal Networking Language, Part Of Speech, Named Entity Recognition, Word Sense Disambiguation, Co-reference, Sentiment Analysis (Tweet and Blog, Indian Language, Word Sense). Current work involves combining Machine


 

## Information
- Information is a set of facts or data that can be used to make decisions and solve problems.
- It is a representation of knowledge that can be stored, retrieved, and manipulated.


## Word2vec and Glove
Word2vec and Glove are two methods of deriving word vectors from a corpus of text. Word2vec uses a neural network to learn the vector representations of words, while Glove uses a co-occurrence matrix to learn the vector representations. Both methods are used to model the probability of a context word given an input word.

### Deriving the Word Vector
The word vector is derived by setting up a projection layer with weights from the input word neuron to the projection layer (Udog) and weights from the projection layer to the output word neuron (Ubark). The weights and word vectors are then fixed by back propagation.

### Optimization
The probability of a context word given an input word is modeled by computing the dot product of Udog and Ubark, exponentiating the dot product, and taking the softmax over all dot products in the vocabulary.

### Gradient Descent
The word vector is further optimized by using gradient descent to calculate the change in the weights (Δuk).


 

Word2vec Architectures
Mikolov 2013 was a classic work that caught the attention of the world with equations like ‘king’-’man’+’woman ’=‘queen’. Word vectors capture syntagmatic relations in an N-dimensional space.

Syntagmatic and Paradigmatic Relations
Studies have found that when a subject hears a word, the words that come on hearing are 50% syntagmatic and 50% paradigmatic. On hearing ‘dog’, the words ‘animal’, ‘mammal’, ‘tail’ etc. are pulled as paradigmatic and ‘bark’, ‘friend’, ‘police’ etc. as syntagmatic.

Lexical Matrix
The lexical matrix is a fundamental device for representing word meanings. It consists of word forms and meanings, with examples of each.

Wordnet
Wordnet is a Princeton-developed resource for English that was released in 1992. Eurowordnet and IndoWordnet are linked structures of European language wordnets and an effort of 10 years respectively.

Basic Principle


 

Relational Semantics is a method of understanding the meaning of words by looking at how they are related to each other. Componential Semantics is an alternative approach which assigns features to words to represent them. WordNet is a database of lexical and semantic relations between words, such as synonymy, hypernymy/hyponymy, antonymy, meronymy/holonymy, gradation, entailment, and troponymy.

Synsets are created by following three principles: minimality, coverage, and replacability. For example, the words "house" and "home" can be combined to create a synset with the sense of a social unit living together. Representation using syntagmatic relations can be done using co-occurrence matrices.


 

## Collocation and Co-occurrence
Collocation refers to two or more words that tend to appear frequently together, such as "heavy rain" and "scenic view". Co-occurrence is a relation between two or more phenomena such that they tend to occur together, such as "thunder" and "lightning" or "bread" and "butter".

## Project Idea
A project idea is to detect oxymorons given a piece of text. An oxymoron is a figure of speech in which apparently contradictory terms appear in conjunction, such as "original copy" or "awfully good".

## Co-occurrence Matrix
Co-occurrence matrix is fundamental to NLP and is also called Lexical Semantic Association (LSA). It is very sparse, with many 0s in each row. To compress it, Principal Component Analysis (PCA) or Singular Value Decomposition (SVD) can be applied to do dimensionality reduction and merge columns with high internal affinity.

## GLOVE
GLOVE (Pennigton et al, 2014) is a model for learning word vectors. It has two main models: global matrix factorization methods


 

Word2vec Network

Matrix Factorization:
- Most frequent words contribute disproportionately to the similarity measure.
- Little semantic relatedness is conveyed.

Skip Gram & CBOW:
- Shallow window-based methods fail to take advantage of repetition in the data.

Example:
- 4 words: heavy, light, rain, shower.
- We want to predict heavy->rain and light->shower.
- Input is U and output is V.
- Language and domain impose constraints on what bigrams are possible.


 

## Word Vector Learning
Two main models for learning word vectors are global matrix factorization methods (e.g. LSA) and local context window methods (e.g. skip-gram). Both have drawbacks: matrix factorization methods are disproportionately affected by frequent words, while context window methods fail to take advantage of repetition in the data.

## Glove Architecture
Glove uses a representation based on syntagmatic relations, using a co-occurrence matrix. Dimensionality reduction is achieved through PCA. Intuition for this is demonstrated by a linear regression example, where a hyperplane is used to separate two classes of points.


 

Principal Component Analysis (PCA)

* Preliminaries:
  * Sample mean vector: <µ1, µ2, µ3,…, µp>
  * Variance for the ithattribute: σi2= [Σn j=1 (xij-µi)2]/[n-1]
  * Sample covariance: cab= [Σn j=1 ((xaj-µa)(xbj-µb))]/[n-1]
  * Standardize the variables: Replace the values by yij= (xij-µi)/σi2
  * Create the Correlation Matrix: R
* Eigenvalues and Eigenvectors:
  * AX=λX
  * Here, λs are eigenvalues and the solution <x1, x2, x3,… xp> 
  * For each λis the eigenvector
* Example: IRIS Data
  * Training: 80% of the data; 40 from each class: total 120
  * Testing: Remaining 30
  * Less


 

## Characteristic Function
The characteristic function is used to solve for the eigenvalues and eigenvectors of a matrix. An example is given to illustrate the process. The matrix is:

-9   4
7  -6

The characteristic equation is (-9-λ)(-6-λ)-28=0, which yields two real eigenvalues: -13 and -2. The corresponding eigenvectors are (-1, 1) and (4, 7) respectively.

## Principal Components
49 birds were studied, with 21 surviving and 28 dying in a storm. Five body characteristics were given: body length (X1), alar extent (X2), beak and head length (X3), humerus length (X4), and keel length (X5). The goal was to determine if the fate of the birds could be predicted from the body characteristics.

The eigenvalues and eigenvectors of the matrix R were calculated. The total variance in the data was 5, with the first eigenvalue accounting for 72% of the variance. The first principal component was determined to be the most important and sufficient for studying the classification.



 

Correlation in NLP Tasks
• Correlation is the crux of the matter for PCA
• Merging related attributes to form new attributes, co-occurrence matrix, POS tagging, parsing, and semantic graph are all examples of correlation in NLP

Difference between Explainability & Causality
• Explainability is a surface signal while causality is a deeper signal
• Example: A doctor knows that when body has jaundice it becomes yellowish, but the causal explanation is that liver malfunctioning released increased amount of Bilirubin which makes the urine yellow

PCA of Co-Occurrence Matrix
• Sum of eigenvalues is equal to the sum of diagonal elements
• Working out a simple case of word2vec: 4 words (heavy, light, rain, shower) with input-output prediction (heavy->rain, light->shower)

Note
• Actual probability of bigrams differs from theoretical possibility
• Language, domain, and corpus impose constraints on what bigrams are possible


 

Word2Vec Network

Word2Vec is a neural network that converts English statements into probabilities. It is illustrated with four words: heavy, light, rain, and shower. The weights go from all neurons to all neurons in the next layer. The net input to hidden and output layer neurons play an important role in backpropagation. The notation convention is to use capital letters for the name of the neuron and small letters for the output from the same neuron. The weights are indicated by small 'w' and the index close to 'w' is for the destination neuron, while the other index is for the source neuron.
 u2

 

Word2vec Network
• Word2vec network is a neural network used to represent words as vectors. 
• It consists of an input vector, output vector, and weights connecting all neurons in the next layer. 
• The weight vector from the input vector is called WU0, and the weight vector into the output vector is called WV0. 
• Dimensionality reduction is used to capture the commonality of distributional similarities across words. 
• The probability of a word being represented by a vector is calculated using softmax. 
• The goal is to minimize the negative log of the probability of a word being represented by a vector.
 WW  

 

## Word2Vec Network
Word2Vec is a neural network that uses weights to connect neurons in the input and output layers. The weights are adjusted based on the input and output values of the neurons.

### Weight Change Rule
The weight change rule for the output layer (V2) states that if the output (v2) is close to 1, the change in weight is small. The weight change for the input to the hidden layer (wH0U0) is positive if the input (U0) is ‘heavy’ and the output (V2) is not 1. The weight change for the same input (U0) to the output layer (V1) is negative. This ensures that the probability of ‘rain’ given ‘heavy’ increases while the probability of other outputs decreases.


 

# Word2Vec Network
Word2Vec is a neural network used to generate word embeddings. It consists of an input vector, an output vector, and weights that go from all neurons to all neurons in the next layer. The weights need to be updated for every input word, which can be computationally expensive.

## Efficiency Measures
To increase efficiency, hierarchical softmax and negative sampling are used. Additionally, the cross entropy softmax combination is a very ubiquitous combination in neural networks.

## Weight Change Rules
The weight change rules are derived from gradient descent and the cross entropy loss function. The change in weight is calculated by Δwji= -ηδL/δwji, where η is the learning rate, L is the loss, and wji is the weight of the connection from the ith neuron to the jth neuron. For a single neuron, the sigmoid and cross entropy loss functions are used.


 

## Feedforward Network
A feedforward network with 2 input, 2 hidden and 2 output neurons is discussed. The input neurons are X1 and X2, the hidden neurons are H1 and H2, and the output neurons are O1 and O2. H1 and H2 are RELU neurons, and O1 and O2 form a softmax layer.

## Weight Change Rules
The weight change rules are discussed. The general weight change equation is:

Δwji=ηδjoi

The weight change for the hidden layer is:

ΔW1,21=η(t1-o1)h21

The weight change for the output layer is:

ΔW2,11=η(t1-o1)h1

The weight change for the hidden layer is:

ΔW1,11=η[(t2-o2)W2,21+(t1-o1)W1,11].r’(H1).h1

## Vanishing/Exploding Gradient
The reason why RELU is a solution for vanishing or


 

## Vanishing/Exploding Gradient
Vanishing/Exploding gradients occur when the derivatives of the activation functions are multiplied together. This can lead to either a progressive attenuation of the product or an explosion of the gradient, depending on the value of the parameter K. For the sigmoid function, K needs to be greater than 1 to avoid saturation of the neurons. The same is true for the tanh function.

## Recurrent Neural Network
Recurrent Neural Networks (RNNs) are a type of machine learning model used for sequence processing. They have a branching factor (B) and a number of levels (L). The final expansion of δx1 will have BL terms, each of which is a product of L weights. Acknowledgement is given to sources such as Denny Britz, Jeffrey Hinton, and Dr. Anoop Kunchukuttan.


 
Neural Networks are composed of neurons, which have an activation output corresponding to an input. For example, the state vector for an XOR network is <h1, h2, o>. Part-of-Speech (POS) tagging is the process of assigning a part-of-speech to each word in a sentence. It is used in training data for ML-based POS tagging. There are three generations of POS tagging techniques: rule-based, statistical ML-based (Hidden Markov Model, Support Vector Machine), and neural (deep learning) based. The Noisy Channel Model is a generative model where words are observed from tags as states. Recurrent Neural Networks (RNNs) have two key ideas: summarizing context information into a single vector and recursively constructing the context. RNNs use an unbounded context for the function G, which requires all context inputs at once.


 

Recurrent Neural Networks (RNNs) are powerful models for sequence labelling tasks such as language modelling and part-of-speech tagging. RNNs use two inputs to construct a context vector: the context vector of the previous time-step and the current input. This context vector is then used to generate an output at each time-step. The same parameters are used at each time-step, so the model size does not depend on the sequence length.

The loss function used to train the model is the cross-entropy between the actual distribution and the predicted distribution. This is equivalent to maximizing the likelihood of the model. The model parameters are learned by minimizing the average cross-entropy over the entire corpus. Finally, the quality of the language model can be evaluated by measuring the likelihood of the model.


 

## Evaluating Language Models
Language models are evaluated based on their ability to predict the next word given a context and the probability of a test set of sentences. Standard test sets such as Penn Treebank, Billion Word Corpus, and WikiText are used to evaluate language models. Perplexity is a measure of difference between actual and predicted distribution, with lower perplexity and cross-entropy being better.

## RNN Variants
RNN models outperform n-gram models, with a special kind of RNN network, LSTM, performing even better.

## Importance of Probabilistic Language Modelling
In the past, context free grammar was used to determine if a given string of words was in the language or not. However, belongingness to language is not a black and white issue, as there are no grammatical and ungrammatical sentences, only sentences with probabilities. English has different forms through differences in regional dialects and even through periods of time.


 

## Introduction

Every year, new words and their different sentence positions are introduced, making it impossible to assign 0/1 values to sentences. However, probabilities can be assigned to word orders, which is equivalent to Prob (Wn | W 1,W2,...Wn -1).

## Backpropagation Through Time (BPTT)

BPTT is a method of training recurrent neural networks (RNNs) that is equivalent to feedforward networks. It involves a forward pass at each time step, followed by a backward pass to compute the error derivatives at each time step. After the backward pass, the derivatives at all the different times for each weight are added together.

## Vanishing/Exploding Gradient

Long word sequences can cause the vanishing/exploding gradient problem, where the gradient of the loss function becomes too small or too large to be useful. This can be seen in the famous sentence from Charles Dickens' "A Tale of Two Cities", which has 119 words. An example of this from NLP is the vanishing gradient problem in language models.


 

## Ram and Sitawill Go to University

Ram and Sitawill go to university for higher studies. They both live in London, a large metro. 

## Bank Closure

The bank that Ram used to visit 30 years before was closed due to the lockdown with the Govt. getting worried that crowding of people during the immersion ceremony.


 

## Information

- Information is a set of facts or data used to make decisions, form knowledge and understanding, solve problems, and make decisions.
- It is also the raw material for communication and learning.


 

Principal Component Analysis (PCA):

• 49 birds were studied, with 21 surviving a storm and 28 dying.
• 5 body characteristics were given: body length, alar extent, beak and head length, humerus length, and keel length.
• The goal was to predict the fate of the birds from the body characteristics.
• Eigenvalues and Eigenvectors of R were calculated, with the first eigenvalue being 3.616, representing 72% of the total variance.
• Principal components were formed, with Z1 and Z2 being the first two principal components.
• The new data was used to classify the birds, with the first bird having PC1 of 0.064 and PC2 of 0.602.
• A simple case of word2vec was worked out, with weights going from all neurons to all neurons in the next layer.


 

# Neural Network Weights

Neural networks are composed of neurons, which are connected by weights. The weights go from all neurons in one layer to all neurons in the next layer. The weights can be changed by computing the change in weights (Δw) for the output layer (V1) due to an input (U0) and for the input to the hidden layer (wH0U0).

The weights are calculated using the following equations:

* For output layer: HVHVwEw
* For input to hidden layer: HVHVwEw

The output of the hidden neuron (H0) is calculated using logarithmic functions. The net, e, and WWe equations are used to calculate the input vector (U) and output vector (V).

The Softmax, Cross Entropy, and RELU functions are used in FFNNs with O1-O2sof.


 

## Gradient Descent Rule and Weight Change Equation
Gradient Descent Rule is used to update the weights of the neural network. The General Weight Change Equation is used to calculate the change in weights. The equation is based on the Cross Entropy Loss and the derivative of the RELU activation function.

## Vanishing/Exploding Gradient
The vanishing/exploding gradient problem occurs when the derivatives of the RELU activation function are either 1 or 0, which can cause the gradients to either vanish or explode.

## Recurrent Neural Networks
Recurrent Neural Networks (RNNs) are used for sequence processing and POS tagging. They are used to generate lexical and bigram probabilities.

## Hidden Markov Models
Hidden Markov Models (HMMs) are generative models that use words observed from tags as states. They are used to calculate the probability of a word given its context.


 

## AI Search and Decoding
AI search and decoding involves two approaches: rule/knowledge based and data/ML based. Rule/knowledge based search includes BFS, DFS, Djikstra, A*, and A* algorithms. Data/ML based search includes Viterbi and Beam search. AI is the science and technology of making computers good at tasks that living beings perform effortlessly, such as understanding scenes, language, driving a car, and identifying a person from a picture. AI is highly data driven and cannot give a theory, but instead lets the data give a model.

Planning involves searching for the right sequence of actions, such as which block to pick, stack, or unstack. Computer vision involves searching for which point in the image of the left corresponds to which point in the right. Robot path planning involves searching for the optimal path with the least cost. Natural language processing involves searching among many combinations of parts of speech to decipher the meaning.


 

## Expert Systems
Expert systems are used to search among rules, many of which can apply to a situation. They are composed of four components: state space, operators, start state, and goal state. Cost is also taken into account when using an operator.

## Examples
Two examples are given to illustrate the use of expert systems. The first is an 8-puzzle, where tile movement is represented as the movement of the blank space. The operators used are L (blank moves left), R (blank moves right), U (blank moves up), and D (blank moves down). All operators have a cost of 1. 

The second example is the Missionaries and Cannibals problem. The state is represented as <#M, #C, P>, where #M is the number of missionaries on the left bank, #C is the number of cannibals on the left bank, and P is the position of the boat. The operators used are M2 (two missionaries take the boat), M1 (one missionary takes the boat), C2 (two cannibals take the boat), C1 (one cannibal takes the boat), and MC (one missionary and one cannibal take the boat


 

## General Graph Search (GGS)
GGS is a general umbrella algorithm for AI search problems. It consists of 9 steps:

1. Create a search graph G, consisting solely of the start node S; put S on a list called OPEN.
2. Create a list called CLOSED that is initially empty.
3. Loop: if OPEN is empty, exit with FAILURE.
4. Select the first node on OPEN, remove from OPEN and put on CLOSED, call this node n.
5. If n is the goal node, exit with SUCCESS with the solution obtained by tracing a path along the pointers from n to s in G. (pointers are established in step 7).
6. Expand node n, generating the set M of its successors that are not ancestors of n. Install these memes of M as successors of n in G.
7. Maintain the least cost path and node in OPEN: to Establish a pointer to n from those members of M that were not already in G (i.e., not already on either OPEN or CLOSED). Add these members of M to OPEN. For each member of M that was already on OPEN or CLOSED


 

## A* Algorithm
A* is an algorithm that uses an optimistic approach to find the optimal path from a given node to a goal node. It uses a function f(n) = g(n) + h(n) to determine the node with the least value of f from the open list. g(n) is the actual cost of the optimal path from the start node to the current node, and h(n) is the actual cost of the optimal path from the current node to the goal node.

For example, in the 8-puzzle, h*(n) is the actual number of moves to transform the current node to the goal node. Heuristics h1(n) and h2(n) can be used to estimate h*(n). h1(n) is the number of tiles displaced from their destined position, and h2(n) is the sum of Manhattan distances of tiles from their destined position. h1(n) and h2(n) are both less than or equal to h*(n).

A* is admissible if it always terminates and terminates optimally. The key point about A* search is that


 

## Admissibility of A*
A* is an algorithm that is admissible, meaning it always terminates finding an optimal path to the goal if such a path exists. This is proven by the Lemma that states that at any time before A* terminates, there exists a node in the open list such that f(n) <= f*(s). Additionally, the path formed by A* is optimal when it has terminated, which is proven by showing that if the path formed is not optimal, then the f-value of the nodes expanded would become unbounded.


 

AI Search Algorithms
•A*: Finds optimal path. If f(n) < f*(S) then node nhas to be expanded before termination. If A* does not expand a node nbefore termination then f(n) >= f*(S).
•AO*
•IDA* (Iterative Deepening)
•Minimax Search on Game Trees
•Viterbi Search on Probabilistic FSA
•Hill Climbing
•Simulated Annealing
•Gradient Descent
•Stack Based Search
•Genetic Algorithms
•Memetic Algorithms

Exercise -1
•If the distance of every node from the goal node is “known”, then no “search” is necessary.
•Lemma proved: any time before A* terminates, there is a node min the OL that has f(m) <= f*(S).
•For m, g(m)=g*(m) and hence f(m)=f*(S).
•Thus at every step, the node with f=f* will be picked up, and the journey to the goal will


 

Hidden Markov Models (HMM)
- HMM is a k-order machine that combines lexical and transition probabilities through the product operation (Markov independence assumption).
- RNN is an infinite memory machine and is more general than HMM.
- Softmax operation in RNN encompasses both lexical and transition probabilities.

Calculations
- Most sentences start with a noun, so P(N|^)=0.8, P(V|^)=0.2 and P(‘^’|^)=1.0.
- Transition from N to N is less common than to V.
- Transition from V to V (auxiliary verb to main verb) is quite common.
- V to N is also common (e.g. going home).
- Lexical probabilities: P(‘people’|N )=0.01, P(‘ people’|V )=0.001.

Trellis
- Columns of tags on each input word with transition arcs going from tags (states) to tags in consecutive columns and output arcs going from tags to words (observations).
- Aim is


 

HMM-based POS Tagging
- HMM-based POS tagging cannot handle free word order and agglutination well
- Transition probability is no better than uniform probability which has high entropy and is uninformative
- POS tagging without morphological features is highly inaccurate

Modelling in Discriminative POS Tagging
- T* is the best possible tag sequence
- Tag at any position depends only on the feature vector at that position

Feature Engineering
- Word-based features: dictionary index of current, previous and next word
- POS tag-based feature: index of POS of previous word
- Morphology-based features: does current word have a suffix?

Beam Search Based Decoding
- We can give equal probabilities to sentences ending in noun and verb
- Also, 'dance' as verb is more common than noun


 

Modelling Equations:
The Maximum Entropy Markov Model (MEMM) is a sequence probability model which uses a set of tags to calculate the product of P(ti|Fi) for each position in the sequence.

Word Vectors:
Word vectors are vectors of numbers representing words, and are created from huge corpora. It is not possible to tell which component in the word vector does what.

Beam Search Based Decoding:
Beam search based decoding is a method of decoding which uses a beam width (an integer which denotes how many of the possibilities should be kept open) to retain the top two paths in terms of their probability scores. The actual linguistically viable sub-sequence appears amongst the top two choices. For example, when the word ‘the’ is encountered, the two highest probability paths for “^ The” are P1 and P3.


 

Neural Decoding Encode-Decode Paradigm is a method of using two RNN networks, the encoder and the decoder, to process and generate a representation of a sentence. This is used in English POS tagging with Penn POS tag set, which has approximately 40 tags and allows for 4 finer category POSs under each category. The beam width for POS tagging for English using Penn tagset is 12 (=3 X 4). In order to retain two paths, the beam width is set to 2. This allows for the correct path to be probabilistically chosen in the beam. 'Brown' is an example of a word that can be both an adjective and a noun, and can also be both a past participial adjective and a verb.


 

Sequence to Sequence Learning with Neural Networks

Sequence to sequence learning is a method of machine translation that uses neural networks to map a source sentence to a target sentence. This is done by encoding the source sentence into a single vector and then decoding it using four influencing factors: input encoding, autoregression, cross attention, and self attention. All searching is done through table lookup, which is equivalent to mapping.

Structural AI vs Functional AI

Structural AI is concerned with understanding the anatomy of a system, while functional AI is concerned with understanding the behaviour of a system. Structural AI is analogous to medicine, where doctors use graphs like EEG to understand the anatomy of a system. Functional AI is analogous to medicine, where attributes like facial expression, body language, and pain are used to understand behaviour.

RNN-LSTM

RNN-LSTM is used to capture y<j and x, c=h4. The decoder is doing at each time-step is decoding, which is done by incrementally constructing hypotheses and scoring them using the model. Hypotheses are maintained in a priority queue.

Problems

The representation used is insufficient to capture


 

Encoder-Attend-Decode Paradigm:
- Source sentence representation is not useful after few decoder time steps.
- Solution: Make source sentence information available when making the next prediction.
- Even better, make relevant source sentence information available.
- Represent the source sentence by the set of output vectors from the encoder.
- Each output vector at time t is a contextual representation of the input at time t.

Convolutional Neural Networks (CNN):
- Motivation points: Reduced number of parameters and stepwise extraction of features.
- Applicable to any AI situation, not only vision and image processing.
- Feedforward-like and recurrent-like.
- Input divided into regions and fed forward.
- Window slides over the input.
- Genesis: Neocognitron (Fukusima, 1980).
- Inspiration from biological processes.
- Connectivity pattern between neurons resembles the organization of the animal visual cortex.
- Receptive fields of different neurons partially overlap.

Convolution Basics:
- Continuous and discrete convolution.
- Area under the curve weighted by filter parameters.


 

Convolutional Neural Networks (CNNs) are a type of neural network used for feature extraction. They use four key ideas to take advantage of the properties of natural signals: local connections, shared weights, pooling, and the use of many layers. A typical CNN architecture consists of several layers of convolution with tanh or ReLU applied to the results. The filter components are LEARNT and the sliding of the filter corresponds to taking different receptive fields. By designing the filter as < 0,1,0 >, we emphasise the center of the receptive field “dog” image and “cat” image. The filter should have the ability of detecting this kind of feature. ImageNet, a database of 1,000 different classes of images from the web, was used to achieve spectacular results, almost halving the error rates of the best competing approaches.


 

# Encode-Decode Paradigm
Encode-Decode Paradigm is a method of sequence-to-sequence learning with neural networks. It uses two RNN networks, an encoder and a decoder, to process and generate one element at a time. The encoder processes one input at a time and generates a representation of the sentence, which is used to initialize the decoder state. The decoder then generates one element at a time until the end of sequence tag is generated.

# Problems
The representation generated by the encoder is insufficient to capture all the syntactic and semantic complexities. Long-term dependencies are also a problem, as the source sentence representation is not useful after a few decoder time steps.

# Solutions
To address these problems, a richer representation for the sentences should be used. Additionally, source sentence information should be made available when making the next prediction, and relevant source sentence information should be used.


 

## Introduction to Convolutional Neural Networks (CNNs)

CNNs are a type of AI model that combines feedforward and recurrent-like features. They are inspired by biological processes, and the connectivity pattern between neurons resembles the organization of the animal visual cortex.

## Motivation

Two main motivations for using CNNs are:
- Reduced number of parameters
- Stepwise extraction of features

## Convolution Basics

Convolution is a process of combining two vectors, V1 and V2, to produce a third vector, V1V2. This is done by taking the area under the curve of V1 and V2, weighted by V2.

## Receptive Field and Selective Emphasis/De-emphasis

The filter <1,1,1> is used to emphasize or de-emphasize certain features in the input. This is done by sliding the window over the input, while keeping the filter parameters the same. This is similar to the concept of RNNs.


 

## Convolutional Neural Networks
Convolutional Neural Networks (CNNs) are a type of neural network used for image classification. They use filters to highlight features and obtain those features by sliding the filter over the input layer. This resembles sequence processing, and the filter components are learnt. A typical CNN consists of several layers of convolution with tanh or ReLU applied to the results, and local connections, where each region of the input is connected to a neuron in the output.

Four key ideas are used to take advantage of the properties of natural signals: local connections, shared weights, pooling, and the use of many layers. Learning in CNNs is automated, and the values of its filters are learnt. For example, in image classification, the filters learn to detect edges from raw pixels in the first layer, then use the edges to detect simple shapes in the second layer, and then use these shapes to detect higher-level features, such as facial shapes.

ImageNet is a million-image dataset from the web with 1,000 different classes, which has resulted in spectacular results and almost halving the error rates of the best competing approaches.


 

## Convolutional Neural Networks (CNNs)
CNNs are used for image recognition and natural language processing (NLP). They use a matrix of word vectors as input and multiple filters to capture different views and emphasis angles for each task. Lower order ngrams are important for vocabulary matching, while higher order ngrams capture syntactic structure and dependencies.

## Hyperparameters
The hyperparameters of a CNN include narrow width vs. wide width, stride size, pooling layers, and channels.

## Detailing Out CNN Layers
The stages of a CNN include input matrix, multiple filters, lower order ngrams, higher order ngrams, and hyperparameters.


 

## Convolutional Neural Networks

Convolutional neural networks (CNNs) are a type of deep learning algorithm used for image recognition and classification. They are composed of several layers, including convolution layers, pooling layers, and fully connected layers.

### Tensors and Vectors

Tensors are vectors of vectors, while vectors are a single row of numbers. Images are composed of three channels: red, green, and blue.

### Pooling Layer

The pooling layer is used to summarize the features of a feature map. It involves sliding a two-dimensional filter over each channel of the feature map. The output dimension after pooling is determined by the height and width of the feature map, the number of channels, the height and width of the filter, and the stride length.

### Learning in CNN

The weights of the kernels are adjusted using backpropagation from the final layer of softmax. The kernel values are parameter-shared, and no special steps are needed for the RELU and MAX functions.


 

## Sentiment and Emotion Analysis
Sentiment Analysis is the task of identifying opinion, emotion, or other forms of affective content in a piece of text. Machine Learning based approaches use classifiers such as SVM, KNN, and Random Forest, as well as sentiment-based features such as the number of positive and negative words, and highly emotional words. Emotion features include positive and negative emoticons, and boolean features that indicate the presence of both positive and negative words. Punctuation features include the number of exclamation marks, dots, question marks, capital letter words, and single quotations.

## Deep Learning based Approach
Deep Learning based approaches use CNN-FF models with very little feature engineering. Embedding size is 128, maximum tweet length is 36 words, and filters of size 3, 4, and 5 are used to extract features.

## Sentiment Annotation and Eye Movement
Sarcastic tweets have longer fixations and multiple regressive saccades. Two publicly available datasets have been released by Mishra et al. (2016; 2014).


 

## Learning Cognitive Features from Gaze Data for Sentiment and Sarcasm Classification

Two datasets were used to evaluate the performance of the model:

- **Dataset 1**: 994 text snippets (383 positive and 611 negative, 350 sarcastic/ironic) from movie reviews, tweets, and quotes. Annotated by 7 human annotators with an annotation accuracy of 70%-90% and Fleiss kappa IAA of 0.62.

- **Dataset 2**: 843 snippets (443 positive and 400 negative). Annotated by 5 human subjects with an annotation accuracy of 75%-85% and Fleiss kappa IAA of 0.68.

The model used a Convolutional Neural Network (CNN) to learn features from gaze sequences (fixation duration sequences and gaze positions) and text. The CNN had two channels: a non-static embedding channel to tune embeddings for sentiment analysis and sarcasm, and a static embedding channel to prevent over-tuning of embeddings due to collocation. The convolutional layers were good at capturing compositionality.


 

## Experimental Setup
This experiment uses two publicly available datasets released by Mishra et al. (2016; 2014). Dataset 1 contains 994 text snippets (383 positive and 611 negative, 350 of which are sarcastic/ironic) and was annotated by 7 human annotators with an accuracy of 70-90% and Fleiss kappa IAA of 0.62. Dataset 2 contains 843 snippets (443 positive and 400 negative) and was annotated by 5 human subjects with an accuracy of 75-85% and Fleiss kappa IAA of 0.68.

The experiment setup includes 9 configurations: Text Only, Text_Static, Text_Non-static, Text_Multi Channel, Gaze Only, Gaze_Fixation_Duration, Gaze_Saccade, Gaze_Multi Channel, and Both Text and Gaze.

The model details include Word Embeddings (Word2Vec trained on Amazon Movie Review Data with 300 dimensions), Convolution (filter sizes of 3 and 4 with 150 filters for each size), Feed-Forward (150 hidden neurons with a dropout probability of 0.25), and Training (200 epochs).


 

## Results

### Sentiment Analysis
- Training accuracy reaches 100 within 25 epochs with validation accuracy still at around 50%. 
- Better dropout/regularization configuration required.

### Sarcasm Detection
- Clear differences between vocabulary of sarcasm and non-sarcasm classes in dataset.
- Captured well by non-static embeddings.
- Reducing embedding dimension improves by a little margin.
- Increasing filters beyond 180 decreases accuracy (possibly over-fits).
- Non-static text channels better for sentiment analysis.
- Saccade channel alone handles nuances like incongruity better.
- Fixation channel does not help much.

## Analysis of Features Learned
- Addition of gaze information helps to generate features with more subtle differences.
- Features for sarcastic texts exhibit more intensity than non-sarcastic ones.
- Example 4 is incorrectly classified by both systems.
- Addition of gaze information does not help here.

## Project Idea
- CNN for multitask learning (e.g. sentiment analysis and emotion detection).
- Dataset required.


 

## Introduction
IEmoCaps is a two-filter system used for tasks such as machine translation. It is based on the classic diagram and paper by Vaswani et al. (2017).

## Chronology
The development of IEmoCaps can be traced back to IBM Models of Alignment (Brown et al. 1990, 1993), Phrase Based MT (Koehn 2003), Encoder Decoder (Sutskever et al. 2014, Cho et al. 2014), Attention (Bahadanu et al. 2015), and Transformer (Vaswani et al. 2017).

## Attention
Attention compares every element with all other elements, representing the input context as a weighted average of input word embeddings. This operation can be applied in parallel to all elements in the sequence.

## Self-Attention
Self-attention compares every word with every other word in the same sentence. This allows for direct comparison between arbitrary words and better modelling of long-range dependencies.

## Important Observations
The strength of attention is not related to probability of co-occurrence. Co-occurrence count is used to identify words that frequently appear together, while attention


 

# Attention in Transformer Architecture

Attention strength is determined by co-occurrence count. For example, the pair of words "Peter slept" has high attention, while "Peter early" has low attention. Attention is a two-part process, where each output token should attend to the token before it and the tokens in the input sequence.

The Transformer architecture consists of self-attention blocks stacked together to create deep networks. Each layer has a feedforward layer and a self-attention layer. Additionally, positional embeddings are used to uniquely identify a position in the sentence, which helps the model distinguish between words with the same representation.

Multiple self-attention heads are used at each layer, each of which learns different kinds of dependencies. The decoder layer also has a cross-attention layer and masking for future time-steps. Residual connections and layer normalization are used between layers.


 

# Attention in NLP

Attention is a mechanism used in NLP tasks to enhance the important parts of the input data and fade out the rest. It is learned through training data by gradient descent. There are two kinds of attention: dot product attention and multihead attention. Dependency parse attention is also used, which is based on parsing the root and subject of the sentence. 

Sentence examples are given to illustrate how attention works. In the first sentence, Ram is a good student who lives in London, a large metro, and will go to the University for higher studies. In the second sentence, Sitawho is a good student and lives in London, a large metro, will goto the University for higher studies. 

Attention helps the network devote more computing power to the small part of the data that matters. This has led to tremendous advances in machine translation and large improvements in NLU tasks. Transformer models are now the de-facto standard for many NLP tasks.


 

# Word Alignment and Attention
Word alignment in statistical machine translation is analogous to cross attention in neural machine translation. Phrase alignment uses both self attention and cross attention. Attention was known to linguists but its measurement is the contribution of NLP. Attention is measured in terms of probability.

# FFNN for Alignment
FFNN can be used for alignment, such as in the example of "Peter slept early" being translated to "piitar jaldii soyaa".

# Limitations of RNN
RNNs have a sequential nature which makes training time very large.

# Positional Encoding
Inspired by Shakespeare, Transformer's major contribution is positional encoding. This allows for position sensitivity, such as in the example of "Jack saw Jill" vs. "Jill saw Jack". The main verb (MV) is transitive and in past tense, so the NP to the left of MV should get the 'ne' postposition mark and the NP to the right of MV should get the 'ko' postposition mark.


 

Positional Encoding (PE) is a major contribution of the Transformer model. PE encodes positions as embeddings and supplies them along with input word embeddings. The training phase teaches the transformer to condition the output by paying attention to not only input words, but also their positions. The ith component of the tth position vector is denoted as pos(t,i), varying from 0 to (d/2)-1.

Sine and Cosine functions are used to design PEs due to two foundational observations. Firstly, if the set of patterns created by a set of symbols is larger than the set of symbols, then there must exist patterns with repeated symbols. Secondly, if the patterns can be arranged in a series with equal difference of values between every consecutive pair, then at any given position, the symbols at different positions of the pattern strings must repeat.

Decimal integers and binary numbers are used to illustrate periodicity and decimal integers. The digits repeat after every 10 numbers in the lowest significant position, after every 100 numbers in the next lowest position, after every 1000 numbers in the next to next lowest and so on.

Challenges in designing PEs include not being able to


 

NLP is a field of ambiguity, and thus binary values are not suitable for representing language objects. To represent language objects, a vector should be used with components ranging from 0 to 1. This is why the creators of transformers use sine and cosine functions, as they meet the criteria of being component-by-component, ranging from 0 to 1, and being periodic. 

As an example, consider the sentence "Jack saw Jill". There are three positions indexed as 0, 1 and 2. Assume the word vector dimension is 4 and the frequency is 1/(102i/d), i=0,1. The initial probabilities of each cell denote t(a w), t(a x) etc. The expected count for C[wa; (a b)(w x)] is 1/2. The revised probability for t(a w) is also 1/2.


 
------------------------------------------------
EM Alignment Expressions
EM (Expectation Maximization) is a method used to align two languages, English and French, by mapping their vocabularies. The data consists of sentence pairs, with the number of words on each side being denoted by 𝑙𝑠 and 𝑚𝑠 respectively. The goal is to find the probability of each word in one language being mapped to a word in the other language.

Revised Probabilities Table:
The revised probabilities table is a matrix of probabilities that shows the likelihood of a word in one language being mapped to a word in the other language. The table is revised until convergence is reached, and the binding between words gets progressively stronger.

Vocabulary Mapping:
The vocabulary mapping is a list of words in both languages that are mapped to each other. This list is used to calculate the probability of each word in one language being mapped to a word in the other language.

Key Notations:
The key notations are used to denote the English and French vocabularies (𝑉𝐸 and 𝑉𝐹), the number of observations/sentence


 

## English Vocabulary/Dictionary
This slide discusses the English vocabulary/dictionary and its indexing. The index of a French word is denoted by q and is present in the French vocabulary/dictionary.

## Hidden Variables and Parameters
The total number of hidden variables is equal to the sum of all the sentences. Each hidden variable is either 1 or 0, depending on whether the English word in the sentence is mapped to the French word. The total number of parameters is equal to the product of the English and French vocabularies. Each parameter is the probability that a word in the English vocabulary is mapped to a word in the Hindi vocabulary.

## Likelihoods
The data likelihood, data log-likelihood, and expected value of data log-likelihood are discussed.

## Constraint and Lagrangian
The constraint and Lagrangian are discussed.

## Asymmetric Dictionary Mapping
The dictionary mapping is obtained by looking from the English side. We can also look from the Hindi side and take the average of Pij and Pji. Aligners like GIZA++, Moses, and Berkley do this.

## Final E


 

## Sequence Labelling Task
Sequence labelling tasks involve input and output sequences of the same length, with the output containing categorical labels. The output at any time step typically depends on neighbouring output labels and input elements. Examples of such tasks include part-of-speech tagging and language modelling.

## Recurrent Neural Network
Recurrent Neural Network (RNN) is a powerful model to learn sequence labelling tasks. It consists of an input layer, an output layer, and a context layer. The same parameters are used at each time step, and the model size does not depend on the sequence length. Long range context is modeled by the context layer.

## Training Language Models
Language models are trained using a large monolingual corpus. At each time step, the model predicts the distribution of the next word given all previous words. The loss function minimizes the cross-entropy between the actual distribution and the predicted distribution.

## Evaluating Language Models
Language models are evaluated by measuring their ability to predict the next word given a context, and by evaluating the probability of a test set of sentences. Standard test sets such as Penn Treebank, Billion Word Corpus, and WikiText are used for


 

Language Models Evaluating LM
- Ram likes to play different activities with different probabilities, entropies, and perplexities.
- Perplexity is a measure of difference between actual and predicted distribution, and lower perplexity and cross-entropy is better.
- RNN models outperform n-gram models, and a special kind of RNN network - LSTM - does even better.
- Phrase Based SMT (PBSMT) and distortion Governing equation is used to model P(f|e).
- Distortion Probability is used to measure the distance between the translation of ith phrase and the translation of the (i-1)th phrase.


 

## Encode-Decode Paradigm
Encode-Decode Paradigm is a method used in deep learning for natural language processing (DL-NLP). It uses two RNN networks, the encoder and the decoder, to process and generate one element at a time. The encoder processes one input at a time and generates a representation of the sentence which is used to initialize the decoder state. The decoder generates one element at a time until the end of sequence tag is generated.

## Sequence to Sequence Learning
Sequence to Sequence Learning with Neural Networks is a method used to search for the best translations in the space of all translations. It uses an incremental construction approach where each hypothesis is scored using the model and hypotheses are maintained in a priority queue.

## Problems
The representation of the sentence generated by the encoder is insufficient to capture all the syntactic and semantic complexities. Long-term dependencies are also a problem as the source sentence representation is not useful after few decoder time steps. Solutions to these problems include using a richer representation for the sentences and making relevant source sentence information available when making the next prediction.


 

## Introduction to Convolutional Neural Networks (CNNs)

CNNs are a type of AI model that combines feedforward and recurrent-like features. They are inspired by biological processes, and the connectivity pattern between neurons resembles the organization of the animal visual cortex.

## Motivation

Two main motivations for using CNNs are:
- Reduced number of parameters
- Stepwise extraction of features

These two points are applicable to any AI situation, not just vision and image processing.

## Convolution Basics

Convolution is a process of combining two vectors, V1 and V2, to produce a third vector, V1V2. The filter V2 is used to emphasize or de-emphasize certain elements of V1.

## Receptive Field

The receptive field is the region of the visual field to which individual cortical neurons respond. Different neurons have partially overlapping receptive fields, which cover the entire visual field.


 

## Convolutional Neural Networks (CNNs)
CNNs are a type of neural network used for image classification. They use convolutional layers to detect edges, shapes, and higher-level features from raw pixels. The filter components are learned automatically, and the sliding of the filter corresponds to taking different receptive fields.

## Key Ideas
CNNs take advantage of four key ideas: local connections, shared weights, pooling, and the use of many layers.

## ImageNet
ImageNet is a dataset of 1 million images from the web, divided into 1,000 different classes. It has been used to achieve spectacular results, almost halving the error rates of the best competing approaches.

## Learning in CNNs
In image classification, CNNs learn to detect edges from raw pixels in the first layer, then use the edges to detect simple shapes in the second layer, and then use these shapes to detect higher-level features, such as facial shapes.


 

## Convolutional Neural Networks (CNNs)

CNNs are used for image recognition and natural language processing (NLP). In NLP, the input matrix is a 10x100 matrix for a 10 word sentence using a 100-dimensional embedding.

### Filters

Multiple filters can be used in multitask learning settings, such as sentiment analysis and emotion analysis. The number of filters should be equal to the number of tasks.

### N-grams

Lower order n-grams (unigrams and bigrams) give importance to lexical properties, while higher order n-grams (trigrams, quadrigrams, and pentagrams) give emphasis to syntactic structure and dependencies.

### Hyperparameters

CNNs have several hyperparameters, such as narrow width vs. wide width, stride size, pooling layers, and channels.


 

## Convolutional Neural Networks

Convolutional neural networks (CNNs) are a type of deep learning algorithm used for image recognition and classification. They are composed of several layers, including convolution layers, pooling layers, and fully connected layers.

### Tensors and Vectors

Tensors are vectors of vectors, while vectors are a single row of numbers. Images are composed of three channels: red, green, and blue.

### Pooling Layer

The pooling layer is used to summarize the features of the feature map. It involves sliding a two-dimensional filter over each channel of the feature map. The output dimension after pooling is determined by the height and width of the feature map, the number of channels, the height and width of the filter, and the stride length.

### Learning in CNN

The weights of the kernels are adjusted using backpropagation from the final layer of softmax. The weights are parameter-shared, and no special steps are needed for the RELU and MAX functions.


 

## Sentiment and Emotion Analysis
Sentiment Analysis is the task of identifying if a certain piece of text contains any opinion, emotion or other forms of affective content. Machine Learning based approaches use classifiers such as SVM, KNN and Random Forest, as well as sentiment-based features such as the number of positive and negative words, and highly emotional words. Emotion features include positive and negative emoticons, and boolean features that are 1 if both positive and negative words are present in the tweet. Punctuation features include the number of exclamation marks, dots, question marks, capital letter words, and single quotations.

## Deep Learning based Approach
Deep Learning based approaches use CNN-FF models with very little feature engineering. Embedding size of 128, maximum tweet length of 36 words, padding, and filters of size 3, 4, 5 are used to extract features.

## Sentiment Annotation and Eye Movement
Sarcastic tweets have longer fixations and multiple regressive saccades. Two publicly available datasets have been released by Mishra et al. (2016; 2014).


 

## Learning Cognitive Features from Gaze Data for Sentiment and Sarcasm Classification

Two datasets were used for this study:

**Dataset 1:**
- Eye-tracker: Eyelink-1000 Plus
- 994 text snippets: 383 positive and 611 negative, 350 are sarcastic/ironic
- Mixture of movie reviews, tweets and sarcastic/ironic quotes
- Annotated by 7 human annotators
- Annotation accuracy: 70%-90% with Fleiss kappa IAA of 0.62

**Dataset 2:**
- Eye-tracker: Tobi TX 300
- 843 snippets: 443 positive and 400 negative
- Annotated by 5 human subjects
- Annotation accuracy: 75%-85% with Fleiss kappa IAA of 0.68

The central idea of this study was to learn features from gaze sequences (fixation duration sequences and gaze-positions) and text automatically using deep neural networks. Convolutional Neural Networks (CNNs) were used for this purpose, as they have proven to be good at learning feature representations for image and text classification tasks. Both static and non-static


 

## Experimental Setup
This experiment uses two publicly available datasets released by Mishra et al. (2016; 2014). Dataset 1 consists of 994 text snippets (383 positive and 611 negative, 350 of which are sarcastic/ironic) and was annotated by 7 human annotators with an accuracy of 70-90% and Fleiss kappa IAA of 0.62. Dataset 2 consists of 843 snippets (443 positive and 400 negative) and was annotated by 5 human subjects with an accuracy of 75-85% and Fleiss kappa IAA of 0.68.

The experiment setup consists of 9 configurations: Text Only, Text_Static, Text_Non-static, Text_Multi Channel, Gaze Only, Gaze_Fixation_Duration, Gaze_Saccade, Gaze_Multi Channel, and Both Text and Gaze.

The model details include Word2Vec (Mikolov et.al) trained on Amazon Movie Review Data with 300-dimensional embeddings, convolution with filter sizes of 3 and 4 and 150 filters for each size, feed-forward with 150 hidden neurons and 0.25 dropout probability, and


 

## Results of Sentiment Analysis and Sarcasm Detection

- Overfitting for SA dataset 2: Training accuracy reaches 100 within 25 epochs with validation accuracy still at around 50%. Better dropout/regularization configuration required. 
- Better classification accuracy for Sarcasm detection: Clear differences between vocabulary of sarcasm and non-sarcasm classes in our dataset. Captured well by non-static embeddings.
- Effect of dimension variation: Reducing embedding dimension improves by a little margin.
- Increasing filters beyond 180 decreases accuracy (possibly over-fits). Decreasing beyond 30 decreases accuracy.
- Effect of static/non-static text channels: Better for non-static (word embeddings with similar sentiment come closer in non-static channels, e.g., good ~ nice).
- Effect of fixation/saccade channels: Saccade channel alone handles nuances like incongruity better.
- Fixation channel does not help much, may be because of higher variance in fixation duration.

## Analysis of Features Learned

- Addition of gaze information helps to generate features with more subtle differences.
- Features for the sarcastic texts exhibit more intensity than the


 

## IEmoCaps
IEmoCaps is a two-filter system used for tasks such as machine translation. It is based on attention and transformer models.

### Chronology
The development of IEmoCaps began with IBM models of alignment (Brown et al. 1990, 1993), followed by phrase-based machine translation (Koehn 2003), encoder-decoder models (Sutskever et al. 2014, Cho et al. 2014), attention models (Bahadanu et al. 2015), and finally the transformer model (Vaswani et al. 2017).

### Attention
Attention compares every element with all other elements, representing the input context as a weighted average of input word embeddings. This operation can be applied in parallel to all elements in the sequence.

### Self-Attention
Self-attention compares every word with every other word in the same sentence. This allows for direct comparison between arbitrary words and better modelling of long-range dependencies.

### Important Observations
The strength of attention is not related to probability of co-occurrence, and is based on semantics of a sentence. Additionally, pairs of words differ in their strength of


 

# Attention in Transformer Architecture

Attention strength is determined by co-occurrence count. For example, the pair of words "Peter slept" has high attention, while "Peter early" has low attention. Attention is a two-part process, where each output token should attend to the token before it and the tokens in the input sequence.

The Transformer architecture uses self-attention blocks to create deep networks. It consists of a feedforward layer, self-attention layer, positional embeddings, and multiple self-attention heads. Positional embeddings are used to uniquely identify a position in a sentence, and multiple self-attention heads are used to learn different kinds of dependencies. The decoder layer also has a cross-attention layer and masking for future time-steps. Residual connections and layer-normalization are used between layers.


 

# Attention in Natural Language Processing

Attention is a mechanism used in natural language processing (NLP) to enhance the important parts of the input data and fade out the rest. It is learned through training data by gradient descent and can be divided into two types: dot product attention and multihead attention. Dependency parse attention is also used to parse the data.

Sentence-1 and sentence-2 are examples of how attention can be used to identify important parts of the data. In sentence-1, Ram is identified as a good student who lives in London, a large metro, and will go to the university for higher studies. In sentence-2, Sitawho is identified as a good student who lives in London and will go to the university for higher studies.


 

# Word Alignment and Attention
Word alignment in statistical machine translation is analogous to cross attention in neural machine translation. Phrase alignment uses both self attention and cross attention. Attention was known to linguists, but its measurement is the contribution of NLP. Attention is measured in terms of probability.

# FFNN for Alignment
Introducing an Attention Layer between the Encoder and Decoder can help learn the weights of the alignment.

# Positional Encoding
RNNs have a sequential nature which makes training time very large. Transformer's major contribution is positional encoding, which is inspired by Shakespeare's quote. Position sensitivity is also important, as the NP to the left of the main verb should get the 'ne' postposition mark and the NP to the right should get the 'ko' postposition mark.


 

Positional Encoding (PE) is a major contribution of the Transformer model. PE encodes positions as embeddings and supplies them along with input word embeddings. The training phase teaches the transformer to condition the output by paying attention to not only input words, but also their positions. 

The components of the position vector are denoted as pos(t,i), varying from 0 to (d/2)-1. Sine and Cosine functions are used to encode the positions due to two foundational observations: 1) patterns with repeated symbols exist if |P|>|S|, and 2) the symbols at different positions of the pattern strings must repeat if the patterns can be arranged in a series with equal difference of values between every consecutive pair. 

Decimal integers and binary numbers are used to illustrate periodicity and decimal integers. Challenges in designing PEs include not being able to append decimal integers as position values, and not being able to normalize due to word relations changing with the length of sentences.


 

Natural Language Processing (NLP) is a field of study that deals with ambiguity and requires soft choices in its components. To represent a language object, a vector must be used with components ranging from 0 to 1. The creators of transformers have found that sine and cosine functions meet these requirements.

Statistical Alignment Learning is a non-neural EM algorithm used for word alignment from sentence alignment. Initial probabilities are assigned to each cell, and expected counts are calculated based on the frequency of the words. Revised probabilities are then calculated based on the expected counts.


 
------------------------------------------------
EM-Based Alignment Expressions
EM-based alignment expressions are used to map words between two languages. The expressions involve the use of a data set (D) which consists of sentence pairs, each with a corresponding number of words on the English and French sides. The expressions also involve the use of key notations such as the English vocabulary (VE), French vocabulary (VF), number of observations/sentence pairs (S), number of words on English side in the sentence (lS), and number of words on French side in the sentence (mS). Additionally, the expressions involve the use of revised and re-revised probabilities tables which are used to calculate the binding strength between two words.


 

## English Vocabulary/Dictionary
English vocabulary/dictionary is indexed by French words (𝑞). The total number of hidden variables is equal to the number of sentences (𝑆𝑙𝑠𝑚𝑠). Each hidden variable (𝑧𝑝𝑞𝑠) is equal to 1 if the English word in the sentence is mapped to the French word, and 0 otherwise. The total number of parameters is equal to the number of English words (𝑉𝐸) multiplied by the number of French words (𝑉𝐹). The probability (𝑃𝑖,𝑗) that the ith English word is mapped to the jth French word is known as "asymmetric" because the dictionary mapping is obtained by "looking" from the English side. Aligners like GIZA++, Moses, and Berkley differentiate with respect to 𝑃𝑖𝑗.

## Recurrent Neural Networks
Recurrent Neural Networks (RNN) have two key ideas: summarizing context information into a single vector (𝑐(𝑥𝑖


 

Recurrent Neural Networks (RNNs) are powerful models for sequence labelling tasks such as part-of-speech tagging. RNNs use the same parameters at each time step and the model size does not depend on the sequence length. Long range context is modeled by predicting the distribution of the next word given all previous words. 

The parameters of the model are learned by minimizing the cross-entropy between the actual distribution and the predicted distribution. The quality of the language model is evaluated by predicting the next word given a context and by evaluating the probability of a test set of sentences. Standard test sets such as Penn Treebank, Billion Word Corpus, and WikiText are used for evaluation.


 

Language Models Evaluating LM
- Ram likes to play different activities with different probabilities, entropies, and perplexities.
- Perplexity is a measure of difference between actual and predicted distribution, and lower perplexity and cross-entropy is better.
- RNN models outperform n-gram models, and LSTM does even better.
- Phrase Based SMT (PBSMT) and distortion governing equation is used to model P(f|e).
- Distortion probability is a measure of the distance between the translation of ith phrase and the translation of the (i-1)th phrase.


 

## Convolutional Neural Networks (CNNs)
CNNs are a type of deep learning model used for image classification. They are composed of multiple layers, each of which learns to detect edges, shapes, and other features from raw pixels. The last layer is a classifier that uses these high-level features. CNNs became popular after the ImageNet dataset was released, which contained 1 million images from the web and 1,000 different classes.

## CNNs for Natural Language Processing (NLP)
CNNs can also be used for NLP tasks, such as sentiment analysis and sarcasm detection. For sarcasm detection, longer fixations and multiple regressive saccades are used to identify sarcastic sentences.

## Attention and Transformer
Attention and positional encoding are two important components of the Transformer model, which is used for machine translation. The Transformer model combines these two components to create a powerful model for machine translation.


 

## Attention
Attention is a mechanism used in natural language processing (NLP) to identify relevant words in a sentence. It is used to create a score vector for each word vector in a phrase. This score vector is then used to create a matrix, which is then scaled and weighted to create a soft vector. This soft vector is then used to create a matrix of attention weights, which is used to identify which words should be attended to with more attention.

## Query, Key and Value
The query, key and value are three components used in attention. The query is used to identify which words should be attended to, the key is used to identify how much attention should be given to each word, and the value is used to determine the output of the attention. The weights of these components can be learned by gradient descent.

## Self Attention
Self attention is used when the decoder generates an output sequence. It is a two-part attention, where each output token should attend to the token that has been output before, as well as the tokens in the input sequence.

## Important Observations
In the input sequence, pairs of words differ in their strength of association. For example, for an


# Attention in NLP

Attention is a mechanism used in NLP tasks to enhance the important parts of the input data and fade out the rest. It is learned through training data by gradient descent and is used to determine which part of the data is more important than others depending on the context.

## Types of Attention

There are two main types of attention used in NLP tasks:

- Dot Product Attention
- Multihead Attention

## Dependency Parse Attention

Dependency Parse Attention is a type of attention used to parse the dependencies between words in a sentence. It is based on the root, subject, object, and other relationships between words. For example, in the sentence "Ram who is a good student and lives in London which is a large metro, will goto the University for higher studies.", the following dependencies can be parsed:

- root(ROOT -0, go-18)
- nsubj (go-18, Ram -1)
- nsubj (student -6, who -2)
- cop(student -6, is-3)
- det(student -6, a-4)
- amod (student -6


 

The Transformer model is a neural network architecture that uses two main components: Attention and Positional Encoding. Attention is used to learn the attention weights between words in a sentence, while Positional Encoding is used to capture the position of words in a sentence. The Transformer model is an improvement over the Recurrent Neural Network (RNN) model, as it eliminates the need for the encoder and decoder to wait for each other to process the input. Additionally, the Transformer model is able to capture the position sensitivity of words in a sentence, which is not possible with the RNN model. Finally, the Transformer model was inspired by Shakespeare's famous quote, "All the world's a stage,/ And all the men and women merely players".


 

Positional Encoding
• Positional encoding is a technique used to add additional disambiguation signals to words. 
• It allows words to influence each other through their properties and positions. 
• This is done by encoding positions as embeddings and supplying them along with input word embeddings. 
• The transformer is then trained to condition the output by paying attention to both input words and their positions. 

Position Vector Components
• The kth component of the tth position vector is denoted as pos(t,k), with k varying from 0 to d-1, where d is the dimension of the PE vector. 
• For even and odd positions, the components range from 0 to d/2-1.

Challenges in Designing PEs
• Decimal integers cannot be appended as position values, as words later in the sentence will dominate. 
• Normalization cannot be used either, as word relations change with sentence length. 
• Binary values also will not do, as 0s will contribute nothing and 1s will influence completely. 
• A language object represented by a vector must allow soft choices in its components, preferably represented


 

Position Vector Components
•The kth component of the tth position vector is denoted as pos(t,k), with k varying from 0 to d-1, where d is the dimension of the PE vector. 
•For even and odd positions, i varies from 0 to d/2-1.

Example: “Jack saw Jill”
•Three positions are indexed as 0, 1 and 2. 
•Assume word vector dimension d to be 4 and the frequency to be 1/(102i/d).

Machine Translation: The Tricky Case of ‘Have’ Translation
•If the syntactic subject is animate and the syntactic object is owned by the subject, “have” should translate to “kade… aahe”. 
•If the syntactic subject is animate and the syntactic object denotes kinship with the subject, “have” should translate to “laa… aahe”. 
•If the syntactic subject is inanimate, “have” should translate to “madhye… aahe”. 
•Examples of translations include


 

# CS772: Deep Learning for Natural Language Processing (DL-NLP)

This lecture focuses on prompting, reasoning, bias, SSMT, QE, APE, fake-news & half-truth detection, query intent detection, and speech emotion recognition. It is taught by Prof. Pushpak Bhattacharyya at IIT Bombay in the Computer Science and Engineering Department.

## Need for Prompting

Due to the increase in size of language models, fine-tuning becomes infeasible and ineffective. The goal is to have a single model perform many downstream tasks without any gradient updates. This is done by providing a "prompt" specifying the task to be done. Examples of large language models include ELMo (1B training tokens), BERT (3.3B training tokens), RoBERTa (~30B training tokens), and PaLM (750B tokens).

## Paradigms in NLP

Four paradigms in NLP are discussed, along with terminology and notation of prompting methods.

## Design Considerations for Prompting

Design considerations for prompting include pre-trained model choice, prompt engineering, answer engineering, expanding the paradigm, and


 

## Left-to-Right Language Model

- The earliest architecture chosen for prompting.
- Usually used with prefix prompts and parameters on the LLM are fixed.
- Examples: GPT-2, GPT-3, BLOOM.

## Masked Language Model

- Usually combined with cloze prompt.
- Suitable for NLU tasks, which should be reformulated to cloze tasks.
- Examples: BERT, ERNIE.

## Pretrained Language Model Choice

- The SOTA LLMs created to generalise well on various tasks.
- The LMs are created by:
  - Scaling number of tasks.
  - Scaling the model size.
  - Fine-tuning using chain-of-thought data.
  - Training using RLHF.
- Examples: flanT5, InstructGPT, chatGPT.

## Prompt Engineering

- Traditional vs Prompt formulations.
- Prompt Template Engineering:
  - Prompt shape: Cloze prompt, Prefix prompt.
  - Design of Prompt Template: Hand crafted, Automated search (Discrete space, Continuous space).




 

## Language Model Prompting
Language models can be used to translate English to German, without requiring gradient updates or fine-tuning. However, this approach has several drawbacks, such as requiring domain expertise and lagging behind SotA model tuning results. Additionally, language models are sensitive to the choice of prompts, which may not be effective for the model. 

Two types of prompting are used to address this issue: discrete/hard prompting and continuous/soft prompting. Discrete prompting involves prepending a sequence of additional task-specific tunable tokens to the input text. Continuous prompting involves reasoning with large language prompting, which is a combination of two processes: thinking fast and slow. 

Finally, bias in statistics and ML, as well as bias in social context, should be taken into account when using language models. Bias of an estimator is the difference between the predictions and the true values, while the prior P(X) serves as a bias in a Bayesian framework. Bias in social context refers to preference or prejudice towards certain individuals, groups or communities.


 

Bias in AI Models

Bias can be present in AI models in various ways, such as through data, annotations, data sampling, and representation bias. Bias in core algorithms and loss functions can also lead to biased outputs. Leveraging prompting for model debiasing can help reduce bias.

Speech-to-Speech Machine Translation

Speech-to-Speech Machine Translation (SSMT) is an automated process of converting speech in source language to speech in target language. There are two approaches: cascaded SSMT and end-to-end SSMT. Cascaded SSMT systems have components such as automatic speech recognition, text-to-speech, and machine translation.


 

## Automatic Speech Recognition
The goal of Automatic Speech Recognition (ASR) is to convert speech in a source language to text in the same language. Deep Learning based ASR systems have achieved state-of-the-art performance in many languages and domains, but their performance is insufficient for Indian English due to the nature of speech and dialect. Our aim is to create an excellent quality transcription system for Indian English and Education Domain.

We benchmark three popular approaches in ASR:
1. Facebook’s wav2vec 2.0: Pretraining CNN Encoders & Transformer on unlabelled speech data using self-supervision, followed by finetuning on domain-specific data in specific languages to achieve low word error rate.
2. Vakyansh CLSRIL-23 ASR System: Similar to wav2vec 2.0 with pre-training on 23 Indian languages followed by language-specific finetuning. Achieves the best reported performance on ASR for Indian languages like Hindi, Marathi and Tamil.
3. Open AI’s Whisper ASR System: Addresses the problems of wav2vec which creates more data-centric models


 

Test Set:
- 1335.74 hours of audio data
- Average clip duration of 7.69s
- Average of 17.33 words per utterance

Results:
- Word Error Rate (WER) of 49.2% for Wav2vec 2.0 (no finetuning)
- WER of 32.8% for Vakyansh CLSRIL-23 (no finetuning)
- WER of 28.2% for Open AI's Whisper ASR model
- WER of 28.4% for Vakyansh CLSRIL-23 (with finetuning)

Case Study 1 Sample:
- Whisper was able to detect slight pronunciations for the first word, but had difficulty with the last word due to ambiguous pronunciation.
- Word boundary detection was a problem for both models.

Disfluency Correction Introduction:
- Conversational speech is spontaneous, with speakers thinking about content as they speak.
- Disfluencies are words that don't add semantic meaning to the sentence.
- Speakers often use filler words, repeat phrases, or make corrections in speech.


 

Disfluencies are words that are part of spoken utterances but do not add meaning to the sentence. This thesis studies 6 types of disfluencies: fillers, interjections, discourse markers, repetitions/corrections, false starts, and edits. Transcribing speech and annotating the disfluencies is a tedious task, making it difficult to create a large corpus for Indian languages. To achieve high quality results, a combination of labelled, unlabelled, and pseudo labelled data from English and Indian languages is used. The current SOTA for Indian languages DC is defined by Kundu et al. (2022), which uses a Multilingual Transformer architecture for token classification. The model is trained on English data from the Switchboard corpus and synthetically generated data in Indian languages. Our few shot approach consists of three main components: MuRIL Encoder, Generator, and a rule-based technique for creating data in different disfluency types.


 

This lecture discussed a novel approach to low resource text-to-speech synthesis using a transliteration strategy for domain transfer to high resource languages like English. The approach was tested on auto-regressive and non-autoregressive models.

The lecture also discussed an interesting problem in Indian languages: reduplication vs repetition. Reduplication refers to the phenomenon of repeating words for greater emphasis of certain phrases, while repetition is a disfluency type where words are repeated in conversations as disfluent words.

The lecture also discussed a discriminator model that can classify disfluent/fluent tokens from labeled data, and determine whether Hfake or Hreal comes from a real distribution from unlabeled data. Results showed that the model improved the state-of-the-art in DC in Bengali, Hindi and Marathi by 9.19, 5.85 and 3.40 points in F1 scores. The model also demonstrated high accuracy (87.68 F1 score) for textual stuttering correction.


 
This study explored two models for a Text-to-Speech (TTS) spectrogram generator: Tacotron 2 and Forward Tacotron. The Marathi Text-Speech data from Indic-TTS Corpus was used, which consisted of 4.82 hrs of data with a mean duration of 7.09s. The Mean Opinion Score for Tacotron 2 + Waveglow vocoder was 4.53 out of 5, and for Forward Tacotron + Waveglow vocoder was 4.64 out of 5. The phoneme ‘cã’ was pronounced much better in Forward Tacotron compared to Tacotron 2. Pronunciations were very similar but Forward Tacotron had a higher pitch and more accurate intonation. Future work includes training Whisper model on labelled Indian languages ASR, dialect adaptation and applications in Closed Captioning, collecting more data and expanding to other Indian languages, and working on reduplication vs repetition.


 

# Automatic Post-Editing (APE)

APE is a process of automatically correcting translations generated by a machine translation system to make them publishable. It is a monolingual translation task, and the same MT technology is used for APE. There are two paradigms for APE: rule-based and phrase-based. Rule-based APE uses precise PE rules, but they may not capture all possible scenarios and are not portable across domains. Phrase-based APE has dominated the APE field for a few years and has shown significant improvements when the underlying MT system is refined.

APE systems can be categorized based on the accessibility of the MT system (black-box or glass-box), the type of post-editing data (real or synthetic), and the domain of the data (general or specific). We focus on black-box scenarios, real and synthetic data, and domain-specific APE systems.


 

WMT APE Shared Task
The WMT22 English-Marathi APE Shared Task is a task to develop a robust English-Marathi Automatic Post-Editing (APE) system using the data shared in the WMT22 APE Shared Task. The data consists of triplets of source sentences (SRC), MT output (MT_OP), and human post-edited versions of MT (PE_REF). The dataset includes synthetic APE data of around 2M triplets, real APE data of 18K triplets, and validation and test data of 1K triplets each.

Model Architecture:
The model architecture used is a two-encoders single-decoder model, as there is no vocabulary overlap between English and Marathi data. LaBSE-based data filtering is used to improve the quality of the synthetic data, and data augmentation is done using phrase-level triplet generation. The model is trained using a curriculum.


 

Training Strategy (CTS):
- Our CTS considers in-domain and out-domain triplets and their TER scores to learn more error patterns and prevent over-correction.

Data Pre-processing and Augmentation:
- LaBSE-based Filtering to filter low-quality triplets.
- Form subsets of the data based on domain.
- Phrase-level APE Triplet Generation: Extract SRC-MT and SRC-PE phrase tables and generate MT-PE triplets.
- External MT triplets: using mT5-fine-tuned NMT model.

Training:
- Step 1: MT task.
- Step 2: APE task using out-of-domain synthetic APE triplets.
- Step 3: APE task using in-domain high TER synthetic APE triplets.
- Step 4: APE task using low TER synthetic APE data augmented with External MT candidates (in-domain).
- Step 5: APE task using the real-APE data (Fine-tuning).

Selection of Final Output:
- Sentence-level Quality Estimation to select the final output:


 

Quality Estimation (QE) is a task that scores the translation quality of a given source text and translated text. QE can be done at the document, sentence, and word level. Document level QE scores the entire translated document pair, sentence level QE scores each translated sentence pair, and word level QE assigns an 'OK' or 'BAD' tag to each word and gap in the sentence. QE metrics include Human Translation Error Rate (HTER) and Direct Assessment (DA). HTER is the ratio of number of edits to reference sentence length, while DA is a translation quality score on a scale of 0-100 given by professional human translators. SOTA QE systems are unable to capture adequacy properly, as fluent yet incorrect translations are scored highly.


 

## Quality Estimation

This paper presents a novel application of Multi-Task Learning (MTL) for Quality Estimation (QE) tasks. The authors propose a model that jointly trains a model for both sentence-level and word-level QE tasks. The model uses Linear Scalarization (LS-MTL) for sentence-level QE loss and a combination of Direct Assessment (DA) and Word-level QE loss for word-level QE loss. The authors also introduce a novel application of the Nash-MTL method to both tasks in Quality Estimation.

The authors evaluated the model on two datasets and observed an improvement of up to 3.48% in Pearson's correlation (r) at the sentence-level and 7.17% in F1-score at the word-level. The MTL-based QE models were also found to be more consistent, on word-level and sentence-level QE tasks, for same inputs, as compared to the single-task learning-based QE models.


 

The WMT20 and WMT22 QE shared tasks were evaluated using three experimental settings: Single-Pair, Multi-Pair, and Zero-Shot. 

**Single-Pair Setting**

The Single-Pair Setting used data from one language pair for training and evaluation. Results were obtained for word-level (F1-scores) and sentence-level (Pearson Correlation (r)) QE tasks. The model was trained using the STL approach, and the results indicated that the improvement was not significant with respect to the baseline score. 

**Multi-Pair Setting**

The Multi-Pair Setting combined the data of all language pairs for training and evaluated on each language pair. Results were obtained for word-level (F1-scores) and sentence-level (Pearson Correlation (r)) QE tasks. The model was trained using the Linear Scalarization and Nash-MTL approaches, and the results indicated that the improvement was not significant with respect to the baseline score. 

**Zero-Shot Setting**

The Zero-Shot Setting combined the data of all language pairs for training except the language pair on which


 

## Correlation between Sentence-level and Word-level QE Predictions
Pearson (r) and Spearman (ρ) correlations were computed between the z-standardized Direct Assessment (DA) scores and the bad tag counts normalized by sentence length. A stronger negative correlation denotes more consistent predictions.

## Qualitative Analysis
The numbers in the STL and Nash-MTL columns are predictions (z-standardized DA scores) by the STL QE and Nash-MTL QE models, respectively. The Label column contains the ground truths. The MTL QE model predictions are more appropriate/justified than the STL QE model’s predictions when a source sentence contains many named-entities, the translation is of high quality and only have minor mistakes, or when the source sentence (and therefore its translation) is complex. Both STL and MTL QE models are poor in predicting quality of sentences appropriately when a source sentence (and its translation) is in the passive voice.

## Multi-Task Learning with APE and Quality Estimation
Motivation for a MTL-based model for APE and QE is to generate a high-quality output. Sentence-level


 

We discussed our submission to the WMT22 English-Marathi Automatic Post-Editing (APE) Shared Task. Our approach showed the helpfulness of augmenting APE data with phrase-level triplets and using a sentence-level Quality Estimation (QE) system to select the final output. We also investigated whether Multi-Task Learning (MTL) based training helps APE models when trained with QE tasks. Results showed that Word-level QE and sentence-level QE (DA) are most helpful to APE. We used Nash-MTL to improve the performance of the QE model and observed that jointly training a single model for different QE tasks results in consistent predictions.


 

Conditioning WaveNet on Mel Spectrogram Predictions

This slide discusses various research papers related to disfluency correction, zero-shot disfluency detection, automatic post-editing, translation quality estimation, and multi-task learning. 

References:

[1] Sourabh Deoghare and Pushpak Bhattacharyya. 2022. IIT Bombay’s WMT22 Automatic Post-Editing Shared Task Submission. In Proceedings of the Seventh Conference on Machine Translation (WMT), pages 682–688, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics. 

[2] Pushpak Bhattacharyya, Rajen Chatterjee, Markus Freitag, Diptesh Kanojia, Matteo Negri, and Marco Turchi. 2022. Findings of the WMT 2022 Shared Task on Automatic Post-Editing. In Proceedings of the Seventh Conference on Machine Translation (WMT), pages 109–117, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics. 

[3] Tharindu Ranasinghe, Constantin Orasan, and


 

## Fake News & Half-Truth Detection
This lecture discussed the detection and debunking of fake news and half-truths. It was presented by Singamsetty Sandeep (M.Tech in CSE Dept.), Apurva Kulkarni (IDDDP in CMINDS Dept.), and N V S Abhishek (M.Tech in CSE Dept.).

Fake news is false or misleading information presented as if it is true news. It is spread through social media, news channels, and digital platforms. It is used to increase viewership, communicate with users, and advertise digitally.

Language-agnostic BERT Sentence Embedding and Log-linear combinations of monolingual and bilingual neural machine translation models for automatic post-editing were discussed. Additionally, ESCAPE, a large-scale synthetic corpus for automatic post-editing, was discussed.
true.

 

Half-truths

Definition: A half-truth is a deceptive statement that contains some, but not all, elements of the truth. Half-truths are lies of omission, and even if a statement is technically true, it can not be considered a truth if it leaves out crucial pieces of information.

Example: Electronic gadgets are mandatory for e-census in 2023 (hidden: Govt. will provide the gadgets).

Problem Statement Part 1: Given a claim and the corresponding evidence from a trustworthy source, predict the veracity of the claim and produce counters or supports for the predicted veracity label.

Problem Statement Part 2: Given a claim and the corresponding evidence from a trustworthy source, edit the claim if the claim is half-true.


 

The Dolphins stadium renovation will create temporary jobs. This task aims to verify the veracity of a claim and provide counters or supports for the predicted label. Editing the original claim if it is fake or half-true can help transform fake news into true news, thus countering the spread of misinformation and its associated negative effects.


 

This lecture discussed the literature survey and LIAR-PLUS dataset related to half-truth or fake news and supporting the truth. 

**Literature Survey:**
1. Guo et al. (2022) presented an overview of the models and datasets in the domain of fact-checking, listing out the challenges and future directions. 
2. Kotonya and Toni (2020) discussed techniques used for explaining the verdicts in automated fact-checking. 
3. Alhindi et al. (2018) introduced the LIAR-PLUS dataset, which was used to detect veracity and outscored the LIAR-PLUS dataset paper. 
4. Atanasova et al. (2020a) generated justification for the claim and used the idea of a textual summary as an explanation. 
5. Gardner et al. (2020) showed the effectiveness of contrast sets by creating them for various datasets, which was used to study counterfactuals and debunk fake news. 
6. Atanasova et al. (2020b) discussed a technique to generate adversarial examples for the target label for each claim in the FEVER dataset. 
7


 

The LIAR-PLUS Fact Checking Pipeline is a dataset and model for veracity prediction. It consists of a set of veracity labels, a transformer-based model (BERT), metadata, and an architecture for veracity prediction. The model is used to evaluate the veracity of a statement, such as "The Dolphins stadium renovation will create more than 4,000 new local jobs." The evidence for the statement is provided, including a mailer distributed by Miami First, a 2010 study of a $225 million project, and details on the team's receipt of $379 million from the state and county. The model takes into account the total credit history count, including the current statement, and the extracted justification. The veracity label for this statement is "Mostly True".


 

The Dolphins Stadium renovation project will create temporary jobs, with the goal of hiring the majority of workers from Miami-Dade County. The project is estimated to create more than 4,000 new local jobs, based on a 2010 study of a $225 million project. However, the jobs are associated with the 25-month stadium renovation project and include temporary positions, with no details as to how many of those 4,000 jobs would extend beyond the construction phase. The team will receive $379 million from the state and county over about three decades, and eventually pay back about $159 million.


 

The Dolphins stadium renovation will create temporary jobs. Results from the implementation of the veracity prediction model showed that metadata is useful to improve accuracy, and that the length of the justifications is long and often contains useless information. To address this, the relevant parts of the justifications in the LIAR-PLUS dataset were extracted using the NLI model, which boosted the accuracy of the veracity prediction model. The TaPaCo dataset is a freely available paraphrase corpus for 73 languages extracted from the Tatoeba database. To fine tune T5, the input is masked using the idea of textual entailment and cosine similarity. The original claim that needs an edit has to be masked, and only the parts of the claim that contradict are considered.


 

The Dolphins Stadium Renovation Project
- The Dolphins stadium renovation project will create more than 4,000 new local jobs, according to a mailer distributed by Miami First. 
- The number was based on a 2010 study of a $225 million project that concluded 3,740 jobs in Miami-Dade and Broward. 
- The jobs are associated with the 25-month stadium renovation project and include temporary positions. 
- The team has set a goal to hire the vast majority of the workers from Miami-Dade County, but there is no financial penalty if they fail to do so. 
- The Dolphins will receive $379 million from the state and county over about three decades, and eventually pay back about $159 million. 

T5 Model
- The T5 model was aided by a paraphrase dataset, which helped it accurately learn the semantic and structural level properties of sentences. 
- The KL-divergence loss, along with the original loss, helped the T5 model preserve content and maintain fluency. 
- Filtering the claims using a reward mechanism has increased the overall accuracy of the model.


 

## Evaluation of Edited Claims
The evaluation of edited claims was conducted on the FAVIQ dataset. A scraping bot was used to scrape news articles from Google News. The use case was a claim that Virat Kohli would lead CSK for IPL 2023, with evidence extracted from Google News.

## Experiment-1
The experiment asked how many claims changed to true from half-true and false to true after claim editing. 2000 claims were tested, 1000 of which were half-true and 1000 of which were false. The results after claim editing were:
- Edited Claims (Tailor): 1244 (62.2%)
- GPT-2 PROMPT based: 114 (5.7%)
- ROBERTA (Text infilling): 75 (3.75%)
- PEGASUS (Summary from evidence): 864 (43.2%)
- T5 Claim editing model (Our model): 1694 (84.7%)

## Experiment-2
The task was to use a baseline model to compare the accuracy of veracity prediction model. The labels were True, False, Half-true, Mostly True, Barely True, and Pants-on-


 

# Fake News and Fact Checking

We discussed the issue of fake news and half-truths, and explored the implementation of an end-to-end fact checking pipeline. We looked at the use of claim editing to debunk fake news, and the use of a Google News scraper to extract real-time evidence. We also discussed the experiments, results, and evaluation of edited claims.

## Conclusions

- The size and quality of data is important for NLP tasks, and we improved the quality of the LIAR-PLUS dataset by extracting only the relevant pieces of information from the evidence, which improved the accuracy of the veracity prediction model.
- Even though complex models like GPT exist, a simple T5 model can outperform GPT for the task of claim editing, showing that models need to understand linguistic properties better.
- Search engines like Google are not able to solve the problem of evidence extraction, and should be smarter in understanding figurative speech and complex language.


 

Fact Checking and Fake News Detection

Fact checking and fake news detection are important tasks in natural language processing (NLP). Several methods have been proposed to address these tasks, including generating fact checking explanations (Oma et al., 2020a), generating label cohesive and well-formed adversarial claims (Atanasova et al., 2020b), automatic identification and verification of claims in social media (Barrén-Cedefio et al., 2020), a quantitative argumentation-based automated explainable decision system for fake news detection on social media (Chi and Liao, 2022), and evaluating NLP models via contrast sets (Gardner et al., 2020).

Rumour Detection

Jorrell et al. (2019) proposed SemEval-2019 task 7: RumourEval, which is used to determine the veracity and support for rumours. Tuo et al. (2019) proposed exploiting emotions for fake news detection on social media.

Automated Fact-Checking

Juo et al. (2022) proposed a survey on automated fact-checking, and Gupta et al. (2022) proposed a method for automated fact-checking using a knowledge graph.


 

# Query Intent Detection and Slot Filling
Query Intent Detection is an important Information Extraction problem in NLP used to assist search engines by providing intent information of user queries to fetch appropriate results. It is a classification problem which categorizes an input user query among a set of specific intent classes.

## Problem Statement
Classify user query into set of predefined intent classes. This requires the creation of an intent class taxonomy for specific domains or tasks with the help of domain expertise while taking the downstream tasks into consideration. The taxonomy can be multi level, with broad initial classes and finer subclasses.

## Challenges
The main challenges in Query Intent Detection are:
- Creating a taxonomy with the help of domain expertise
- Categorizing user queries into specific intent classes
- Handling multi-level taxonomies with broad initial classes and finer subclasses


 

This lecture discussed the challenges of query understanding in a real-world setting, such as multi-domain queries with differing terminology and styles, multilingual support, and large number of classes with data skews. Initial approaches used text-based features and classical ML models, but deep learning revolutionised the field with end-to-end systems. The current state of the art involves the use of deep learning architectures with pretrained language models, fine-tuned on specific task data.

For the entertainment domain, the problem statement is to identify the domain, intent, and entities in a multilingual search query, with entities extracted and transliterated to the native script. The system supports Hindi, Marathi, Bengali, Tamil and Telugu, and can handle code-mixed or script-mixed queries with English.

The language challenges include transliteration, code-mixing, script-mixing, and structural orientation. There is no available dataset or taxonomy, and the test data has a large skew to mimic real-world data.

The query understanding pipeline is a deep learning based system, with a binary classifier for domain detection and a multiclass classifier for intent detection.


 

Query Understanding
- Entity extraction labels each word of the query to extract entities from the query
- Transliteration translates all entities to the native script

Intent Detection Model Architecture
- Language models like BERT are pretrained on large corpus (masked language modelling)
- These pretrained models are then fine tuned on specific downstream NLP tasks
- m-BERT or MuRIL is used, which is a 12-layer, 768-hidden, 12-heads, 110M parameters model
- Pre-train BERT has vocab size of ~110k for 104 languages, and pre-trained MuRIL has vocab size of ~197k for 17 Indian languages

Intent Taxonomy
- Taxonomy is a three level tree
- Level 1 intent is broad level or major intent consisting of 8 intent categories
- Levels 2 and 3 are further categorizations of their previous levels
- This method creates 138 classes for intent, of which only 65 classes had sufficient representation and were considered

Dataset
- Total number of queries annotated for intent is 47,475
- The taxonomy gives 65 classes for classification
- Dataset contains Hindi, Marathi,
illing Results 

 

Dialog State Tracking (DST) is a system used to extract information from user sentences and fill slots in a frame with the fillers the user intends. The system then performs the relevant action for the user. DST systems typically have three modules: domain detection, intent detection, and slot filling. 

Datasets used for DST include ATIS, SNIPS, MultiWOZ, and Schema-Guided Dialogue (SGD). ATIS has 17 intent classes and 5000 train set instances, while SNIPS has 7 intent classes and 13000 train set instances. SGD introduces complexity of different intent and slot classes from different domains. 

Intent detection results for ATIS and SNIPS are available, as well as slot filling results.


 

# Query Intent Detection
This paper discussed the problem of query intent detection and its applications, as well as the challenges and approaches for query intent detection. We also discussed a multilingual query understanding pipeline revolving around user intent detection for search engines for Indian languages, and the problems of intent detection and slot filling in dialogue state tracking.

References:
1. J. Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT 
2. Liat Ein Dor, Yosi Mass, Alon Halfon, Elad Venezian, Ilya Shnayderman, Ranit Aharonov, and Noam Slonim. 2018. Learning thematic similarity metric from article sections using triplet networks. pages 49–54 
3. Teresa Gonalves and Paulo Quaresma. 2010. Multilingual text classification through combination of monolingual classifiers. CEUR Workshop Proceedings, 605. 
4. Simran Khanuja, Diksha Bansal, Sarvesh Mehtani, Savya Kh


 

## Speech Emotion Recognition
Speech emotion recognition is the task of predicting an emotion class (categorical model) like happy, angry, sad, etc. or a real-valued metric like valence, arousal and dominance. It has massive applications across multiple domains.

### Evolution of Intelligent Interactive Agents
Mensio et al. (2018) discussed three steps of evolution for conversational agents: 
* Textual interaction 
* Vocal interaction 
* Embodied interaction 

### Popular Speech Representation Models
* Wav2Vec 2.0
* Other models

### Ongoing Work
Ongoing work includes experiments for speech emotion recognition and shareable representations for search query understanding.


 

## Speech Emotion Recognition (SER)
SER is a process of recognizing emotions from audio signals. It can classify emotions such as happy, angry, neutral, and sad. It also measures valence, arousal, and dominance.

## Emotion Recognition in Conversation (ERC)
ERC is the task of predicting the emotion label of each utterance in a conversation. It is a challenging task due to the lack of emotion-labeled speech data, noise and speech variations, and the presence of emotion shifts and sarcasm.

## Controlling Variables in Conversation
Conversations are governed by different factors such as topic, interlocutors’ personality, argumentation logic, viewpoint, intent, etc.

## Two-Dimensional Emotion Model
The two-dimensional emotion model is based on Plutchik's wheel of emotions. It is used to recognize emotions from speech features.


 

Acoustic Features
Acoustic features can be extracted from raw audio signals, such as MFCC, energy, pitch, and mel-spectrograms.

Automatic Speech Recognition
Automatic Speech Recognition (ASR) is the task of converting a speech utterance into a sequence of tokens (words, subwords, characters). Traditional cascaded ASR systems use acoustic, language, and pronunciation models, as well as searching. End-to-end ASR systems use encoder-decoder with attention.

Wav2Vec 2.0
Wav2Vec 2.0 is a self-supervised architecture that learns powerful speech representations from raw audio. It has a feature encoder, quantization module, and transformer block. It is pre-trained using a large amount of unlabeled speech data and fine-tuned using smaller labeled speech data.

Performance
The authors tested Wav2Vec 2.0 on Librispeech clean/other test data and found that it performs well with different amounts of fine-tuning data.


 

## Question
Can wav2vec2 features do better than acoustic features for SER (Speech Emotion Recognition)?

## Methodology
- Resampled and extracted low level acoustic features from 5K audio files
- Extracted wav2vec2 features for the 5K audio files
- Implemented the architecture mentioned in Pepino et al., (2021)
- Trained the model using only acoustic features (downstream-lla), only wav2vec2 features (downstream-w2v2), and both low level acoustic and wav2vec2 features (downstream-lla_w2v2)
- Datasets: RAVDESS, TESS and CREMA-D

## Model Architecture
- The figure shows the trainable parts in green
- eGeMAPS is a minimalistic acoustic feature set
- (a) uses only w2v2 features while (b) uses both w2v2 and eG2MAPS features

## Experimental Results
- wav2vec2-fine_tuned: 0.79
- downstream-lla: 0.65
- downstream-w2v2


 

# WHISPER by OpenAI
OpenAI has developed a weakly supervised technique to generate powerful speech representations using 680K hours of weakly supervised speech data. The multi-task approach includes speech recognition, language recognition, and X speech to English text. WHISPER has shown good generalizing capabilities close to human level performance for “out-of-distribution” data and is robust to variations like noise.

# WHISPER Architecture
The WHISPER architecture consists of a simple Transformer architecture. When compared to Wav2Vec2, WHISPER has a better Word Error Rate (WER) and performs better on “out-of-distribution” data.

# SUPERB Benchmark
SUPERB is a collection of benchmarking resources to evaluate the capability of a universal shared representation for speech processing. The leaderboard is ongoing and the challenges for ERC include basis of emotion annotation, conversational context modeling, speaker specific modeling, listener specific modeling, presence of emotion shift, fine-grained emotion recognition, and presence of sarcasm.


 

## Speech Emotion Recognition
Speech emotion recognition (SER) is an important task for developing intelligent interaction systems. Pre-trained transformer-based models have been found to be useful for SER. Experiments have shown that using both wav2vec2 features and acoustic features are ideal for SER. For multilingual SER, XLSR-wav2vec2 can be used. WHISPER can be used to introduce robustness to SER models. Disfluent features along with word-level VAD values of speech transcripts can enable us to detect emotion shift better. 

## Challenges
Variations in speech such as dialects, accents, noisy environments, and code-mixed and code-switched speech can pose challenges for SER. 

## Possible Approaches
Fine-tuning with adequate annotated data and continual pre-training with adequate unlabeled data and inadequate annotated data are possible approaches for SER. 

## Overall Architecture
The overall architecture for SER includes an ASR, a DNN-based model, acoustic and textual features extraction, disfluency features extraction, and ASR transcripts and VAD values.


 

## Speech Representation Learning

* Advances in Neural Information Processing Systems, 33:12449–12460. 
* SUPERB: Speech Processing Universal PERformance Benchmark (Shu wen Yang et al., 2021)
* Wavlm: Large-scale self-supervised pre-training for full stack speech processing (Chen et al., 2022)
* Unsupervised cross-lingual representation learning for speech recognition (Conneau et al., 2020)
* Conformer: Convolution-augmented transformer for speech recognition (Gulati et al., 2020)

## Emotion Recognition in Conversation

* Emotion recognition in conversation: Research challenges, datasets, and recent advances (Poria et al., 2019)
* EmoCaps: Emotion Capsule based Model for Conversational Emotion Recognition (Li et al., 2022)
* Fusing ASR Outputs in Joint Training for Speech Emotion Recognition (Li et al., 2022)
* Joint Training for Speech Emotion Recognition (Rovetta et al., 2022)


 

## Self Attention Block
This lecture discussed the self-attention block, a classic diagram and paper from NeurIPS 2017. It is used to create a contextual word embedding from a phrase. The example used was "bank of the river". The word embeddings of 'bank', 'of', 'the', and 'river' are V1, V2, V3, and V4 respectively. A score vector is created for each word vector, and a S-matrix is created from these score vectors. The S-matrix is then scaled and a W-matrix is created. The W-matrix is then used to create a Y-matrix, which is a vector of the words in the phrase. The Y1 vector is then interpreted, with the goal of making it have place properties instead of financial properties. This is done by making the V4 vector (corresponding to 'river') stronger through attention.


 

Neural Networks:
- Neural networks use query, key and value weights with learnable parameters.
- The weight w14 should be learnt to create the required query, key and value.
- These weights can be the weights of 3 linear layers of neurons which can be learnt by gradient descent.

Positional Encoding:
- The kth component of the tth position vector is denoted as pos(t,k).
- For even and odd positions, the frequency is 1/(102i/d).

Machine Translation:
- MT is difficult due to language divergence, which includes lexico-semantic and structural divergence.
- MT is based on analysis-transfer-generation and uses NMT to encode the input, enrich it with self-attention, cross-attention and auto-regression.
- Parameters are estimated using bilingual parallel corpora.

Language Model:
- Language model is used to detect good English sentences.
- Probability of an English sentence is calculated using Pr(w1) * Pr(w2|w1) *. . . * Pr(wn|w1w2 . . .).
 difficult problem 
Output : This problem is a mountain to climb.

 

N-Gram Model Probability:
- Here Pr(wn|w1w2 . . . wn-1) is the probability that word wn follows word string w1w2 . . . wn-1.
- Trigram model probability calculation is also used.

Argmax(.): A Very General Framework:
- Argmax(.) is a very general framework used for various tasks such as QA, summarization, ASR, TTS, image captioning, question generation, disfluency correction, POS tagging, chunking, parsing, spell checking, named entity recognition, and dialogue intent.

PaLM Experiments:
- Metaphor to normal conversion and normal to metaphor conversion are tested using prompts.
- Metaphor generation is also tested using prompts.


 

Abstract Underlying Structure
Abstract underlying structure is required to handle center embedding, which is difficult even for humans. Three phenomena display the same abstract pattern: negation, emphasis, and verb phrase ellipsis.

Negation
Negation involves using words such as "didn't" and "shouldn't" to express the opposite of a statement.

Emphasis
Emphasis involves using words such as "did" and "should" to emphasize a statement.

Verb Phrase Ellipsis
Verb phrase ellipsis involves using words such as "did too" to indicate that the same action was performed by two or more people.

Metaphor Generation
Metaphor generation involves using metaphors to express a feeling or situation. Examples include "I'm feeling blue" and "He is a ball of fire".


 

## Classical Language Modelling

Classical language modelling is the study of how sentences are structured and the rules underlying grammaticality. It is used to determine if a string belongs to a language, and assigns probabilities to each sentence. This is used in Natural Language Processing (NLP) tasks such as Machine Translation, Question Answering, Summarization, and Paraphrasing.

An example of a sentence with syntactic correctness but semantic oddity is "Colorless green Ideas sleep furiously". Constituency and dependency parse trees are used to represent the structure of the sentence. Grammar rules are also used to parse sentences, with probability values assigned to each rule.

CYK Parsing is a method used to determine if a string belongs to a language. It starts with a segment of English and assigns probability values to each rule.

 

CYK Algorithm:

The CYK algorithm is used to parse a sentence into its constituent parts. It works by filling diagonals with higher level structures and continuing the diagonal until the sentence is fully parsed. In this example, the sentence "The gunman sprayed the building with bullets" is parsed into its parts: "The" (DT), "gunman" (NN), "sprayed" (VBD), "the" (DT), "building" (NP), "with" (DT), and "bullets" (NN).


 

CYK Algorithm:

The CYK algorithm is a parsing algorithm used to determine the structure of a sentence. It starts by filling the 5th column and then continues to fill the remaining columns. If an 'S' is found, but there is no termination, the algorithm will continue to fill the remaining columns. In the example given, the gunman sprayed the building with bullets. The CYK algorithm was used to determine the structure of the sentence, which was DT NP NN VBD VP DT NP NN.


 

A gunman sprayed bullets at a building.

The gunman was identified using the CYK algorithm, which moves control to the last column. The algorithm identified the gunman as a noun phrase (NP) and the building as a noun (NN). The gunman was then identified as a verb phrase (VP) and the building as a prepositional phrase (PP).


 

CYK Parsing
CYK (Cocke-Younger-Kasami) parsing is a parsing algorithm used to determine the structure of a sentence. It works by filling a matrix with the words of the sentence and then using backpointers to extract the parse tree. This algorithm can be used to create conversational AI, such as InstructGPT and ChatGPT, which can respond to commands and requests and carry out conversations.


 

Gricean Maxims:

Gricean Maxims are a set of principles for cooperative conversation, proposed by philosopher Paul Grice. They include the maxims of Quantity, Quality, Relation, and Manner. The Quantity maxim states that contributions should be neither more nor less than is required. The Quality maxim states that contributions should be truthful and based on adequate evidence. The Relation maxim states that contributions should be relevant to the current exchange. The Manner maxim states that contributions should be clear and avoid obscurity. 

Examples:

For the Quantity maxim, Grice uses the analogy of assisting someone to mend a car, where the contribution should be neither more nor less than is required. For the Quality maxim, Grice uses the analogy of making a cake, where the contribution should be genuine and not spurious. For the Relation maxim, Grice uses the analogy of mixing ingredients for a cake, where the contribution should be appropriate to the immediate needs. For the Manner maxim, Grice uses the analogy of avoiding obscurity of expression. 

Input/Response:

As an example of the maxims in action, if someone says "I have been promoted," an appropriate response would be


 

Gricean Maxims
Gricean maxims are a set of guidelines for communication that help ensure clarity and brevity. They include the maxims of quantity, quality, relation, and manner.

Quantity: This maxim states that the speaker should provide enough information to answer the question, but not too much.

Quality: This maxim states that the speaker should provide true information.

Relation: This maxim states that the speaker should provide information that is relevant to the question.

Manner: This maxim states that the speaker should provide information in a clear and concise way.

ChatGP
ChatGP is a chatbot that uses Gricean maxims to determine the validity of an answer. If an answer is too long, ambiguous, or irrelevant, it is considered a violation of the maxims.



 

## Conversational AI Attempts at Automation

Conversational AI attempts to automate conversations by responding to commands, requests, and orders. ChatGPT is an example of this, which is able to carry out conversations, respect context, and provide personalized responses.

## Gricean Maxims

The Gricean Maxims are a set of principles for cooperative conversations, proposed by Paul Grice, a philosopher of language. These maxims include Quantity, Quality, Relation, and Manner. The Quantity maxim states that contributions should be neither more nor less than is required. The Quality maxim states that contributions should be truthful and based on adequate evidence. The Relation maxim states that information should be relevant to the current exchange.


 

Gricean Maxims

Gricean maxims are a set of rules for communication that help ensure clarity and accuracy. They include:

* Quantity: Provide enough information to answer the question, but not too much.
* Quality: Provide truthful information.
* Relation: Provide information that is relevant to the question.
* Manner: Be clear and concise.

ChatGP "Thinking"

ChatGP is a computer program that uses Gricean maxims to interpret conversations. For example, if someone is asked "Where is the library?" and they answer "Yes, I do," this would be a violation of the Quantity maxim. Similarly, if they answer "Up yonder in the citadel of learning where polynominals are the bread and operators are the butter and where Hardy and Ramanujam permeate the atmosphere, thither will thee find the storehouse of what bibliophiles love (maybe used for humorous effect)," this would be a violation of the Manner maxim.


 

AI Chatbots Comparison:

Chatbots are computer programs that can interact with humans in a conversational manner. Three popular chatbots are Google's Bard, Microsoft's Bing, and OpenAI's ChatGPT. They are compared based on their ability to answer a range of questions, from holiday tips to gaming advice to mortgage calculations. ChatGPT is the most verbally dextrous, Bing is best for getting information from the web, and Bard is doing its best. OpenAI's ChatGPT uses GPT-4, while Bing has other abilities such as generating images, accessing the web, and offering sources for its responses. OpenAI has also announced plug-ins for ChatGPT that will allow it to access real-time data from the internet. For example, when asked for a recipe for chocolate cake, the chatbot can offer creative solutions by shifting the ratio of flour to water to oil to butter to sugar to eggs.


 

## Recipe Bots

Recipe bots are AI language models that combine different recipes to achieve a desired effect. 

### ChatGPT

ChatGPT chose a chocolate cake recipe from one site, a buttercream recipe from another, shared the link for one of the two, and reproduced both of their ingredients correctly. It even added some helpful instructions, like suggesting the use of parchment paper and offering some tips on how to assemble the cake’s layers.

### Bing

Bing gets in the ballpark but misses in some strange ways. It cites a specific recipe but then changes some of the quantities for important ingredients like flour, although only by a small margin. For the buttercream, it fully halves the instructed amount of sugar to include.

### Bard

Bard makes some changes that meaningfully affect flavor, like swapping buttermilk for milk and coffee for water. It also fails to include milk or heavy cream in its buttercream recipe, so the frosting is going to end up far too thick. It is not recommended to ask Bard for a hand in the kitchen.

## Installing RAM

The instructions should guide people to their motherboard manual to ensure RAM is being


 

## Optimizing RAM Performance
When building a PC, it is important to enable the BIOS settings to optimize RAM performance. The advice given is solid but basic, and could have been improved by including BIOS changes and dual-channel parts. 

## Poetry
Anapestic tetrameter is an arcane meter, as demonstrated by the poem “Twas the night before Christmas”. 

## Question Answering
ChatGPT/GPT-4 was the best at answering questions about passages taken from fiction, as it was able to parse nuances and make human-like inferences. Bard was able to identify the source text, but was not as specific. 

## Basic Math
None of the chatbots were able to determine the monthly repayments and total repayment for a mortgage of $125,000 repaid over 25 years at 3.9 percent interest. GPT-4 was consistent but failed the task due to its long-winded explanation. ChatGPT and Bing got the 20% increase of 2,230 correct, but not BARD. Bing booted the user to a mortgage calculator site when asked about mortgages, and ChatGPT's forthcoming plugins include a Wolfram Alpha


 

## Math Model
When dealing with complicated sums, it is best to use a calculator rather than relying on a language model.

## Average Salary for Plumbers in NYC
ChatGPT gave a ballpark figure, explained that there were caveats, and told about what sources one could check for more detailed numbers. Bing gave specific numbers, cited its sources, and even gave links. Bard gave a lot of hallucination and made up two different sources to attribute a number to.

## Designing a Training Plan to Run a Marathon
ChatGPT is the winner in the race to make a marathon training plan, while Bing barely bothered to make a recommendation and linked out to a Runner’s World article. Bard's plan was confusing, as it promised to lay out a three-month training plan but only listed specific training schedules for three weeks.


 

## Running Plans

A running plan was discussed that gradually increases mileage over three months, with schedules and general tips provided.

## Chat Bots

Three chat bots were tested for their ability to provide holiday tips for Rome. They were found to be broad in their choices, but good for getting away from the busiest areas.

## Reasoning Test

A reasoning test was conducted to find the diamond in a story. Bard and Bing sometimes got the answer right, and ChatGPT occasionally got it wrong. The results do not prove or disprove that these systems have reasoning capability.
 actually 
made from sand, soda ash, and limestone.

 

History of Glassmaking
Glass was not invented by a shipwrecked captain as described in the story. The history of glassmaking goes back thousands of years and involves the contributions of many different cultures. Glass is actually made from sand, soda ash, and limestone, not potatoes.


 

The Invention of Glass

The story of the invention of glass is not a reliable source of information. Glass was actually first produced by the ancient Mesopotamians around 3500 BCE. The process of making glass is complex and requires specialized knowledge and equipment. The story is written in a simplistic and unconvincing manner, with several inconsistencies and unrealistic details.

The three stages of LLM based CAI are Generative Pretraining (GP), Supervised Fine Tuning (SFT), and Reinforcement Learning based on Human Feedback (RLHF). Dialogue Act Classification (DAC) and Dialogue Intent are also used to analyze dialogue sequences and dialogue turns with intent.


 

Natural Language Processing (NLP) is a field of study that involves understanding and processing human language. It consists of several layers, such as morphology, POS tagging, chunking, parsing, semantics, pragmatics, and discourse. Generative models are used to observe words from tags as states, similar to Hidden Markov Models (HMM). Pragmatics can constrain semantics, for example, in the sentence "The gunman sprayed the building with bullets", the former meaning is more likely. This is corroborated by data from parse t1 and t2. Examples of pragmatics constraining semantics include the improbable meaning of "Command Center to Track Best Buses" and the increased risk of Covid-19 for elderly people with young faces.


 

## Pragmatics
Pragmatics is the study of language in use, and is concerned with the meaning of a sentence in a particular context. It is distinct from lexical semantics (word meanings) and sentential semantics (truth value of a sentence and entailment). Pragmatics is extra-sentential and arises due to the limitations of lexical and formal semantics. 

Examples of pragmatics include dialogue or conversation settings, where the implication of a response can be different from the literal meaning, and politeness, where the same request can be expressed in different ways. 

Elements of pragmatics include deixis (pointing with words), presupposition, speech acts, implicatures, politeness, and information structure. The Sanskrit tradition of Shabdshakti (power inherent in word) is also related to pragmatics.


 

Meaning of Hall:
The hall is packed (avidha), burst into laughing (lakshana) and full (vyanjana).

Vachyartha, Lakshyartha, Vyangaartha:
The river Gangaa (abhidhaa), house on river gangaa (lakshanaa) and house will have nice view, breeze etc. (vyanjana).

Pragmatics:
Sentence vs. Utterance, Semantics + Intent  Pragmatics. The Trinity of Pragmatics includes Speaker, Hearer and Linguistic Expression.

Communicative Aspects of Language:
Linguistics, Psychology and Sociology/Anthropology have all focused on different aspects of communication, but none have fully explored the nature of communication itself.

Syntax and Semantics not Enough:
Communicative process does not end with processing structural properties and decoding meaning. Ambiguity and Reference are examples of problems beyond the reach of plain syntax and semantics.


# Deixis
Deixis is a universal phenomenon across languages, used to individuate objects in the immediate context in which they are uttered, by pointing at them so as to direct attention to them. This results in the Speaker (Spr) and Addressee (Adr) attending to the same referential object. 

## Intention
Intention can be expressed through language, such as a promise ("meitumhe bataataa hu") or a threat ("I will teach you a lesson"). These problems are beyond the reach of plain syntax and semantics.

## Non-Literality
Non-literality includes sarcasm and metaphor, such as "I love being ignored".

## Indirection
Indirection is used to communicate a message that is not explicitly stated. For example, when a speaker says "My car has a flat tire" to a car mechanic, they are not just stating a fact, but are asking for help.

## Non-Communicative Acts
Non-communicative acts are used to communicate a message that has a normative, formal standing. An example of this is "I pronounce you man and wife", which is used to legalize a marriage.

##

 

## Speech Acts
Speech acts are expressions that not only present information, but also perform an action. There are four types of speech acts: locutionary, illocutionary, perlocutionary, and performative. 

### Locutionary Speech Act
Locutionary speech acts are the meaning that is on the surface of the utterance. For example, "It is raining" is a statement of fact. In Bengali, classifiers are used to introduce definitiveness and shared understanding between the speaker and the addressee. 

### Illocutionary Speech Act
Illocutionary speech acts are when by saying something, we do something. An example is when someone says "I am hungry" and the listener goes to the fridge to get them something to eat. 

### Perlocutionary Speech Act
Perlocutionary speech acts always have a perlocutionary effect, which is the effect a speech act has on a listener. An example is when someone says "I am hungry" and the listener goes to the fridge to get them something to eat. 

### Performative Speech Act
Performative speech acts are when the action that the sentence


# Speech Acts
Speech acts are utterances that have a communicative purpose. They can be divided into four categories: assertives, directives, commissives, and expressives. 

## Assertives
Assertives are utterances that commit the speaker to something being the case. Examples include suggesting, putting forward, swearing, boasting, and concluding. 

## Directives
Directives are attempts by the speaker to get the addressee to do something. Examples include asking, ordering, requesting, inviting, advising, and begging. 

## Commissives
Commissives are utterances that commit the speaker to some future course of action. Examples include promising, planning, vowing, betting, and opposing. 

## Expressives
Expressives are utterances that express the psychological state of the speaker about a situation. 

## Self-Reference
Speech acts can have self-reference, meaning that the utterance itself is the action. An example is "I promise to pay you back," where the utterance itself is the promise. 

## Subtle Differences
The subtle differences between illocutionary, perlocutionary, and performative speech acts can be seen


 

## Dialogue Act Classification
Dialogue act classification (DAC) is a computational approach to understanding conversations. It involves assigning labels to dialogue turns, such as 'question', 'elaboration', 'affirmation', and 'command/request'. It also involves assigning intent to dialogue sequences.

## Emotion-Aided Multi-Modal Dialogue Act Classification
Tulika Saha, Aditya Patra, Sriparna Saha, and Pushpak Bhattacharyya investigated the role of emotion and multi-modality in determining dialogue acts (DAs) of an utterance. They created a novel dataset, EMOTyDA, containing emotion-rich videos of dialogues.

## Speaker Turn Modeling for Dialogue Act Classification
Zihao He, Leili Tavabi, Kristina Lerman, and Mohammad Soleymani proposed a speaker turn modeling approach for dialogue act classification. This approach was presented at the Association for Computational Linguistics: EMNLP 2021.


 

## Dialogue Act Classification (DAC) and Multimodality
DAC is used to identify intent and each turn is primarily a question, statement, or request for action. Prior work on DAC dates back to the late 1990s and early 2000s. Recent work has used deep learning techniques such as stacked LSTMs, hierarchical bi-LSTMs and CRFs, contextual self-attention frameworks, and CNNs.

## Emotion and Dialogue
Non-verbal features such as changes in tone and facial expressions can provide beneficial cues to identify DAs. Emotion-aided multimodal DAC has been proposed, as certain expressions can denote agreement (statement) or disagreement (sarcasm).

## Contributions
The EMOTyDA dataset was created, which consists of short videos of dialogue conversations manually annotated with its DA along with its pre-annotated emotions. An attention-based multi-modal, multi-task framework was proposed for joint optimization of DAs and emotions. Results showed that multi-modality and multi-tasking boosted the performance of DA identification compared to its unimodal and single task DAC variants. Future plans include incorporating conversation history, speaker information, and fine


 

EMOTyDA is a dataset of 1341 dyadic and multi-party conversations, resulting in 19,365 utterances with corresponding dialogue act (DA) and emotion tags. The dataset is composed of a subset of 1039 dialogues from MELD and the entire IEMOCAP dataset of 302 dialogues. The utterances were manually annotated using the SWBD-DAMSL tag-set consisting of 42 DAs. Out of these, 12 most commonly occurring tags were selected. Three annotators with graduate-level English proficiency were assigned to annotate the utterances, with an inter-annotator score of more than 80% considered as reliable agreement.

For feature extraction, text transcripts of each video were concatenated with pretrained GloVe, while OpenSMILE was used for audio. This included 12 Mel-frequency coefficients, glottal source parameters, maxima dispersion quotients, several low-level descriptors, voice intensity, MFCC, voiced/unvoiced segmented features, pitch and their statistics, and voice quality.


 

## CS772: Deep Learning for Natural Language Processing (DL-NLP)

This week's lecture covered the topics of summarization, opinion summarization, and DNN. The lecture was given by Pushpak Bhattacharyya from the Computer Science and Engineering Department at IIT Bombay.

### Gricean Maxims

Gricean Maxims are the Cooperative Principle in Conversation, as described by philosopher of language Paul Grice. They include Quantity, Quality, Relation, and Manner, and capture the link between utterances.

### AI Chatbots Compared

The lecture discussed the comparison of Google's Bard, Microsoft's Bing, and OpenAI's ChatGPT.

### 3 Stages of LLM Based CAI

The lecture discussed the 3 stages of LLM based CAI: Generative Pretraining (GP), Supervised Fine Tuning (SFT), and Reinforcement Learning from Human Feedback (RLHF).

### Pragmatics Modeling

The lecture discussed Pragmatics Modeling, which includes Dialogue Act Classification (DAC), Dialogue Intent, Deixis, Presupposition, Speech Acts, Implicatures, and


 

## Summarization
Summarization is the task of automatically creating a compressed version of a text document (e.g. set of tweets, web page, single/multi-document) that is relevant, non-redundant, and representative of the main idea of the text. The metric used to measure summarization is the compression ratio, which is the number of words in the summary divided by the number of words in the document.

## Categorization
Summarization can be broadly categorized into extractive and abstractive summarization. Extractive summarization involves selecting sentences from the input text to form the summary, while abstractive summarization involves generating a summary that conveys the essence of the text using natural language generation. Other categorizations include single/multi-document, generic/query-focused, personalized, sentiment-based, update, e-mail-based, and web-based summarization.

## NLP Layer
Summarization involves various levels of natural language processing (NLP), including lexical, syntax, semantics, and pragmatics.

## Speech Acts
Speech acts are a type of linguistic expression that involve a speaker, hearer, locutionary


 

### Morphology Generation
- Morphology generation can be solved using Byte Pair Encoding (BPE), which divides a string into subwords and assigns each subword its own probability.
- If subwording is not used, all forms of the root word must be shown (e.g. go, went, going, gone).
- Languages differ in morphological complexity, with French being more complex than English.

### Computation of Summaries
- Hierarchical Encoder-Decoder and SummaRuNNer (a two-layer RNN based sequence classifier) can be used for summarization.
- Pointer-Generator Network can also be used for summarization.
- Generative Pre-training, Supervised Fine Tuning, and Reinforcement Learning with Human Feedback (RLHF) are used for opinion/review summarization.
- BERT (12 layers) and GPT (12 layers) are pre-trained on 160GB of news, books, and web text, and fine-tuned on CNN/DM dataset.
- Properties of opinion summaries include monotonicity (subjectivity increases with more sentences) and diminishing return (lower intensity sentences


 

## Submodular Function
Submodular functions are used to find a set of sentences in a document that maximizes a submodular function subject to budget constraints. The total utility of the summary is composed of two parts: relevance and non-redundancy. Relevance measures the coverage of the summary set to the document, while non-redundancy rewards diversity in the summary. As soon as an element is selected from a cluster, other elements from the same cluster start having diminishing return.

## Pointer Generator Network
The Pointer Generator Network proposed by Abigail See, Peter J. Liu, and Christopher D. Manning in 2017 is a novel architecture for summarization. It uses a combination of extractive and abstractive techniques to generate summaries.


 

## Hybrid Pointer-Generator Network
A hybrid pointer-generator network is a sequence-to-sequence attentional model that augments the standard model in two ways. First, it uses a hybrid pointer-generator network that can copy words from the source text via pointing, which aids accurate reproduction of information while retaining the ability to produce novel words through the generator. Second, it uses coverage to keep track of what has been summarized, which discourages repetition.

## Modeling
Input processing involves tokens being fed one-by-one into the encoder (a single-layer bidirectional LSTM), producing a sequence of encoder hidden states. At each step, the decoder (a single-layer unidirectional LSTM) receives the word embedding of the previous word. The encoder hidden states are used to produce a probability distribution over the source words, known as the attention distribution. This is used to produce a weighted sum of the encoder hidden states, known as the context vector. The context vector is concatenated with the decoder state and fed through two linear layers to produce the vocabulary distribution. This provides the final distribution from which to predict words. The loss


 

### Loss Calculation:
- The loss for a timestep is the negative log likelihood of the target word for that time step.
- The overall loss for the whole sequence is calculated.

### Pointer Generator Network:
- Allows both copying words via pointing and generating words from a fixed vocabulary.
- Generation probability pgen is calculated from the context vector, decoder state and decoder input.

### Generator Probability:
- Generation probability pgen is used as a soft switch to choose between generating a word from the vocabulary or copying a word from the input sequence.
- If the word is out-of-vocabulary or does not appear in the source document, the probability is zero.

### ROUGE vs BLEU:
- ROUGE incorporates recall and handles incorrect words better than BLEU.
- ROUGE consists of ROUGE-N, ROUGE-L, ROUGE-W and ROUGE-S.
- ROUGE-L considers longest common subsequence, ROUGE-W considers all common subsequences with weight based on length and ROUGE-S matches skip bigrams.


 

## Information

- Information is a set of facts or data used to make decisions, form knowledge, and gain understanding.
- It serves as the raw material for decision making and is the foundation of knowledge.


 

# CS772: Deep Learning for Natural Language Processing (DL-NLP)

Course Summary:

This course is taught by Pushpak Bhattacharyya from the Computer Science and Engineering Department at IIT Bombay. It covers topics related to natural language processing (NLP) such as morphology, part-of-speech tagging, parsing, semantics, discourse and coreference. It also covers single neuron, perceptron and sigmoid applications to NLP, multilayered FFNN, backpropagation, softmax applications to NLP, recurrent neural net (RNN) applications to NLP-seq2seq, recursive neural net applications to NLP parsing, convolutional neural nets, multimodal NLP, transformers, machine translation and MT evaluation, conversational AI and pragmatics.

Evaluation Scheme (tentative):

- 40%: Reading, Thinking, Comprehending (Quizzes (20%) (4 nos.), Endsem (20%))
- 60%: Doing things, Hands on (Assignments (20%), Course Project (40%))
- Quizzes and Endsem: ONE/TWO subjective questions -only


 

## Perceptron Model
A perceptron is a computing element with input lines having associated weights and the cell having a threshold value. It is motivated by the biological neuron and its output is denoted by y. The Perceptron Training Algorithm (PTA) converges if the vectors are from a linearly separable function, with |G(Wn)| bounded and n tending to infinity.

## Sigmoid Neuron
The sigmoid neuron is a single neuron with input xi and weights w1. It has a sigmoid or logit function which can saturate in case of extreme agitation or emotion. The output of sigmoid is between 0-1 and can be used to decide between two classes (C1 and C2).

## Softmax Neuron
The softmax neuron is used for multi-class classification (C classes). It has a softmax function which takes input vector Z of size K and gives the ith component of the output vector. The output for class c (small c) is given by the RHS.

## Weight Change Rule
The weight change rule for a single sigmoid neuron is given by the equation x0x


# Summary

This lecture discussed the weight change rule with Total Sum Square (TSS) loss and the general backpropagation rule. 

## Single Neuron: Sigmoid + Total Sum Square (TSS) Loss

For a single neuron, the weight change rule is given by:

### $$\Delta w_1 = -\eta \frac{\delta L}{\delta w_1}$$

where $\eta$ is the learning rate, $L$ is the loss, $t$ is the target, and $o$ is the observed output. 

## Multiple Neurons: Sigmoid + Total Sum Square (TSS) Loss

For multiple neurons in the output layer, the weight change rule is given by:

### $$\Delta w_{11} = \eta (t_1 - o_1) o_1 (1 - o_1) x_1$$

where the target vector is $\langle t_1, t_0 \rangle$ and the observed vector is $\langle o_1, o_0 \rangle$. The TSS loss is given by:

### $$L = \frac


 

## Word2vec Architectures

Word2vec is a neural network architecture developed by Mikolov in 2013. It is used to model the probability of a context word given an input word. The architecture consists of two layers: an input layer and a projection layer. The input layer takes in a word sequence and the projection layer outputs a word vector. The word vector is derived by optimizing the weights of the network using gradient descent.

The word vector is initialized by taking the weight vector from the input neuron to the projection layer (Udog) and the weight vector from the projection layer to the output neuron (Ubark). The probability of a context word given an input word is modeled by computing the dot product of Udog and Ubark, exponentiating the dot product, and taking the softmax over all dot products in the vocabulary.


 

## Classic Work
Caught the attention of the world by equations like ‘king’-’man’+’woman ’=‘queen’ in N-dimensional space.

## Symbolic Approach to Representing Word Meaning
Syntagmatic and paradigmatic relations, such as lexico-semantic relations (synonymy, antonymy, hypernymy, mernymy, troponymy, etc.) and co-occurence, are used to capture semantics. Resources such as Wordnet and ConceptNet are used to capture these relations.

## Two Main Models for Learning Word Vectors
1. Global matrix factorization methods, such as latent semantic analysis (LSA).
2. Local context window methods, such as the skip-gram model of Mikolov et al. (2013).

## Drawbacks
Matrix factorization: most frequent words contribute a disproportionate amount to the similarity measure. Skip Gram & CBOW: shallow window-based methods fail to take advantage of the vast amount of repetition in the data.


 

### Dimensionality Reduction with PCA

PCA is a technique used to reduce the number of dimensions in a dataset. It is based on the idea that the data can be represented in fewer dimensions while still preserving the essential information. This is done by finding the principal components of the data, which are the directions of maximum variance.

### Intuition for Dimensionality Reduction

The intuition behind dimensionality reduction is that it is possible to classify data points in fewer dimensions than the original data. For example, if we have four points in two dimensions, it is easy to set up a hyperplane that will separate the two classes. However, if we form another attribute of these points, such as the distance of their projections on the line from the origin, then the points can be classified by a threshold on these distances. This effectively is classification in the reduced dimension (1 dimension).

Example: IRIS Data

The IRIS dataset contains 150 instances with 4 attributes (petal length, petal width, sepal length, sepal width). To reduce the dimensionality of the data, 80% of the data (120 instances) is used for training and the remaining 30 instances are used for testing. It


 

## Principal Component Analysis
Principal Component Analysis (PCA) is a technique used to reduce the dimensionality of data by extracting the most important features. It is used to study the classification of data by predicting the fate from body characteristics.

The data is first standardized by subtracting the mean and dividing by the standard deviation. Then, the principal components are calculated by taking a linear combination of the standardized data. The first two principal components are used as the new data for classification.

The importance of each principal component is determined by the eigenvalues. The first eigenvalue is the most important and accounts for 72% of the total variance. The second, third, fourth, and fifth eigenvalues account for 10.6%, 7.7%, 6.0%, and 3.3% of the total variance, respectively.

Word2Vec is a neural network used to compute the weights between neurons. The weights are calculated by taking the difference between the output of the hidden neuron and the output of the output neuron.


 

# Word2vec Network
Word2vec is a neural network that uses two RNN networks, the encoder and the decoder, to encode and decode a sentence. The encoder processes one input at a time and generates a representation of the sentence, which is used to initialize the decoder state. The decoder then generates one element at a time until the end of sequence tag is generated. 

Weights go from all neurons to all neurons in the next layer, and the capital letter is used for the name of the neuron, while the small letter is used for the output from the same neuron. The weight change for input to hidden layer, say, wH0U0, is calculated by subtracting the weight change for the output layer, say, V1, due to input U0.


 

# Sequence to Sequence Learning with Neural Networks
Sequence to Sequence Learning is a type of machine learning that uses neural networks to map an input sequence to an output sequence. It is used in tasks such as machine translation, image captioning, and speech recognition. This technique is based on Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks, which use a softmax activation function to capture the output.

## Decoding
The decoder is responsible for generating the output sequence from the input sequence. This is done by incrementally constructing hypotheses and scoring them using the model. The hypotheses are maintained in a priority queue, and the final hypothesis is generated by expanding the partial hypothesis.

## Encoding-Attend-Decode Paradigm
The source sentence is represented by a set of output vectors from the encoder, called annotation vectors. Each output vector is a contextual representation of the input at a given time. To capture the syntactic and semantic complexities, a richer representation for the sentences is used. To address the issue of long-term dependencies, the source sentence information is made available when making the next prediction.

## Convolutional Neural Networks
Con


 

## Feed-Forward Backpropagation (FF-BP)
FF-BP is useful to understand Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM). Input is divided into regions and fed forward, with the filter parameters remaining the same as the window slides over the input. This is similar to the Neocognitron (Fukushima, 1980).

## A Typical ConvNet
LeCun, Bengio, and Hinton (2015) proposed a typical ConvNet, which became a rage due to its ability to classify images with 1,000 different classes from the web. It almost halved the error rates of the best competing approaches.

## Learning in CNN
CNNs automatically learn the values of its filters. For example, in image classification, it learns to detect edges from raw pixels in the first layer, then use the edges to detect simple shapes in the second layer, and then use these shapes to detect higher-level features, such as facial shapes in higher layers. The last layer is then a classifier that uses these high-level features.

## CNN for NLP
CNN


 

## Attention Mechanism
Attention is a mechanism used in natural language processing (NLP) to identify relevant information in a given text. It is used to create a vector representation of the text, which is then used to generate a summary.

### Word Embedding and Contextual Word Embedding
Word embedding is a technique used to represent words as vectors. This vector representation is used to create a score vector for each word vector. This score vector is then used to create a matrix, which is then scaled and weighted to create a soft vector.

### Attention Block
The attention block is composed of a query, key, and value. The query, key, and value are used to create a soft vector, which is then used to create a matrix. This matrix is then used to create a vector, which is then used to create a matrix. This matrix is then used to create a vector, which is then used to create a matrix.

### Attempts at Automation
InstructGPT and ChatGPT are two attempts at automating the attention mechanism. InstructGPT is used to generate a response to a command or request, while ChatGPT is used to carry out a


 

## Gricean Maxims

Gricean Maxims are a set of principles for cooperative conversation, proposed by philosopher Paul Grice. They are:

* Quantity: Make your contribution as informative as is required for the current purposes of the exchange, but no more.
* Quality: Do not say what you believe is false or for which you lack adequate evidence.
* Relation: Information should be relevant to the current exchange, omitting any irrelevant information.
* Manner: Be perspicuous and avoid obscurity, ambiguity, and other forms of confusion.

Grice's analogy for these maxims is that if you are helping someone mend a car, you should provide the exact number of screws required, not more or less. Similarly, you should provide genuine ingredients for a cake, not salt instead of sugar or a rubber spoon instead of a real one.


 

AI Chatbots:
- Google's Bard, Microsoft's Bing, and OpenAI's ChatGPT are three AI chatbots that can be compared.
- Long-Short Term Memory (LSTM) based Conversational AI (CAI) has three stages: Generative Pretraining (GP), Supervised Fine Tuning (SFT), and Reinforcement Learning from Human Feedback (RLHF).

Pragmatics Modeling:
- Dialogue Act Classification (DAC) and Dialogue Intent are two elements of pragmatics modeling.
- Deixis, Presupposition, Speech Acts, Implicatures, Politeness, and Information Structure are other elements of pragmatics modeling.


 

Summarization is the task of automatically creating a compressed version of a text document (e.g. set of tweets, web-page, single/multi-document) that should be relevant, non-redundant and representative of the main idea of the text. The metric used to measure summarization is the compression ratio, which is the number of words in the summary divided by the number of words in the document. Summarization can be broadly categorized into extractive and abstractive summarization. Extractive summarization involves selecting sentences from the input text to form the summary, while abstractive summarization involves generating a summary that conveys the essence of the original text using natural language generation. Other categorizations include single/multi-document summarization, generic/query-focused summarization, personalized/sentiment-based/update/e-mail-based/web-based summarization. Morphology generation can be solved using byte pair encoding (BPE), which divides a string into subwords and assigns each subword its own probability. Languages differ in morphological complexity, with French being more complex.


 

## Computation of Summaries

Summarization is the process of reducing a text to its most important points. This can be done using various techniques such as hierarchical encoder-decoder, SummaRuNNer, pointer-generator network, BART, BERT, GPT, and reinforcement learning with human feedback (RLHF).

## Opinion/Review Summaries

Opinion summaries have two properties: monotonicity and diminishing return. Monotonicity states that as more sentences are added to the summary, the subjectivity and information content increases. Diminishing return states that if multiple sentences of varying intensity are added to the summary, the effect of lower intensity sentences is diminished in the presence of higher intensity sentences.

For example, if the budget allows for only two subjective sentences, picking up A and B will capture only batting. Picking up C with one of A and B will cover both aspects (i.e. batting and bowling). Sentences should not overlap in aspects, so no information is lost.


 

## Submodular Functions
Submodular functions are a type of set function that have diminishing returns. They are defined as a function F: 2V→R, where F(φ)= 0, and for all A, B ⊂V, F (A) + F (B) ≥ F (A ∩ B) + F (A ∪B). This is equivalent to the statement that for all A ⊂B, ∀k ∉A, F (A ∪{k}) − F (A) ≥ F (B ∪{k}) − F (B). Examples of submodular functions include cut functions, set cover, and extractive summarization.

## Monotone Submodular Objective
The total utility of a summary is calculated using a monotone submodular objective, which is a combination of relevance (L(S)) and diversity (R(S)).

## Pointer Generator Network
The Pointer Generator Network (See et al., 2017) is a novel architecture that augments the standard sequence-to-sequence attentional model in two ways. First, it uses a hybrid pointer-generator network that can copy


 

## Pointer Generator Network
The Pointer Generator Network is a model used to generate a probability distribution over a fixed vocabulary, allowing for both copying words via pointing and generating words from the vocabulary. It consists of an encoder, decoder, and attention mechanism. 

At test time, the decoder takes the previous word emitted by the decoder and its decoder state as input. The encoder hidden states are used to produce an attention distribution, which is then used to produce a weighted sum of the encoder hidden states known as the context vector. This context vector is concatenated with the decoder state and fed through two linear layers to produce the vocabulary distribution. 

During training, the loss for each time step is the negative log likelihood of the target word for that time step. The generation probability pgen is used as a soft switch to choose between generating a word from the vocabulary or copying a word from the input sequence. It is calculated from the context vector, decoder state, and decoder input.
