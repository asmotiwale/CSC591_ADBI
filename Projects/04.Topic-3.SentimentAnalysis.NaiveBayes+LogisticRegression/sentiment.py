import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    #Remove stopwords from the data sets.
    filtered, data_pos, data_neg, remove, final_words = [], [], [], [], []
    train_neg_vec, train_pos_vec, test_neg_vec, test_pos_vec = [], [], [], []
    feat_dict = {}
    for words in train_pos:
    	remove = [w for w in words if w not in stopwords]
    	filtered.extend(remove)
    	data_pos.append(remove)
    for words in train_neg:
    	remove = [w for w in words if w not in stopwords]
    	filtered.extend(remove)
    	data_neg.append(remove)

    filtered = list(set(filtered))
    positive = [word for sub in data_pos for word in sub]
    negative = [word for sub in data_neg for word in sub]


    positive_word_count = dict.fromkeys(list(set(positive)), 0.0)
    negative_word_count = dict.fromkeys(list(set(negative)), 0.0)

    def keep_count(datasets, count_dict):
    	for words in datasets:
    		for w in set(words):
    			count_dict[w] += 1
    	return count_dict		

    positive_word_count = keep_count(data_pos, positive_word_count) 		
    negative_word_count = keep_count(data_neg, negative_word_count)

    #Filtering the words present only in 1% of the positive or negative texts.
    for w in filtered:
    	positive_count = positive_word_count.get(w, 0)
    	negative_count = negative_word_count.get(w, 0)
    	if(positive_count >= (len(train_pos)/100) or negative_count >= (len(train_neg)/100)):
    		final_words.append(w)
    

    #Filtering the words which are at least twice as many postive texts as negative texts, or vice-versa.
    for w in final_words:
    	positive_count = positive_word_count.get(w, 0)
    	negative_count = negative_word_count.get(w, 0)
    	if not((positive_count >= (negative_count * 2)) or (negative_count >= (positive_count * 2))):
    		final_words = [x for x in final_words if x != w]


    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    count = 0
    for w in final_words:
    	feat_dict[w] = count
    	count += 1


    def feature_vector(data, data_vec):
    	for words in data:
    		temp = [0] * len(final_words)
    		for w in words:
    			if(w in feat_dict.keys()):
    				temp[feat_dict[w]] = 1
    		data_vec.append(temp)		

    feature_vector(data_pos, train_pos_vec)
    feature_vector(data_neg, train_neg_vec)
    feature_vector(test_pos, test_pos_vec)
    feature_vector(test_neg, test_neg_vec)	

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    def label_Reviews(rev, type):
        labeled = []
        for i in range(1, len(rev) + 1):
            label = '%s_%s'%(type, i)
            labeled.append(LabeledSentence(rev[i - 1], [label]))
        return labeled

    labeled_train_pos = label_Reviews(train_pos, 'TRAIN_POS')
    labeled_train_neg = label_Reviews(train_neg, 'TRAIN_NEG')
    labeled_test_pos = label_Reviews(test_pos, 'TEST_POS')
    labeled_test_neg = label_Reviews(test_neg, 'TEST_NEG')        
 
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    def extract_feature_vector(rev, type):
        extract = []
        for i in range(1, len(rev) + 1):
            label = '%s_%s' %(type, i)
            extract.append(model.docvecs[label])
        return extract    

    train_pos_vec = extract_feature_vector(train_pos, 'TRAIN_POS')
    train_neg_vec = extract_feature_vector(train_neg, 'TRAIN_NEG')
    test_pos_vec = extract_feature_vector(test_pos, 'TEST_POS')
    test_neg_vec = extract_feature_vector(test_neg, 'TEST_NEG') 

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = sklearn.naive_bayes.BernoulliNB(alpha = 1.0, binarize = None)
    nb_model.fit((train_pos_vec + train_neg_vec), Y)

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit((train_pos_vec + train_neg_vec), Y)

    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = sklearn.naive_bayes.GaussianNB()
    nb_model.fit((train_pos_vec+train_neg_vec),Y)

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit((train_pos_vec+train_neg_vec),Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    positive_predicted = model.predict(test_pos_vec)
    negative_predicted = model.predict(test_neg_vec)
    tp = sum(positive_predicted == 'pos')
    fn = sum(positive_predicted == 'neg')
    fp = sum(negative_predicted == 'pos')
    tn = sum(negative_predicted == 'neg')

    accuracy = float(tp + tn) / float(tp + tn + fn + fp)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
