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
        #train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = 
        feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        #nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    #evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    #evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



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
    filtered = []
    tweets_pos = []
    tweets_neg = []
    print "Here"
    for line in train_pos:
         removed = [ word for word in line if word not in stopwords]
         filtered.extend(removed)
         tweets_pos.append(removed)
    print 2     
    for line in train_neg:
        removed = [ word for word in line if word not in stopwords]
        filtered.extend(removed)
        tweets_neg.append(removed)
    print 3
    filtered = list(set(filtered)) 
    pos_temp = [x for sublist in tweets_pos for x in sublist]
    neg_temp = [x for sublist in tweets_neg for x in sublist]

    word_count_pos = dict.fromkeys(list(set(pos_temp)),0.0)
    word_count_neg = dict.fromkeys(list(set(neg_temp)),0.0)
    final = []

    
    print len(set(pos_temp))
    print len(set(neg_temp))

    for tweet in tweets_pos:
        for word in set(tweet):
            word_count_pos[word] += 1 

    for tweet in tweets_neg:
        for word in set(tweet):
            word_count_neg[word] += 1 

    for word in filtered:
        pos_count = word_count_pos.get(word,0)
        neg_count = word_count_neg.get(word,0)

        if(pos_count >= (0.01*len(train_pos)) or neg_count>=(0.01*len(train_neg))):
            final.append(word)  

    print len(final)

    for word in final:
        pos_count = word_count_pos.get(word,0)
        neg_count = word_count_neg.get(word,0)

        if not ((pos_count >= 2*neg_count)or (neg_count >= 2*pos_count)):
            final = [word1 for word1 in final if word1 != word]

    #if((pos_count >= (0.01*len(train_pos)) or neg_count>=(0.01*len(train_neg))) and ((pos_count >= 2*neg_count)or (neg_count >= 2*pos_count))):
     #       final.append(word)
    print len(final)
    train_pos_vec = []
    feature_dict = {}
    i=0
    for word in final:
        feature_dict[word] = i
        i+=1
    
    for tweet in tweets_pos:
        temp = [0]*len(final)
        for word in tweet:
            if(word in feature_dict.keys()):
                temp[feature_dict[word]]=1
        train_pos_vec.append(temp)

    train_neg_vec = []
    for tweet in tweets_neg:
        temp = [0]*len(final)
        for word in tweet:
            if(word in feature_dict.keys()):
                temp[feature_dict[word]]=1
        train_neg_vec.append(temp)

    test_pos_vec = []
    for tweet in test_pos:
        temp = [0]*len(final)
        for word in tweet:
            if(word in feature_dict.keys()):
                temp[feature_dict[word]]=1
        test_pos_vec.append(temp)

    test_neg_vec = []
    for tweet in test_neg:
        temp = [0]*len(final)
        for word in tweet:
            if(word in feature_dict.keys()):
                temp[feature_dict[word]]=1
        test_neg_vec.append(temp)

    print "Lengths of vectors"
    print len(train_pos_vec)    
    print len(train_neg_vec)    
    print len(test_pos_vec)    
    print len(test_neg_vec)    


    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    # Return the four feature vectors
    #return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i in range(1,len(reviews)+1):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(reviews[i-1], [label]))
        return labelized

    labeled_train_pos = labelizeReviews(train_pos,'TRAIN_POS')
    labeled_train_neg = labelizeReviews(train_neg,'TRAIN_NEG')
    labeled_test_pos = labelizeReviews(test_pos,'TEST_POS')
    labeled_test_neg = labelizeReviews(test_neg,'TEST_NEG')
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

    def extract_feature_vector(reviews,label_type):
        extracted = []
        for i in range(1,len(reviews)+1):
            label = '%s_%s'%(label_type,i)
            extracted.append(model.docvecs[label])
        return extracted

    train_pos_vec = extract_feature_vector(train_pos,'TRAIN_POS')
    train_neg_vec = extract_feature_vector(train_neg,'TRAIN_NEG')
    test_pos_vec = extract_feature_vector(test_pos,'TEST_POS')
    test_neg_vec = extract_feature_vector(test_neg,'TEST_NEG')

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    
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
    nb_model = sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit((train_pos_vec+train_neg_vec),Y)

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit((train_pos_vec+train_neg_vec),Y)
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
    #print "evaluating"
    predicted_pos = model.predict(test_pos_vec)
    predicted_neg = model.predict(test_neg_vec)
    tp = sum(predicted_pos == 'pos')
    fn = sum(predicted_pos == 'neg')
    tn = sum(predicted_neg == 'neg')
    fp = sum(predicted_neg == 'pos')
    
    accuracy = float(tp+tn) / float(tp+tn+fp+fn )
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()