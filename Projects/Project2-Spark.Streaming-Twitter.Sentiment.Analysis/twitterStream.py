from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt


def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")

    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
	"""
	Plot the counts for the positive and negative words for each timestep.
	Use plt.show() so that the plot will popup.
	"""
	# YOUR CODE HERE
	ax=plt.subplot(111)
	positive_count, negative_count = [], []
	for x in counts:
		if x:
			positive_count.append(x[0][1])
			negative_count.append(x[1][1])
	
	print len(positive_count)
	x= range(0,len(positive_count))
	ax.plot(x,positive_count,'bo-',label = "positive")
	ax.plot(x,negative_count,'go-', label = "negative")
	ymax = max(max(positive_count),max(negative_count))+50
	ax.set_ylim([0,ymax])
	plt.xlabel("Time step")
	plt.ylabel("Word count")		
	plt.legend(fontsize = 'small',loc=0)
	plt.savefig("plot.png")
	plt.show()
    
    
    
def load_wordlist(filename):
	""" 
	This function should return a list or set of words from the given filename.
	"""
	# YOUR CODE HERE
	words = []
	f = open(filename, 'r')
	for i in f:
		words.append(i.strip())
	return words



def stream(ssc, pwords, nwords, duration):
	def updateFunction(newValues, runningCount):
		if runningCount is None:
			runningCount = 0
		return sum(newValues, runningCount)
		
		
	kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
	tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
	w = tweets.flatMap(lambda line: line.split(' ')).map(lambda str: ('positive', 1) if str in pwords else ('negative', 1) if str in nwords else ('none', 1)) \
            .filter(lambda x: x[0]=='positive' or x[0]=='negative').reduceByKey(lambda x, y: x + y)

	updatedWords = w.updateStateByKey(updateFunction)
	updatedWords.pprint()
    
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
	counts = []
    # YOURDSTREAMOBJECT.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
	w.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))

	ssc.start()                         # Start the computation
	ssc.awaitTerminationOrTimeout(duration)
	ssc.stop(stopGraceFully=True)

	return counts


if __name__=="__main__":
    main()