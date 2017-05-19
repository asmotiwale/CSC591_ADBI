import igraph
from igraph import *
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import scipy

# Taking the alpha input.
print "Please enter the value for alpha: "
alpha = input()

# Reading the data sets to generate a graph and attribute table.
attributes = pd.read_csv('/Users/Warrior/Desktop/MS/Sem2/ADBI/PROJECTS/P6/06.Topic-7.Project-6.MarketSegmentation.AttributedGraphCommunityDetection/data/fb_caltech_small_attrlist.csv', header = None)
g = Graph.Read_Edgelist('/Users/Warrior/Desktop/MS/Sem2/ADBI/PROJECTS/P6/06.Topic-7.Project-6.MarketSegmentation.AttributedGraphCommunityDetection/data/fb_caltech_small_edgelist.txt', directed = False)

# Adding the attributes to the graph.
for column in attributes:
	g.vs[attributes[column][0]] = map(int, list(attributes[column][1:]))

#print g.vs["major228"]
# Initial membership list.
global membership
membership = range(324)

similarity_matrix = [[0 for i in range(1, len(attributes))] for j in range(1, len(attributes))]
# Method to compute the similarity matrix.
def compute_cosine_similarity():
	for i in range(0, len(attributes) - 1):
		for j in range(0, len(attributes) - 1):
 			similarity_matrix[i][j] = 1 - scipy.spatial.distance.cosine(map(int, attributes.iloc[i + 1]), map(int, attributes.iloc[j + 1]))
 	return similarity_matrix
#print similarity_matrix
similarity_matrix = compute_cosine_similarity()


################################################################################################################################
# Method for phase 1 of the algorithm.
def phase1(membership):
	stop = 0
	for i in range(len(attributes) - 1):
		final_gain = 0.0
		delta_QNewman = 0.0
		max_j = -1
		mem_i = membership[i]
		# Initial modularity before moving i to j's community.
		original_QNewman = g.modularity(membership)
		for j in range(len(attributes) - 1):
			# For nodes belonging to same community, we simply skip the phase and move on to next node.
			if j == i or membership[i] == membership[j]:
				continue

			# Moving i to j's community.
			membership[i] = membership[j]
			# Getting the new modularity after moving i to j's community.
			updated_QNewman = g.modularity(membership)
			# Getting the delta Q Newman.
			delta_QNewman = updated_QNewman - original_QNewman

			# Getting the delta Q attribute.
			delta_Qattr = 0.0
			index = 0
			for element in membership:
				if element == membership[j]:
					delta_Qattr += similarity_matrix[i][index]
				else:
					index += 1	
			# Finding the size of community to normalize delta_Qattr.
			size = 0
			for d in membership:
				if membership[j] == d:
					size += 1

			# Normalizing the delta_Qattr.
			delta_Qattr = delta_Qattr / size

			# Finding the composite gain.
			delQ = alpha * delta_QNewman + ((1 - alpha) * delta_Qattr)

			# Checking for the maximum gain.
			if delQ > final_gain and delta_QNewman > 0:
				final_gain = delQ
				max_j = j

			# Setting the membership of i back to its original membership for the next iteration of j.
			membership[i] = mem_i
		# Checking if the final gain exists or not.	
		if final_gain > 0 and max_j != -1:
			stop = 1
			membership[i] = membership[max_j]

	# Returning the membership set.
	if(stop == 0):
		return membership, 0
	else:
		return membership, 1	

					
################################################################################################################################

# Method for phase 2 of the algorithm.
def phase2(membership):
	stop = 0
	if(True):
		#print "Initial memlist", membership		
		# Applying phase 1 logic to move one community to other community and checking the composite gain.
		for i in xrange(max(membership) + 1):
			final_gain = 0.0
			delta_QNewman = 0.0
			max_j = -1
			mem_i = membership[i]
			# Initial modularity before moving i to j's community.
			original_QNewman = g.modularity(membership)
			for j in range(max(membership) + 1):
				# For nodes belonging to same community, we simply skip the phase and move on to next node.
				if j == i or membership[i] == membership[j]:
					continue
				# Moving i community to j communit and checking the composite gain.
				#print "after change"
				indices = [t for t, x in enumerate(membership) if x == i]
				for tind in indices:
					membership[tind] = j
				#print membership
				# Getting the new modularity after moving i to j's community.
				updated_QNewman = g.modularity(membership)
				# Getting the delta Q Newman.
				delta_QNewman = updated_QNewman - original_QNewman

				# Getting the delta Q attribute.
				indices2 = [t for t, x in enumerate(membership) if x == j]
				delta_Qattr = 0.0
				index = 0

				for n1 in indices:
					for n2 in indices2:
						delta_Qattr += similarity_matrix[n1][n2]
				delta_Qattr = delta_Qattr/(len(indices)*len(indices2))

				# Finding the composite gain.
				delQ = alpha * delta_QNewman + ((1 - alpha) * delta_Qattr)

				# Checking for the maximum gain.
				if delQ > final_gain and delta_QNewman > 0:
					final_gain = delQ
					max_j = j

				# Setting the membership of i back to its original membership for the next iteration of j.
				#print"after revert"
				for tind in indices:
				 	membership[tind] = i

			# Checking if the final gain exists or not.	
			if final_gain > 0 :
				stop = 1
				#print "Max of j ", max_j
				indices = [t for t, x in enumerate(new_mem) if x == i]
				for tind in indices:
					membership[tind] = max_j

		#print membership
		if(stop == 0):
			return membership, 0
		else:
			return membership, 1			
			
################################################################################################################################


for iterations in xrange(15):
	# SAC Phase 1.
	print "This is iteration: ", iterations
	membership , stop1 = phase1(membership)
	node_dict = {}
	new_mem, count = [], 0
	for i in membership:
		# If the node is already inserted in the node_dict, then simply append the value corresponding to that node from node_dict.
		if i in node_dict.keys():
			new_mem.append(node_dict[i])
		# If the node is not present in the dictionary, then add that node to the dict and update the count.
		else:	
			node_dict[i] = count
			new_mem.append(count)
			count += 1
	membership = new_mem
	print membership
	if(stop1 == 0):
		print "exiting"
		break

	# SAC Phase 2.
	membership, stop2  = phase2(membership)
	if(stop2 == 0):
		print "exiting"
		break
	print membership

g.contract_vertices(membership)
g.simplify(multiple = True)
print "Final communities."
communities = (list(Clustering(membership)))

orig_stdout = sys.stdout
file_output = open("communities.txt", "w+")
sys.stdout = file_output
for c in communities:
	print ",".join(str(x) for x in c)
sys.stdout = orig_stdout
file_output.close()