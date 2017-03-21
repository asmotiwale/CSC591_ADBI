import sys
import random
import pandas as pd
import numpy as np

##########################################################################################################################################################

# Method for the greedy algorithm.
def greedy(bidder_dataset, queries, budget_dict):
	# Creating a dictionary for the queries and the corresponding advertisers who have placed a bid on that query.
	queries_dict = {}
	for q in queries['q'].values:
		if q not in queries_dict.keys():
			# Sorting the advertisers based on the Bid Values.
			queries_dict[q] = bidder_dataset.loc[(bidder_dataset.Keyword == q)].sort_values(by = 'Bid Value', ascending = False).values		
	
	tot_revenue = 0
	for i in xrange(100):
		revenue = 0
		new_budget = budget_dict.copy()
		# Shuffling the queries for every iteration.
		queries_new = queries['q'].sample(frac = 1).values
		# Iterating over the queries
		for q in queries_new:
			# Iterating over the advertisers to check if the bid is valid.
			for b in queries_dict[q]:
				if new_budget[b[0]] - b[2] >= 0:
					# Updating the budget dictionary and the revenue for this iteration.
					new_budget[b[0]] -= b[2]
					revenue += b[2]
					break

		# Adding the revenues from every iteration.		
		tot_revenue += revenue
	return tot_revenue / 100

##########################################################################################################################################################

# Method for the balance algorithm.
def balance(bidder_dataset, queries, budget_dict):
	# Creating a dictionary for the queries and the corresponding advertisers who have placed a bid on that query.
	queries_dict = {}
	for q in queries['q'].values:
		if q not in queries_dict.keys():
			queries_dict[q] = bidder_dataset.loc[(bidder_dataset.Keyword == q)].values

	tot_revenue = 0
	for i in xrange(100):
		revenue = 0
		new_budget = budget_dict.copy()
		# Shuffling the queries for every iteration.
		queries_new = queries['q'].sample(frac = 1).values
		# Iterating over the queries
		for q in queries_new:
			m, m_bid, m_adv = 0, 0, 0
			# Iterating over the advertisers who have placed the bid on the incoming query.
			for b in queries_dict[q]:
				# Selecting the advertiser with the highest unspent budget for the incoming query.
				if m < new_budget[b[0]] and new_budget[b[0]] >= b[2]:
					m = new_budget[b[0]]
					m_bid = b[2]
					m_adv = b[0]

			# Updating the budget and revenue for the current iteration.
			new_budget[m_adv] -= m_bid
			revenue += m_bid
		
		# Adding the revenues from every iteration.			
		tot_revenue += revenue	
	return tot_revenue / 100

##########################################################################################################################################################

#Method for the MSVV algorithm.
def msvv(bidder_dataset, queries, budget_dict):
	# Creating a dictionary for the queries and the corresponding advertisers who have placed a bid on that query.
	queries_dict = {}
	for q in queries['q'].values:
		if q not in queries_dict.keys():
			queries_dict[q] = bidder_dataset.loc[(bidder_dataset.Keyword == q)].values

	tot_revenue = 0
	for i in xrange(100):	
		# Creating a dictionary to keep the records for the amount spent by the advertisers.
		spent_budget_dict = dict.fromkeys(budget_dict, 0)
		revenue = 0	
		# Shuffling the queries for every iteration.
		queries_new = queries['q'].sample(frac = 1).values		
		# Iterating over the queries
		for q in queries_new:
			m_bid, m_adv = 0, 0
			msvv = 0
			# Iterating over the advertisers who have placed the bid on the incoming query.
			for b in queries_dict[q]:
				curr_msvv = (b[2] * (1 - np.exp((spent_budget_dict[b[0]] / budget_dict[b[0]]) - 1)))
				# Selecting the advertiser with the highest MSVV function value.
				if (msvv < curr_msvv) and ((spent_budget_dict[b[0]] + b[2]) <= budget_dict[b[0]]):
					msvv = curr_msvv
					m_bid = b[2]
					m_adv = b[0]

			# Updating the spent budget and the revenue for the current iteration.
			spent_budget_dict[m_adv] += m_bid
			revenue += m_bid
		# Adding the revenues from every iteration.			
		tot_revenue += revenue
	return tot_revenue / 100

##########################################################################################################################################################

# Reading the datasets.
## Please change the path for the datasets below.
bidder_dataset = pd.read_csv('/Users/Warrior/Desktop/MS/Sem2/ADBI/PROJECTS/P5/07.Topic-7.Project-7.AdWordsPlacement.BipartiteGraphMatching/bidder_dataset.csv')
queries = pd.read_csv('/Users/Warrior/Desktop/MS/Sem2/ADBI/PROJECTS/P5/07.Topic-7.Project-7.AdWordsPlacement.BipartiteGraphMatching/queries.txt', header = None, names = ['q'])

# Creating a dictionary to map the advertisers with their budget.
budget_bidder =  bidder_dataset.loc[(bidder_dataset.Budget > 0), ['Advertiser', 'Budget']]
budget_dict = budget_bidder.set_index('Advertiser')['Budget'].to_dict()
# Sum of all the budgets.
tot_sum = sum(budget_dict.values())
# Reading the command line argument
algorithm_name = sys.argv[1]
# Setting a random seed
random.seed(0)
# Calling the appropriate algorithm based on the command line argument goven.
if algorithm_name == "greedy":
	revenue = greedy(bidder_dataset, queries, budget_dict)
	print "Revenue collected using the greedy method is: ", revenue
	print "Competitive ratio for the greedy method is: ", revenue / tot_sum

elif algorithm_name == "balance":
	revenue = balance(bidder_dataset, queries, budget_dict)
	print "Revenue collected using the balance method is: ", revenue
	print "Competitive ratio for the balance method is: ", revenue / tot_sum

elif algorithm_name == "msvv":
	revenue = msvv(bidder_dataset, queries, budget_dict)
	print "Revenue collected using the MSVV method is: ", revenue
	print "Competitive ratio for the MSVV method is: ", revenue / tot_sum

else:
	print "Please select the appropriate algorithms: greedy, balance or msvv."	