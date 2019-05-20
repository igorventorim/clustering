import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
import random
from sklearn.metrics.cluster import adjusted_rand_score
from time import time

class Compare():

	def __init__(self,X,y,eps,minpts,delta=1):
		self.X = X
		self.y = y
		self.eps = eps
		self.minpts = minpts
		self.threshold = delta*eps

		self.dbscan_database = DBSCAN(eps=self.eps,min_samples=self.minpts)

		self.dbscan_sampling = DBSCAN(eps=self.eps,min_samples=self.minpts)
		self.birch = Birch(branching_factor=self.minpts, n_clusters=None,threshold=self.threshold)

		self.labels_birch=None
		self.labels_dbscan=None

		self.time_dbscan_database=None
		self.time_birch_dbscan=None

		self.size_sampling=None

	def run(self):
		print("\tDBSCAN - ", end='',flush=True)		
		self.run_dbscan_database()
		print("[OK]")
		
		print("\tBIRCH+DBSCAN - ", end='',flush=True)
		self.run_birch_dbscan()
		print("[OK]")

		print("Time DBSCAN: ", self.time_dbscan_database)
		print("Time DBSCAN + Birch:", self.time_birch_dbscan)

		print("\nCLUSTER DBSCAN\n",self.labels_dbscan)
		print("\nCLUSTER BIRCH+DBSCAN:\n", self.labels_birch)

		print("\nSIZE SAMPLING: ",self.size_sampling)
		print("\nARI:",adjusted_rand_score(self.labels_birch,self.labels_dbscan)*100,"%")

	def run_dbscan_database(self):
		t = time()
		self.labels_dbscan=self.dbscan_database.fit(self.X).labels_
		self.time_dbscan_database = time() - t

	def run_birch_dbscan(self):
		t = time()
		birch = self.birch.fit(self.X)
		self.labels_birch = self.birch.fit(self.X).labels_

		#ORGANIZE CLUSTERS BIRCH
		clusters = {}
		for ind,cluster in enumerate(self.birch.labels_):
			if cluster in clusters:
				clusters[cluster].append(ind)
			else:
				clusters[cluster] = [ind]

		print(str(len(clusters)) +"="+ str(len(birch.subcluster_centers_)))

		sampling = np.array(self.build_sampling(clusters))
		clustering = self.dbscan_sampling.fit(sampling)
		self.expansion(clusters,clustering)
		self.size_sampling=len(sampling)
		self.time_birch_dbscan = time() - t

	def build_sampling(self,clusters):
		positions=[]
		for key, value in clusters.items():
			if(len(value) > (self.minpts+1)):
				positions += random.sample(value,k=self.minpts+1)
			else:
				positions += value

		sampling = []
		for i in positions:
			sampling.append(self.X[i])

		return sampling

	def expansion(self,groups,clustering):
		clusters = {}
		result=clustering.labels_
		for key, value in groups.items():
			if(len(value) > (self.minpts+1)):
				avaliation = result[:self.minpts+1]
				result = result[self.minpts+1:]
			else:
				qtd = len(value)
				avaliation = result[:qtd]
				result = result[qtd:]

			element = np.bincount(np.array(np.where(avaliation==-1,999999,avaliation))).argmax()
			if element == 999999:
				element = -1
			clusters[key] = element

		for i in range(len(self.labels_birch)):
			self.labels_birch[i] = clusters[self.labels_birch[i]]

		# return clusters