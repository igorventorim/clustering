import numpy as np
import sklearn.datasets as datasets
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from sklearn.cluster import Birch
import random
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.datasets.samples_generator import make_blobs
from time import time
import matplotlib.pyplot as plt

MIN_SAMPLES=5
EPS=2.65
THRESHOLD=0.9*EPS
EXCESS=1

print("N SAMPLES:",30000," N FEATURES:",10)

X, y = make_blobs(n_samples=30000, centers=3, n_features=10, random_state=500)
plt.plot(X[:,0], X[:,1], "ko")
plt.show()

#TESTING IRIS DATASET
iris = datasets.load_iris()
featname = iris['feature_names']
# X = iris.data
# y = iris.target
# classname = iris.target_names

dbscan = DBSCAN(eps=EPS,min_samples=MIN_SAMPLES)
t = time()
clustering = dbscan.fit(X)
time_ = time() - t
print("Time DBSCAN: ", (time() - t))
# print(clustering)
# print(clustering.labels_)
result_pure_dbscan = clustering.labels_
print("\nPURE DBSCAN\nPARAMETERS: EPS =",EPS,", MIN PTS=",MIN_SAMPLES,"\nCLUSTERS:\n",result_pure_dbscan)

brc = Birch(branching_factor=MIN_SAMPLES, n_clusters=None, threshold=THRESHOLD)

t = time()
brc.fit(X)
# time_dbscan_birch = time() - t
labels_birch = brc.labels_

print("\n\nBIRCH\nPARAMETERS: branching factor = ",MIN_SAMPLES,",n clusters = None, threshold=",THRESHOLD,"\nNúmero de clusters encontrados:",np.unique(brc.labels_).size)
print("CLUSTERS:\n",brc.labels_)

#SAVE CLUSTER BIRCH
classes = {}
for ind,classe in enumerate(brc.labels_):
	if classe in classes:
		classes[classe].append(ind)
	else:
		classes[classe] = [ind]
# print(classes)

#BUILDING SAMPLING
positions = []
for key, value in classes.items():
	if(len(value) > (MIN_SAMPLES+EXCESS)):
		positions += random.sample(value,k=MIN_SAMPLES+EXCESS)
	else:
		positions += value
# print("\nPosições - ",MIN_SAMPLES+EXCESS,"elementos por cluster retirados para amostra (Caso o cluster seja maior que",MIN_SAMPLES+EXCESS,"):\n",positions)


sampling = []
result_pure_dbscan_sampling = []
for i in positions:
	sampling.append(X[i])
	result_pure_dbscan_sampling.append(result_pure_dbscan[i])
sampling = np.array(sampling)
print("\nTamanho amostragem:",len(sampling))
print("Amostragem:\n", sampling)

# t = time()
clustering = dbscan.fit(sampling)
# time_dbscan_birch += time() - t
print("\n\nCLUSTERIZAÇÃO DA AMOSTRA NO DBSCAN COM OS MESMOS PARAMETROS INICIAIS:\n",clustering.labels_)

print("\n\nRESULTADO DO DBSCAN NA BASE COMPLETA ESPERADO NA AMOSTRA:\n",np.array(result_pure_dbscan_sampling))

print("\nARI:",adjusted_rand_score(clustering.labels_,result_pure_dbscan_sampling)*100,"%")

cluster_birch_label_dbscan = {} 
result = clustering.labels_
for key, value in classes.items():
	if(len(value) > (MIN_SAMPLES+EXCESS)):
		avaliation = result[:MIN_SAMPLES+EXCESS]
		result = result[MIN_SAMPLES+EXCESS:]
	else:
		qtd = len(value)
		avaliation = result[:qtd]
		result = result[qtd:]

	element = np.bincount(np.array(np.where(avaliation==-1,999999,avaliation))).argmax()
	if element == 999999:
		element = -1
	cluster_birch_label_dbscan[key] = element

# print(cluster_birch_label_dbscan)

for i in range(len(labels_birch)):
	labels_birch[i] = cluster_birch_label_dbscan[labels_birch[i]]

time_ = time() - t
print("\nCLUSTER DBSCAN\n",result_pure_dbscan)
print("\nCLUSTER BIRCH+DBSCAN:\n", labels_birch)

print("\nARI:",adjusted_rand_score(labels_birch,result_pure_dbscan)*100,"%")
print("TIME BIRCH+DBSCAN+OVERHEAD: ", (time() - t))

# print("\nTIME DBSCAN+BIRCH:",time_dbscan_birch)

# print(brc.predict(X))
# # print(len())
# sampling = brc.subcluster_centers_
# print(sampling)
# dbscan = DBSCAN(eps=0.4,min_samples=1)
# clustering = dbscan.fit(sampling)
# print(clustering.labels_)

# import ipdb; ipdb.set_trace()


# lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# X = lfw_people.data
# y = lfw_people.target

# fetch_covtype = datasets.fetch_covtype()
# X = fetch_covtype.data
# y = fetch_covtype.target


#TESTING FETCH OLIVETTI FACES DATASET 
# fetch_olivetti_faces = datasets.fetch_olivetti_faces()
# X = fetch_olivetti_faces.data
# y = fetch_olivetti_faces.target
# dbscan = DBSCAN(eps=7.7,min_samples=2)
# clustering = dbscan.fit(X)
# print(len(clustering.labels_))
# print(clustering.labels_)
# # print(y)
# brc = Birch(branching_factor=2, n_clusters=None, threshold=2)
# brc.fit(X)
# # print(brc.predict(X))
# sampling = brc.subcluster_centers_
# print(sampling)
# clustering = dbscan.fit(sampling)
# print(len(clustering.labels_))
# print(clustering.labels_)
