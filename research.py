from compare import Compare
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import sklearn.datasets as datasets

MIN_SAMPLES=3
EPS=2.65
DELTA=0.9

def main():
	
	print('Executing main() ....')
	print("N SAMPLES:",30000," N FEATURES:",10)
	X, y = make_blobs(n_samples=15000, centers=3, n_features=10, random_state=500)

    # plt.plot(X[:,0], X[:,1], "ko")
	# plt.show()

	# iris = datasets.load_iris()
	# X = iris.data
	# y = iris.target
	# MIN_SAMPLES=4
	# EPS=0.4
	# DELTA=0.7

	# fetch_covtype = datasets.fetch_covtype()
	# X = fetch_covtype.data
	# y = fetch_covtype.target
	# plt.plot(X[:,0], X[:,1], "ko")
	# plt.show()
	# EPS=250

	# fetch_olivetti_faces = datasets.fetch_olivetti_faces()
	# X = fetch_olivetti_faces.data
	# y = fetch_olivetti_faces.target
	# plt.plot(X[:,0], X[:,1], "ko")
	# plt.show()
	# EPS=7.3
	# DELTA=0.6

	# lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
	# X = lfw_people.data
	# y = lfw_people.target
	# plt.plot(X[:,0], X[:,1], "ko")
	# plt.show()
	# EPS=1400
	# DELTA=0.7

	compare = Compare(X,y,EPS,MIN_SAMPLES,DELTA)
	compare.run()

if __name__ == "__main__":
	main()


