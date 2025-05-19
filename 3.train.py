from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

#initilizing of embedding & recognizer
embeddingFile = "output/embeddings.pickle"
#New & Empty at initial
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

print("Loading face embeddings...")
data = pickle.loads(open(embeddingFile, "rb").read())


print("Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])

# Check the number of classes
import numpy as np
unique_classes = np.unique(labels)

if len(unique_classes) < 2:
    print(f"Only one class found ({unique_classes[0]}). Training OneClassSVM model instead.")
    from sklearn.svm import OneClassSVM
    recognizer = OneClassSVM(gamma='auto')
    recognizer.fit(data["embeddings"])
else:
    from sklearn.svm import SVC
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

print("Training model completed.")

f = open(recognizerFile, "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(labelEncFile, "wb")
f.write(pickle.dumps(labelEnc))
f.close()
