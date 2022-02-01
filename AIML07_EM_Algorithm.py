#
# from sklearn.cluster import KMeans
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# data=pd.read_csv("AIML07_EM_Algorithm.csv")
# f1=np.array(data.Distance_Feature)
# f2=np.array(data.Speeding_Feature)
#
# X=np.matrix(list(zip(f1,f2)))
#
# # plt.plot()
# plt.xlim([0, 90])
# plt.ylim([0, 90])
#
#
# plt.title('Dataset')
# plt.ylabel('speeding_feature')
# plt.xlabel('Distance_Feature')
#
# plt.scatter(f1,f2)
# plt.show()
# # plt.plot()
#
# colors = ['b', 'g', 'r']
# markers = ['o', 'v', 's']
#
# kmeans_model = KMeans(n_clusters=3).fit(X)
# plt.plot()
#
#
# for i, l in enumerate(kmeans_model.labels_):
#     plt.plot(f1[i], f2[i], color=colors[l], marker=markers[l],ls='None')
#     plt.xlim([0, 100])
#     plt.ylim([0, 50])
# plt.show()



from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("AIML07_EM_Algorithm.csv")
f1=np.array(data.Distance_Feature)
f2=np.array(data.Speeding_Feature)
X=np.matrix(list(zip(f1,f2)))
plt.title('Dataset')
plt.ylabel('speeding_feature')
plt.xlabel('Distance_Feature')
plt.scatter(f1,f2)
plt.show()
colors = ['r', 'g', 'b']
markers = ['o', 'v', 's']
kmeans_model = KMeans(3).fit(X)
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(f1[i], f2[i], color=colors[l], marker=markers[l])
plt.show()
