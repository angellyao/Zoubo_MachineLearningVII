# =============================================================================
# AP聚类
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import euclidean_distances

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    # 1.模拟数据
    N = 400
    centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)
    m = euclidean_distances(data, squared=True)
    preference = -np.median(m)
    print('Preference：', preference)
    # 2.查看聚类结果
    plt.figure(figsize=(12, 9), facecolor='w')
    for i, mul in enumerate(np.linspace(1, 4, 9)):
        print(mul)
        p = mul * preference
        model = AffinityPropagation(affinity='euclidean', preference=p)
        af = model.fit(data)
        center_indices = af.cluster_centers_indices_
        n_clusters = len(center_indices)
        print(('p = %.1f' % mul), p, '聚类簇的个数为：', n_clusters)
        y_hat = af.labels_
        
        # 3.画图
        plt.subplot(3, 3, i+1)
        plt.title('Preference：%.2f，簇个数：%d' % (p, n_clusters))
        clrs = []
        for c in np.linspace(16711680, 255, n_clusters, dtype=int):
            clrs.append('#%06x' % c)
        # clrs = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
        for k, clr in enumerate(clrs):
            cur = (y_hat == k)
            plt.scatter(data[cur, 0], data[cur, 1], s=15, c=clr, edgecolors='none')
            center = data[center_indices[k]]
            for x in data[cur]:
                plt.plot([x[0], center[0]], [x[1], center[1]], color=clr, lw=0.5, zorder=1)
        plt.scatter(data[center_indices, 0], data[center_indices, 1], s=80, c=clrs, marker='*', edgecolors='k', zorder=2)
        plt.grid(b=True, ls=':')
    plt.tight_layout()
    plt.suptitle('AP聚类', fontsize=20)
    plt.subplots_adjust(top=0.92)
    plt.show()