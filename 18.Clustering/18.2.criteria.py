# =============================================================================
# 模拟数据，做聚类的评价
# =============================================================================

from sklearn import metrics

if __name__ == "__main__":
    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 1, 2, 2]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    print('同一性(Homogeneity)：', h)
    print('完整性(Completeness)：', c)
    v2 = 2 * c * h / (c + h)
    v = metrics.v_measure_score(y, y_hat)
    print('V-Measure：', v2, v)

    print()
    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 3, 3, 3]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    v = metrics.v_measure_score(y, y_hat)
    print('同一性(Homogeneity)：', h)
    print('完整性(Completeness)：', c)
    print('V-Measure：', v)

    # 允许不同值
    print()
    y = [0, 0, 0, 1, 1, 1]
    y_hat = [1, 1, 1, 0, 0, 0]
    h = metrics.homogeneity_score(y, y_hat)
    c = metrics.completeness_score(y, y_hat)
    v = metrics.v_measure_score(y, y_hat)
    print('同一性(Homogeneity)：', h)
    print('完整性(Completeness)：', c)
    print('V-Measure：', v)
    
    print()
    y = [0, 0, 1, 1]
    y_hat = [0, 1, 0, 1]
    ari = metrics.adjusted_rand_score(y, y_hat)
    print(ari)

    y = [0, 0, 0, 1, 1, 1]
    y_hat = [0, 0, 1, 1, 2, 2]
    ari = metrics.adjusted_rand_score(y, y_hat)
    print(ari)