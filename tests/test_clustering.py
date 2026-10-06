"""
游戏内聚类分析与无监督学习系统 - 验收测试
运行方式：python -m pytest tests/test_clustering.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.clustering import Clustering


def make_blobs():
    blob_a = [[1.0, 1.0], [1.1, 1.2], [1.2, 1.1], [1.1, 1.0]]
    blob_b = [[10.0, 10.0], [10.1, 10.2], [10.2, 10.1], [10.1, 10.0]]
    return blob_a + blob_b


class TestClustering:
    def setup_method(self):
        self.system = Clustering()

    def test_case_01(self):
        d = self.system.euclidean_distance([0.0, 0.0], [3.0, 4.0])
        assert d == pytest.approx(5.0)
        m = self.system.manhattan_distance([0.0, 0.0], [3.0, 4.0])
        assert m == pytest.approx(7.0)

    def test_case_02(self):
        s = self.system.cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert s == pytest.approx(0.0)
        s2 = self.system.cosine_similarity([1.0, 2.0], [2.0, 4.0])
        assert s2 == pytest.approx(1.0)

    def test_case_03(self):
        data = make_blobs()
        matrix = self.system.distance_matrix(data)
        assert len(matrix) == len(data)
        for i in range(len(data)):
            assert matrix[i][i] == pytest.approx(0.0)
            for j in range(len(data)):
                assert matrix[i][j] == pytest.approx(matrix[j][i])
        sim = self.system.similarity_matrix(data)
        assert len(sim) == len(data)
        for i in range(len(data)):
            assert sim[i][i] == pytest.approx(1.0)

    def test_case_04(self):
        data = make_blobs()
        result = self.system.cluster_data(data, method="kmeans", k=2)
        assert len(result["labels"]) == len(data)
        assert result["n_clusters"] == 2
        assert "distance_matrix" in result
        assert "similarity_matrix" in result
        assert "validation" in result

    def test_case_05(self):
        data = make_blobs()
        result = self.system.kmeans(data, k=2)
        centroids = result["centroids"]
        assert len(centroids) == 2
        xs = [p[0] for p in data]
        ys = [p[1] for p in data]
        for c in centroids:
            assert min(xs) <= c[0] <= max(xs)
            assert min(ys) <= c[1] <= max(ys)
        assert self.system.euclidean_distance(centroids[0], centroids[1]) > 5.0

    def test_case_06(self):
        data = make_blobs()
        result = self.system.kmeans(data, k=2)
        assert result["iterations"] >= 1
        assert result["iterations"] <= 100
        assert result["inertia"] < 1.0

    def test_case_07(self):
        data = make_blobs()
        result = self.system.kmeans(data, k=2)
        labels = result["labels"]
        assert len(set(labels[:4])) == 1
        assert len(set(labels[4:])) == 1
        assert labels[0] != labels[4]

    def test_case_08(self):
        data = make_blobs()
        result = self.system.hierarchical_clustering(data, n_clusters=2, mode="agglomerative")
        labels = result["labels"]
        assert len(set(labels)) == 2
        assert len(set(labels[:4])) == 1
        assert len(set(labels[4:])) == 1
        assert labels[0] != labels[4]

    def test_case_09(self):
        data = make_blobs()
        result = self.system.hierarchical_clustering(data, n_clusters=2, mode="agglomerative")
        history = result["merge_history"]
        assert len(history) == len(data) - 2
        distances = [h["distance"] for h in history]
        assert distances == sorted(distances)

    def test_case_10(self):
        data = make_blobs()
        result = self.system.hierarchical_clustering(data, n_clusters=2, mode="divisive")
        labels = result["labels"]
        assert len(set(labels)) == 2
        assert len(set(labels[:4])) == 1
        assert len(set(labels[4:])) == 1
        assert labels[0] != labels[4]
        assert len(result["split_history"]) == 1

    def test_case_11(self):
        data = make_blobs()
        result = self.system.dbscan(data, eps=0.5, min_samples=3)
        assert len(result["core_points"]) == len(data)
        assert result["n_clusters"] == 2
        assert result["noise_points"] == []

    def test_case_12(self):
        data = make_blobs() + [[50.0, 50.0]]
        result = self.system.dbscan(data, eps=0.5, min_samples=3)
        assert result["labels"][-1] == -1
        assert result["noise_points"] == [len(data) - 1]
        assert result["n_clusters"] == 2
        for core, reachable in result["density_reachable"].items():
            assert core in reachable
            assert len(reachable) == 4

    def test_case_13(self):
        data = make_blobs()
        result = self.system.kmeans(data, k=2)
        validation = self.system.cluster_validation(data, result["labels"])
        silhouette = validation["silhouette_score"]
        assert -1.0 <= silhouette <= 1.0
        assert silhouette > 0.9

    def test_case_14(self):
        data = make_blobs()
        result = self.system.kmeans(data, k=2)
        validation = self.system.cluster_validation(data, result["labels"])
        db_index = validation["davies_bouldin_index"]
        assert db_index >= 0.0
        assert db_index < 0.1
        assert validation["n_clusters"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
