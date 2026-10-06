"""
游戏内聚类分析与无监督学习系统 - 核心模块
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random
import time


@dataclass
class Config:
    pass


class Clustering:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 距离与相似度
    # ------------------------------------------------------------------
    @staticmethod
    def euclidean_distance(a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def manhattan_distance(a: List[float], b: List[float]) -> float:
        return sum(abs(x - y) for x, y in zip(a, b))

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def distance(self, a: List[float], b: List[float], metric: str = "euclidean") -> float:
        if metric == "euclidean":
            return self.euclidean_distance(a, b)
        if metric == "manhattan":
            return self.manhattan_distance(a, b)
        if metric == "cosine":
            return 1.0 - self.cosine_similarity(a, b)
        raise ValueError(f"未知的距离度量: {metric}")

    def distance_matrix(self, data: List[List[float]], metric: str = "euclidean") -> List[List[float]]:
        n = len(data)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(data[i], data[j], metric)
                matrix[i][j] = d
                matrix[j][i] = d
        return matrix

    def similarity_matrix(self, data: List[List[float]]) -> List[List[float]]:
        n = len(data)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                s = self.cosine_similarity(data[i], data[j])
                matrix[i][j] = s
                matrix[j][i] = s
        return matrix

    # ------------------------------------------------------------------
    # 数据聚类入口
    # ------------------------------------------------------------------
    def cluster_data(self, data: List[List[float]], method: str = "kmeans", **kwargs) -> Dict:
        if not data:
            raise ValueError("数据不能为空")
        if method == "kmeans":
            result = self.kmeans(data, **kwargs)
        elif method == "hierarchical":
            result = self.hierarchical_clustering(data, **kwargs)
        elif method == "dbscan":
            result = self.dbscan(data, **kwargs)
        else:
            raise ValueError(f"未知的聚类方法: {method}")
        labels = result["labels"]
        dist_matrix = self.distance_matrix(data)
        sim_matrix = self.similarity_matrix(data)
        validation = self.cluster_validation(data, labels)
        summary = {
            "method": method,
            "labels": labels,
            "n_clusters": len({l for l in labels if l != -1}),
            "distance_matrix": dist_matrix,
            "similarity_matrix": sim_matrix,
            "validation": validation,
            "detail": result,
        }
        self._history.append({"method": method, "n_samples": len(data),
                              "n_clusters": summary["n_clusters"], "time": time.time()})
        return summary

    # ------------------------------------------------------------------
    # K-Means
    # ------------------------------------------------------------------
    def kmeans(self, data: List[List[float]], k: int = 2,
               max_iters: int = 100, tol: float = 1e-6, seed: int = 42) -> Dict:
        if k < 1 or k > len(data):
            raise ValueError("k 必须在 1 与样本数之间")
        centroids = self._init_centroids_kmeanspp(data, k, seed)
        labels = [0] * len(data)
        iterations = 0
        for iterations in range(1, max_iters + 1):
            labels = [self._nearest_centroid(p, centroids) for p in data]
            new_centroids = []
            for c in range(k):
                members = [data[i] for i in range(len(data)) if labels[i] == c]
                if members:
                    new_centroids.append([sum(p[d] for p in members) / len(members)
                                          for d in range(len(data[0]))])
                else:
                    new_centroids.append(list(centroids[c]))
            shift = max(self.euclidean_distance(new_centroids[c], centroids[c])
                        for c in range(k))
            centroids = new_centroids
            if shift < tol:
                break
        inertia = sum(self.euclidean_distance(data[i], centroids[labels[i]]) ** 2
                      for i in range(len(data)))
        return {"labels": labels, "centroids": centroids,
                "iterations": iterations, "inertia": inertia, "k": k}

    def _init_centroids_kmeanspp(self, data: List[List[float]], k: int, seed: int) -> List[List[float]]:
        rng = random.Random(seed)
        centroids = [list(data[rng.randrange(len(data))])]
        while len(centroids) < k:
            dist_sq = [min(sum((p[d] - c[d]) ** 2 for d in range(len(p)))
                           for c in centroids) for p in data]
            total = sum(dist_sq)
            if total == 0.0:
                candidates = [i for i, p in enumerate(data) if list(p) not in centroids]
                centroids.append(list(data[candidates[rng.randrange(len(candidates))]]))
                continue
            threshold = rng.random() * total
            cumulative = 0.0
            chosen = len(data) - 1
            for i, d2 in enumerate(dist_sq):
                cumulative += d2
                if cumulative >= threshold:
                    chosen = i
                    break
            centroids.append(list(data[chosen]))
        return centroids

    def _nearest_centroid(self, point: List[float], centroids: List[List[float]]) -> int:
        best, best_dist = 0, float("inf")
        for i, c in enumerate(centroids):
            d = self.euclidean_distance(point, c)
            if d < best_dist:
                best, best_dist = i, d
        return best

    # ------------------------------------------------------------------
    # 层次聚类（凝聚 + 分裂）
    # ------------------------------------------------------------------
    def hierarchical_clustering(self, data: List[List[float]], n_clusters: int = 2,
                                mode: str = "agglomerative", linkage: str = "average") -> Dict:
        if n_clusters < 1 or n_clusters > len(data):
            raise ValueError("n_clusters 必须在 1 与样本数之间")
        if mode == "agglomerative":
            return self._agglomerative(data, n_clusters, linkage)
        if mode == "divisive":
            return self._divisive(data, n_clusters)
        raise ValueError(f"未知的层次聚类模式: {mode}")

    def _cluster_distance(self, cluster_a: List[int], cluster_b: List[int],
                          dist_matrix: List[List[float]], linkage: str) -> float:
        pairs = [dist_matrix[i][j] for i in cluster_a for j in cluster_b]
        if linkage == "single":
            return min(pairs)
        if linkage == "complete":
            return max(pairs)
        if linkage == "average":
            return sum(pairs) / len(pairs)
        raise ValueError(f"未知的连接方式: {linkage}")

    def _agglomerative(self, data: List[List[float]], n_clusters: int, linkage: str) -> Dict:
        dist_matrix = self.distance_matrix(data)
        clusters = [[i] for i in range(len(data))]
        merge_history = []
        while len(clusters) > n_clusters:
            best_pair, best_dist = None, float("inf")
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    d = self._cluster_distance(clusters[a], clusters[b], dist_matrix, linkage)
                    if d < best_dist:
                        best_pair, best_dist = (a, b), d
            a, b = best_pair
            merged = clusters[a] + clusters[b]
            merge_history.append({"merged": (sorted(clusters[a]), sorted(clusters[b])),
                                  "distance": best_dist})
            clusters = [c for idx, c in enumerate(clusters) if idx not in (a, b)]
            clusters.append(merged)
        labels = [0] * len(data)
        for cid, members in enumerate(clusters):
            for i in members:
                labels[i] = cid
        return {"labels": labels, "n_clusters": n_clusters, "mode": "agglomerative",
                "linkage": linkage, "merge_history": merge_history}

    def _divisive(self, data: List[List[float]], n_clusters: int) -> Dict:
        dist_matrix = self.distance_matrix(data)
        clusters = [list(range(len(data)))]
        split_history = []
        while len(clusters) < n_clusters:
            def diameter(members):
                if len(members) < 2:
                    return -1.0
                return max(dist_matrix[i][j] for i in members for j in members)
            target = max(range(len(clusters)), key=lambda c: diameter(clusters[c]))
            members = clusters[target]
            if len(members) < 2:
                break
            subset = [data[i] for i in members]
            sub = self.kmeans(subset, k=2)
            group_a = [members[i] for i in range(len(members)) if sub["labels"][i] == 0]
            group_b = [members[i] for i in range(len(members)) if sub["labels"][i] == 1]
            if not group_a or not group_b:
                group_a, group_b = members[:-1], members[-1:]
            split_history.append({"split": sorted(members),
                                  "into": (sorted(group_a), sorted(group_b))})
            clusters = [c for idx, c in enumerate(clusters) if idx != target]
            clusters.extend([group_a, group_b])
        labels = [0] * len(data)
        for cid, members in enumerate(clusters):
            for i in members:
                labels[i] = cid
        return {"labels": labels, "n_clusters": len(clusters), "mode": "divisive",
                "split_history": split_history}

    # ------------------------------------------------------------------
    # DBSCAN
    # ------------------------------------------------------------------
    def dbscan(self, data: List[List[float]], eps: float = 0.5, min_samples: int = 5) -> Dict:
        if eps <= 0:
            raise ValueError("eps 必须为正数")
        if min_samples < 1:
            raise ValueError("min_samples 必须 >= 1")
        n = len(data)
        dist_matrix = self.distance_matrix(data)
        neighbors = [[j for j in range(n) if dist_matrix[i][j] <= eps] for i in range(n)]
        core_points = [i for i in range(n) if len(neighbors[i]) >= min_samples]
        labels = [-1] * n
        cluster_id = 0
        for i in range(n):
            if i not in core_points or labels[i] != -1:
                continue
            labels[i] = cluster_id
            queue = list(neighbors[i])
            while queue:
                j = queue.pop()
                if labels[j] == -1:
                    labels[j] = cluster_id
                    if j in core_points:
                        for m in neighbors[j]:
                            if labels[m] == -1:
                                queue.append(m)
            cluster_id += 1
        density_reachable = {i: sorted(j for j in range(n)
                                       if labels[j] != -1 and labels[j] == labels[i])
                             for i in core_points}
        return {"labels": labels, "n_clusters": cluster_id, "eps": eps,
                "min_samples": min_samples, "core_points": core_points,
                "noise_points": [i for i in range(n) if labels[i] == -1],
                "density_reachable": density_reachable}

    # ------------------------------------------------------------------
    # 聚类验证
    # ------------------------------------------------------------------
    def cluster_validation(self, data: List[List[float]], labels: List[int]) -> Dict:
        valid = [(i, l) for i, l in enumerate(labels) if l != -1]
        cluster_ids = sorted({l for _, l in valid})
        if len(cluster_ids) < 2:
            return {"silhouette_score": 0.0, "davies_bouldin_index": 0.0,
                    "n_clusters": len(cluster_ids)}
        dist_matrix = self.distance_matrix(data)
        members = {c: [i for i, l in valid if l == c] for c in cluster_ids}
        silhouette = self._silhouette_score(dist_matrix, members)
        db_index = self._davies_bouldin_index(data, dist_matrix, members, cluster_ids)
        return {"silhouette_score": silhouette, "davies_bouldin_index": db_index,
                "n_clusters": len(cluster_ids)}

    def _silhouette_score(self, dist_matrix: List[List[float]],
                          members: Dict[int, List[int]]) -> float:
        scores = []
        cluster_ids = list(members.keys())
        for cid in cluster_ids:
            for i in members[cid]:
                own = members[cid]
                a = (sum(dist_matrix[i][j] for j in own) / (len(own) - 1)
                     if len(own) > 1 else 0.0)
                b = min(sum(dist_matrix[i][j] for j in members[other]) / len(members[other])
                        for other in cluster_ids if other != cid)
                denom = max(a, b)
                scores.append((b - a) / denom if denom > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def _davies_bouldin_index(self, data: List[List[float]], dist_matrix: List[List[float]],
                              members: Dict[int, List[int]], cluster_ids: List[int]) -> float:
        centroids = {}
        scatter = {}
        for cid in cluster_ids:
            pts = members[cid]
            centroid = [sum(data[i][d] for i in pts) / len(pts) for d in range(len(data[0]))]
            centroids[cid] = centroid
            scatter[cid] = (sum(self.euclidean_distance(data[i], centroid) for i in pts)
                            / len(pts))
        total = 0.0
        for cid in cluster_ids:
            worst = max(
                (scatter[cid] + scatter[other])
                / self.euclidean_distance(centroids[cid], centroids[other])
                if self.euclidean_distance(centroids[cid], centroids[other]) > 0 else float("inf")
                for other in cluster_ids if other != cid)
            total += worst
        return total / len(cluster_ids)
