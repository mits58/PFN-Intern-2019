# ライブラリのインポート
import numpy as np


# 使用する関数の定義
def ReLU(x):
    return np.maximum(0, x)


# Graphクラス
class Graph:
    # コンストラクタ : 隣接行列形式のデータを引数として取る
    def __init__(self, _A):
        self.A = _A  # 隣接行列
        self.V = self.A.shape[0]  # 頂点数
        self.T = 2  # 集約回数

    # 引数で与えられたパラメータWのもと集約を行うメソッド
    def Aggregate(self, W):
        D = W.shape[0]  # パラメータの次元
        F = np.zeros((D, self.V))   # 各頂点の特徴ベクトルを横に並べたD×Vのゼロ行列を定義
        F[0] = 1    # 各頂点の特徴ベクトルの最初の要素を1に初期化

        for i in range(self.T):
            F = ReLU(W.dot(F.dot(self.A.T)))

        # READOUTを行いそのベクトルを返す
        return F.dot(np.ones((self.V, 1)))


# main function
if __name__ == "__main__":
    # テスト入力のグラフ(問題文に記載のグラフ)
    A = np.array([[0, 1, 0, 0],
                  [1, 0, 1, 1],
                  [0, 1, 0, 1],
                  [0, 1, 1, 0]])
    G = Graph(A)
    W1 = np.ones((8, 8))    # 8次元の全ての要素が1の正方行列
    W2 = np.triu(np.ones((8, 8)))   # 上記の行列を上三角行列に変換した行列
    Vec1 = G.Aggregate(W1)  # W1に対応する特徴ベクトル
    Vec2 = G.Aggregate(W2)  # W2に対応する特徴ベクトル
    F0 = np.zeros((8, 4))
    F0[0] = 1
    print("Test 1")
    print(Vec1)
    print("Anticipated Output")
    print(((W1.dot(W1)).dot(F0)).dot((A.T).dot(A.T)).dot(np.ones((4, 1))))
    print("------------\n Test 2")
    print(Vec2)
    print("Anticipated Output")
    print(((W2.dot(W2)).dot(F0)).dot((A.T).dot(A.T)).dot(np.ones((4, 1))))
