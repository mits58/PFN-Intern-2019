# ライブラリのインポート
import random
import glob
import numpy as np
import matplotlib.pyplot as plt


# 乱数シードを固定
random.seed(1124)


# --- 使用する関数の定義 --- #
def ReLU(x):
    # ReLU関数
    return np.maximum(0, x)


def Sigmoid(s):
    # Sigmoid関数
    return 1.0 / (1.0 + np.exp(-s))


def Loss_Function(s, y):
    # binary cross-entropy 損失関数(loss)
    if s >= 35:
        return y * np.log(1 + np.exp(-s)) + (1 - y) * s
    elif s <= -35:
        return -s * y + (1 - y) * np.log(1 + np.exp(s))
    else:
        return y * np.log(1 + np.exp(-s)) + (1 - y) * np.log(1 + np.exp(s))


# --- Graphクラス(課題1と同様) --- #
class Graph:
    def __init__(self, _A):
        self.A = _A
        self.V = self.A.shape[0]
        self.T = 2

    def Aggregate(self, W):
        F = np.zeros((W.shape[0], self.V))
        F[0] = 1
        for i in range(self.T):
            F = ReLU(W.dot(F.dot(self.A.T)))

        return F.dot(np.ones((self.V, 1)))


# --- 分類器クラス(課題2・3とほぼ同じため、説明を省略している部分があります) --- #
class Classifier:
    def __init__(self):
        # ハイパパラメータ
        self.D = 8
        self.alpha = 0.0032
        self.epsilon = 0.001
        self.B = 32

        self.W = np.random.normal(0.0, 0.4, (self.D, self.D))
        self.A = np.random.normal(0.0, 0.4, (self.D, 1))
        self.b = 0.0

        # Adamで用いる更新ステップ数
        self.t = 0

        # Adamで用いる1次モーメント、2次モーメントを格納しておく変数
        self.mW = np.zeros((self.D, self.D))
        self.vW = np.zeros((self.D, self.D))
        self.mA = np.zeros((self.D, 1))
        self.vA = np.zeros((self.D, 1))
        self.mb = 0
        self.vb = 0

    def Calculation_Loss(self, G, y):
        h_G = G.Aggregate(self.W)
        s = self.A.T.dot(h_G) + self.b
        return Loss_Function(s, y)

    def Calculation_Loss_and_Accuracy(self, G, y):
        loss = []
        accuracy = []
        for i in range(len(G)):
            h_G = G[i].Aggregate(self.W)
            s = self.A.T.dot(h_G) + self.b
            predict = int(Sigmoid(s) > 0.5)
            loss.append(Loss_Function(s, y[i]))
            accuracy.append(int(y[i] == predict))
        return np.mean(np.array(loss)), np.mean(np.array(accuracy))

    def Calculation_Gradient(self, G, y):
        grad_W = np.zeros((self.D, self.D))
        grad_A = np.zeros((self.D, 1))
        grad_b = 0
        now_Loss = self.Calculation_Loss(G, y)

        for i in range(self.D):
            for j in range(self.D):
                self.W[i][j] = self.W[i][j] + self.epsilon
                grad_W[i][j] = (self.Calculation_Loss(G, y) - now_Loss) / self.epsilon
                self.W[i][j] = self.W[i][j] - self.epsilon

        for i in range(self.D):
            self.A[i] = self.A[i] + self.epsilon
            grad_A[i] = (self.Calculation_Loss(G, y) - now_Loss) / self.epsilon
            self.A[i] = self.A[i] - self.epsilon

        # bに対して数値微分
        self.b = self.b + self.epsilon
        grad_b = (self.Calculation_Loss(G, y) - now_Loss) / self.epsilon
        self.b = self.b - self.epsilon

        return grad_W, grad_A, grad_b

    def Adam(self, train_G, train_y):
        # Adam特有のパラメータは、論文における値をそのまま流用しています(下記参照)
        # β1 = 0.9, β2 = 0.999 and  epsilon = 10^−8
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 10e-8
        Data_num = len(train_G)
        shuffle_index = list(range(Data_num))
        random.shuffle(shuffle_index)

        # ステップ数を更新
        self.t = self.t + 1

        # ミニバッチを用い、更新していく
        for start in range(0, Data_num, self.B):
            grad_W = []
            grad_A = []
            grad_b = []

            for i in range(start, min(start + self.B + 1, Data_num)):
                Gi_grad_W, Gi_grad_A, Gi_grad_b = self.Calculation_Gradient(train_G[shuffle_index[i]], train_y[shuffle_index[i]])
                grad_W.append(Gi_grad_W)
                grad_A.append(Gi_grad_A)
                grad_b.append(Gi_grad_b)

            del_theta_W = np.mean(grad_W, axis=0)
            del_theta_A = np.mean(grad_A, axis=0)
            del_theta_b = np.mean(np.array(grad_b))

            # 勾配の1次モーメント、2次モーメントの指数移動平均であるmW, vW等を更新
            self.mW = beta1 * self.mW + (1 - beta1) * del_theta_W
            self.vW = beta2 * self.vW + (1 - beta2) * del_theta_W ** 2
            self.mA = beta1 * self.mA + (1 - beta1) * del_theta_A
            self.vA = beta2 * self.vA + (1 - beta2) * del_theta_A ** 2
            self.mb = beta1 * self.mb + (1 - beta1) * del_theta_b
            self.vb = beta2 * self.vb + (1 - beta2) * del_theta_b ** 2

            # 更新した勾配の1次モーメント、2次モーメントを用い、パラメータを更新
            self.W = self.W - self.alpha * (self.mW / (1 - beta1 ** self.t)) / ((self.vW / (1 - beta2 ** self.t) + epsilon) ** 0.5)
            self.A = self.A - self.alpha * (self.mA / (1 - beta1 ** self.t)) / ((self.vA / (1 - beta2 ** self.t) + epsilon) ** 0.5)
            self.b = self.b - self.alpha * (self.mb / (1 - beta1 ** self.t)) / ((self.vb / (1 - beta2 ** self.t) + epsilon) ** 0.5)


# main function
if __name__ == "__main__":
    # グラフデータとラベルの読み込み
    path = "../datasets/train/"
    Graphs = []
    Labels = []
    for i in range(2000):
        graph_name = path + str(i) + "_graph.txt"
        label_name = path + str(i) + "_label.txt"

        with open(graph_name, "r") as f:
            A = [l.strip().split(" ") for l in f.readlines()]
        A = A[1:]
        A = [*map(lambda l: [int(n) for n in l], A)]

        with open(label_name, "r") as f:
            label = int(f.readlines()[0].strip())

        G = Graph(np.array(A))
        Graphs.append(G)
        Labels.append(label)

    # 分類器を用意(Adamを用いパラメータを更新していくもの)
    clf_Adam = Classifier()

    # 初期平均損失と初期平均精度を計算し、リストへ保存
    train_loss, train_acc = clf_Adam.Calculation_Loss_and_Accuracy(Graphs, Labels)
    train_loss = [train_loss]
    train_acc = [train_acc]

    # 分類器を訓練データセットで学習させ、平均損失と平均精度を計算しリストへ保存していく
    for epoch in range(1, 50):
        clf_Adam.Adam(Graphs, Labels)
        trainloss, trainacc = clf_Adam.Calculation_Loss_and_Accuracy(Graphs, Labels)
        train_loss.append(trainloss)
        train_acc.append(trainacc)

    # 学習済みの分類器であるclf_Adamを用い、テストデータセットに対する予測を行う
    Predicts = []  # 予測したラベルを格納するリスト
    TestDataList = glob.glob("../datasets/test/*.txt")  # testディレクトリの内部にある全てのグラフデータファイルのリスト

    for file in TestDataList:
        # グラフデータの読み込みとオブジェクトの作成
        graph_data = open(file, "r")
        A = []
        for line in graph_data:
            if not len(line.split(" ")) == 1:
                A = A + [[int(v) for v in line.split(" ")]]
        graph_data.close()
        A = Graph(np.array(A))

        # Sigmoid(s)の値を求め、1/2より大きいならば1、1/2以下ならば0と予測し、先述のリストへ予測したラベルを保存
        h_G = A.Aggregate(clf_Adam.W)
        s = clf_Adam.A.T.dot(h_G) + clf_Adam.b
        Predicts.append(int(Sigmoid(s) > 0.5))

    # 予測結果をファイルに書き込み
    with open("./prediction.txt", "w") as f:
        f.write("\n".join(map(str, Predicts)))

    plt.figure(figsize=(6, 4))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.subplot(2, 1, 1)
    plt.title("Train Loss")
    plt.plot(np.array(range(0, len(train_loss))), train_loss, 'k')
    plt.subplot(2, 1, 2)
    plt.title("Train Accuracy")
    plt.plot(np.array(range(0, len(train_acc))), train_acc, 'k')
    plt.show()
