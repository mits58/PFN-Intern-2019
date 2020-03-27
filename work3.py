# ライブラリのインポート
import random
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
        F[3] = 1
        for i in range(self.T):
            F = ReLU(W.dot(F.dot(self.A.T)))

        return F.dot(np.ones((self.V, 1)))


# --- 分類器クラス(課題2とほぼ同じため、説明を省略している部分があります) --- #
class Classifier:
    # コンストラクタ : 引数としてハイパパラメータをとり、変更できるようにしています
    def __init__(self, _D=8, _alpha=0.0001, _epsilon=0.001, _B=8, _eta=0.9):
        # ハイパパラメータ
        self.D = _D
        self.alpha = _alpha
        self.epsilon = _epsilon
        self.B = _B  # ミニバッチサイズ
        self.eta = _eta  # モーメントの値

        # 分類器のパラメータ集合
        self.W = np.random.normal(0.0, 0.4, (self.D, self.D))
        self.A = np.random.normal(0.0, 0.4, (self.D, 1))
        self.b = 0.0

        # Momentum SGDにおける、前ステップの更新量(ゼロで初期化)
        self.w_W = np.zeros((self.D, self.D))
        self.w_A = np.zeros((self.D, 1))
        self.w_b = 0.0

    # グラフGとラベルyに対して、その損失関数の値を計算するメソッド
    def Calculation_Loss(self, G, y):
        h_G = G.Aggregate(self.W)
        s = self.A.T.dot(h_G) + self.b
        return Loss_Function(s, y)

    # グラフデータとラベルデータの集合G, yに対して、その損失関数と精度の平均を計算するメソッド
    def Calculation_Loss_and_Accuracy(self, G, y):
        loss = []
        accuracy = []
        for i in range(len(G)):
            h_G = G[i].Aggregate(self.W)
            s = self.A.T.dot(h_G) + self.b
            predict = int(Sigmoid(s) > 0.5)     # predictには、分類器が予測したラベルが入っている
            loss.append(Loss_Function(s, y[i]))
            accuracy.append(int(y[i] == predict))   # int(y[i] == predict) は、予測したラベルと正解が一致した場合1、不一致の場合0になる
        return np.mean(np.array(loss)), np.mean(np.array(accuracy))

    # グラフGとラベルyに対して、その損失から勾配を計算するメソッド
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

        self.b = self.b + self.epsilon
        grad_b = (self.Calculation_Loss(G, y) - now_Loss) / self.epsilon
        self.b = self.b - self.epsilon

        return grad_W, grad_A, grad_b

    # Stochastic Gradient Descentに基づき、パラメータを更新するメソッド
    def Stochastic_Gradient_Descent(self, train_G, train_y):
        Data_num = len(train_G)  # 訓練データ数
        shuffle_index = list(range(Data_num))  # 1エポックにおいてサンプリングする順番を示すリストを作成
        random.shuffle(shuffle_index)  # 先述のリストをシャッフル

        # 1エポックの学習を行う(B個のミニバッチに対し勾配の計算を行っていく)
        for start in range(0, Data_num, self.B):
            # 勾配を格納するためのリスト
            grad_W = []
            grad_A = []
            grad_b = []

            # ミニバッチ内(shuffle_index内の[start:start + self.B + 1]の要素に対し、勾配を計算
            for i in range(start, min(start + self.B, Data_num)):
                Gi_grad_W, Gi_grad_A, Gi_grad_b = self.Calculation_Gradient(train_G[shuffle_index[i]], train_y[shuffle_index[i]])
                grad_W.append(Gi_grad_W)
                grad_A.append(Gi_grad_A)
                grad_b.append(Gi_grad_b)

            # 計算した勾配の平均をとる
            del_theta_W = np.mean(grad_W, axis=0)
            del_theta_A = np.mean(grad_A, axis=0)
            del_theta_b = np.mean(np.array(grad_b))

            # 勾配の平均をもとにパラメータを更新
            self.W = self.W - self.alpha * del_theta_W
            self.A = self.A - self.alpha * del_theta_A
            self.b = self.b - self.alpha * del_theta_b

    # Momentum SGDに基づき、パラメータを更新するメソッド(Stochastic Gradient Descentと同じ部分はコメントを省略しています)
    def Momentum_SGD(self, train_G, train_y):
        Data_num = len(train_G)
        shuffle_index = list(range(Data_num))
        random.shuffle(shuffle_index)

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

            # 勾配の平均と、前のステップにおける更新量に基づきパラメータを更新
            self.W = self.W - self.alpha * del_theta_W + self.eta * self.w_W
            self.A = self.A - self.alpha * del_theta_A + self.eta * self.w_A
            self.b = self.b - self.alpha * del_theta_b + self.eta * self.w_b

            # 前のステップにおける更新量を更新
            self.w_W = -self.alpha * del_theta_W + self.eta * self.w_W
            self.w_A = -self.alpha * del_theta_A + self.eta * self.w_A
            self.w_b = -self.alpha * del_theta_b + self.eta * self.w_b


# main function
if __name__ == "__main__":
    path = "../datasets/train/"     # データセットが存在するディレクトリのパス
    Graphs = []     # グラフデータ
    Labels = []     # ラベルデータ

    # 先述のパスに存在するグラフ、ラベルのデータを読み込む
    for i in range(2000):
        graph_name = path + str(i) + "_graph.txt"
        label_name = path + str(i) + "_label.txt"

        # グラフデータの読み込み
        with open(graph_name, "r") as f:
            A = [l.strip().split(" ") for l in f.readlines()]
            A = A[1:]
            A = [*map(lambda l: [int(n) for n in l], A)]

        # ラベルデータの読み込み
        with open(label_name, "r") as f:
            label = int(f.readlines()[0].strip())

        # 読み込んだデータからグラフクラスのインスタンスを生成
        A = np.array(A)
        G = Graph(A)

        # グラフとラベルのリストに追加
        Graphs.append(G)
        Labels.append(label)

    # 学習用データと検定用データに分割する
    train_G = Graphs[:int(len(Graphs) / 2)]
    train_y = Labels[:int(len(Labels) / 2)]
    test_G = Graphs[int(len(Graphs) / 2):]
    test_y = Labels[int(len(Labels) / 2):]

    Batch_Size = [1, 2, 4, 8, 16, 32, 64, 128]  # 今回変えていくバッチサイズのリスト
    colorlist = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']  # 色のリスト
    Epoch = 100  # エポック数

    # グラフ描画に関する初期化(Stochastic Gradient Descent、Momentum SGDのそれぞれにおけるグラフを初期化)
    fig_SGD, ax_SGD = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    fig_MSGD, ax_MSGD = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # 各バッチサイズについて実際に学習させ、グラフを描画していく
    for i, B in enumerate(Batch_Size):
        # SGD・Momentum SGDのそれぞれを用いパラメータを更新する学習器を用意
        clf_SGD = Classifier(_B=B)
        clf_MSGD = Classifier(_B=B)

        # 学習用データ、検定用データのそれぞれに対し、初期損失・初期精度を計算
        trainloss_SGD, trainacc_SGD = clf_SGD.Calculation_Loss_and_Accuracy(train_G, train_y)
        testloss_SGD, testacc_SGD = clf_SGD.Calculation_Loss_and_Accuracy(test_G, test_y)

        # 計算した平均損失と平均精度を格納するリストを定義
        train_loss_SGD = [trainloss_SGD]
        train_acc_SGD = [trainacc_SGD]
        test_loss_SGD = [testloss_SGD]
        test_acc_SGD = [testacc_SGD]

        # 前述と同様にMomentum SGDを用いる学習器においても計算を行い、リストを定義
        trainloss_MSGD, trainacc_MSGD = clf_MSGD.Calculation_Loss_and_Accuracy(train_G, train_y)
        testloss_MSGD, testacc_MSGD = clf_MSGD.Calculation_Loss_and_Accuracy(test_G, test_y)

        train_loss_MSGD = [trainloss_MSGD]
        train_acc_MSGD = [trainacc_MSGD]
        test_loss_MSGD = [testloss_MSGD]
        test_acc_MSGD = [testacc_MSGD]

        # エポックを進め学習を行っていく
        for ep in range(1, Epoch + 1):
            # SGD、Momentum SGDを用い分類器をそれぞれ1エポック学習させる
            clf_SGD.Stochastic_Gradient_Descent(train_G, train_y)
            clf_MSGD.Momentum_SGD(train_G, train_y)

            # SGDを用いる分類器に対する現在の平均損失・精度を学習用データ、検定用データのそれぞれに対し計算し先程のリストに追加
            trainloss_SGD, trainacc_SGD = clf_SGD.Calculation_Loss_and_Accuracy(train_G, train_y)
            testloss_SGD, testacc_SGD = clf_SGD.Calculation_Loss_and_Accuracy(test_G, test_y)

            train_loss_SGD.append(trainloss_SGD)
            train_acc_SGD.append(trainacc_SGD)
            test_loss_SGD.append(testloss_SGD)
            test_acc_SGD.append(testacc_SGD)

            # Momentum SGDを用いる分類器に対し同様のことを行う
            trainloss_MSGD, trainacc_MSGD = clf_MSGD.Calculation_Loss_and_Accuracy(train_G, train_y)
            testloss_MSGD, testacc_MSGD = clf_MSGD.Calculation_Loss_and_Accuracy(test_G, test_y)

            train_loss_MSGD.append(trainloss_MSGD)
            train_acc_MSGD.append(trainacc_MSGD)
            test_loss_MSGD.append(testloss_MSGD)
            test_acc_MSGD.append(testacc_MSGD)

        # SGDを用いる分類器に関して、得られた平均損失と平均精度のデータをプロットする
        ax_SGD[0, 0].plot(np.array(range(len(train_loss_SGD))), train_loss_SGD,
                          color=colorlist[i], label='$B={0}$'.format(B))
        ax_SGD[0, 1].plot(np.array(range(len(train_acc_SGD))), train_acc_SGD,
                          color=colorlist[i], label='$B={0}$'.format(B))
        ax_SGD[1, 0].plot(np.array(range(len(test_loss_SGD))), test_loss_SGD,
                          color=colorlist[i], label='$B={0}$'.format(B))
        ax_SGD[1, 1].plot(np.array(range(len(test_acc_SGD))), test_acc_SGD,
                          color=colorlist[i], label='$B={0}$'.format(B))

        # Momentum SGDを用いる分類器に関して同様のことを行う
        ax_MSGD[0, 0].plot(np.array(range(len(train_loss_MSGD))), train_loss_MSGD,
                           color=colorlist[i], label='$B={0}$'.format(B))
        ax_MSGD[0, 1].plot(np.array(range(len(train_acc_MSGD))), train_acc_MSGD,
                           color=colorlist[i], label='$B={0}$'.format(B))
        ax_MSGD[1, 0].plot(np.array(range(len(test_loss_MSGD))), test_loss_MSGD,
                           color=colorlist[i], label='$B={0}$'.format(B))
        ax_MSGD[1, 1].plot(np.array(range(len(test_acc_MSGD))), test_acc_MSGD,
                           color=colorlist[i], label='$B={0}$'.format(B))
        # 作成した分類器を削除する
        del clf_SGD
        del clf_MSGD

    # グラフのタイトル・凡例を設定
    ax_SGD[0, 0].set_title("Train Loss")
    ax_SGD[0, 1].set_title("Train Accuracy")
    ax_SGD[1, 0].set_title("Test Loss")
    ax_SGD[1, 1].set_title("Test Accuracy")
    ax_SGD[0, 0].legend(loc='upper right')

    ax_MSGD[0, 0].set_title("Train Loss")
    ax_MSGD[0, 1].set_title("Train Accuracy")
    ax_MSGD[1, 0].set_title("Test Loss")
    ax_MSGD[1, 1].set_title("Test Accuracy")
    ax_MSGD[0, 0].legend(loc='upper right')

    # グラフをpngファイルとして保存
    fig_SGD.savefig("SGD_Graph.png")
    fig_MSGD.savefig("MSGD_Graph.png")

    # グラフを表示
    plt.show()
