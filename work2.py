# ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

''' 使用する関数の定義 '''
# ReLU関数
def ReLU(x):
  return np.maximum(0, x)

# Sigmoid関数
def Sigmoid(s):
  return 1 / (1 + np.exp(-s))

# binary cross-entropy 損失関数(loss) 詳細はREADME.mdに記載
def Loss_Function(s, y):
  if s >= 35:
    return y * np.log(1 + np.exp(-s)) + (1 - y) * s
  elif s <= -35:
    return -s * y + (1 - y) * np.log(1 + np.exp(s))
  else:
    return y * np.log(1 + np.exp(-s)) + (1 - y) * np.log(1 + np.exp(s))


''' Graphクラス(課題1と同様) '''
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


''' 分類器クラス '''
class Classifier:
  # コンストラクタ
  def __init__(self):
    # ハイパパラメータ
    self.D = 8  # Wの次元数
    self.alpha = 0.0001  # 学習率
    self.epsilon = 0.001  # 微分摂動
    
    # 分類器のパラメータ集合
    # W, Aの各要素は、平均0、標準偏差0.4の正規分布からサンプリングし初期化
    # bは0で初期化
    self.W = np.random.normal(0, 0.4, (self.D, self.D))
    self.A = np.random.normal(0, 0.4, (self.D, 1))
    self.b = 0

  # グラフGとラベルyに対して、損失を計算するメソッド
  def Calculation_Loss(self, G, y):
    h_G = G.Aggregate(self.W)
    s = self.A.T.dot(h_G) + self.b
    return Loss_Function(s, y)

  # グラフGとラベルyに対して、その損失から勾配を計算するメソッド
  def Calculation_Gradient(self, G, y):
    # 計算した勾配を代入するための変数
    grad_W = np.zeros((self.D, self.D))
    grad_A = np.zeros((self.D, 1))
    grad_b = 0

    # 今の損失を計算
    now_Loss = self.Calculation_Loss(G, y)

    # Wの各要素に対して数値微分
    for i in range(self.D):
      for j in range(self.D):
        self.W[i][j] = self.W[i][j] + self.epsilon  # 摂動分パラメータを増やす
        grad_W[i][j] = (self.Calculation_Loss(G, y) - now_Loss) / self.epsilon  # 数値微分を行う
        self.W[i][j] = self.W[i][j] - self.epsilon  # 増やしたパラメータを戻す
        
    # Wと同様にAの各要素に対して数値微分 
    for i in range(self.D):
      self.A[i] = self.A[i] + self.epsilon
      grad_A[i] = (self.Calculation_Loss(G, y) - now_Loss) / self.epsilon
      self.A[i] = self.A[i] - self.epsilon

    # Wと同様にbに対して数値微分
    self.b = self.b + self.epsilon
    grad_b = (self.Calculation_Loss(G, y) - now_Loss) / self.epsilon
    self.b = self.b - self.epsilon

    # 計算した勾配を返す
    return grad_W, grad_A, grad_b

# main function
if __name__ == "__main__":
  path = "../datasets/train/" # データセットが存在するディレクトリのパス
  data_ids = ["10", "20"] # 今回用いるグラフデータのID

  plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
  # data_idsで指定したデータセットに対し、パラメータを更新していくと損失が下がることをグラフで確認する
  for id in data_ids:
    # ファイルからグラフデータ、ラベルデータの読み込み
    graph_name = path + id + "_graph.txt"
    label_name = path + id + "_label.txt"
  
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

    # 分類器クラスのインスタンスを生成
    clf = Classifier()

    # 初期値における損失を計算
    loss = clf.Calculation_Loss(G, label)

    LossData = [loss] # 各エポックにおける損失を保存するリスト
    
    # 500回パラメータを更新する
    for i in range(0, 500):
      grad_W, grad_A, grad_b = clf.Calculation_Gradient(G, label)
      clf.W = clf.W - clf.alpha * grad_W
      clf.A = clf.A - clf.alpha * grad_A
      clf.b = clf.b - clf.alpha * grad_b

      # 損失を計算し記録する
      LossData.append(float(clf.Calculation_Loss(G, label)))

    # idに対応するグラフデータにおける損失のグラフを描画
    plt.subplot(2, 1, int(int(id)/10))
    plt.title('Loss Graph(ID:{0})'.format(id))
    plt.xlabel('Reputation')
    plt.ylabel('Loss')
    plt.plot(list(range(0, len(LossData))), LossData)
    
  plt.show()
