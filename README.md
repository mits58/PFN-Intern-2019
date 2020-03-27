# Preferred Networks Summer Internship Coding Tasks on Machine Learning
Preferred Networks（PFN）さんの2019年インターンシップコーディング課題（機械学習）の解答例です．
問題は[こちら](https://research.preferred.jp/2019/06/internship-coding-task-2019/)です。

## 課題概要
課題を通しGNNをフルスクラッチで実装し，実データで学習させてみようといったものでした．

課題1では，各頂点に特徴ベクトルが配置されたグラフに対する，集約（Aggregation）という演算の実装を行い，グラフ全体の特徴ベクトルを計算，
それを元に課題2では損失を計算することで，数値微分を行い各パラメータを変更していくGNNを作成するという流れになっています．
これを踏まえ，課題3では，確率的勾配降下法（Stochastic Gradient Descent）とMomentum SGDを用い，実グラフデータセットに対し学習を行い，
課題4は，パラメータ更新アルゴリズムなどの変更を加えた際の性能変化を見るものとなっています．

## 解答概要
work1.py 〜 work4.pyがそれぞれ課題1〜4に対応しています．
解答環境は以下を想定しています．
- Python 3.7.3
- numpy 1.16.2
- matplotlib 2.2.2

選択課題であった課題４では，最適化アルゴリズムとしてAdamを実装しました．
