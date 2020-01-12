# python機器學習與深度學習
本文是我在讀"Python機器學習與深度學習特訓班：看得懂也會做的AI人工智慧實戰"這本書還有自己學習機器學習時的學習筆記與實作的內容
## Chapter1 Introduction

### 1.1 人工智慧、機器學習和深度學習的關係
<img src="./image/1.png" width="75%" />

#### 人工智慧:
指讓電腦具有人類的知識與行為。簡而言之，人工智慧主要在研究如何以電腦的程式技巧，執行一些需要人類智慧才能完成的工作。
#### 機器學習:
是人工智慧的一個分支，簡單來說機器學習就是透過特殊演算法，讓電腦能經由訓練從一大堆數據中找出規律性並產生模型，然後利用訓練出來的模型進行預測。
#### 深度學習:
是機器學習的一種方式，簡單來說深度學習就是透過各種神經網路，如多層感知器(MLP)、卷積神經網路(CNN)、循環神經網路(RNN)等，將一大堆數據輸入神經網路當中，讓電腦透過大量數據的訓練找出規律並自動學習，最後讓電腦能依據自動學習累積的經驗做出預測。

### 1.2 什麼是機器學習
機器學習上可分為訓練(Training)與預測(predict)。
機器學習的訓練資料由特徵(Features)和標籤(Label)組成。
將大量的數據資料經過特徵萃取後產生Features和Label，就可以透過機器學習產生模型，再用訓練完的模型對新的資料進行預測。
<img src="./image/2.png" width="75%"/>  

### 1.3 什麼是深度學習
深度學習利用電腦模擬人類的神經網路，並將神經網路分成多個層，一般會有1個輸入層(Input Layer)、隱藏層(Hidden Layer)和1個輸出層(Output Layer)，因為隱藏層可以是1層，也可以是非常多層，因此稱為深度學習。  
<img src="./image/3.PNG" width="75%"/>  
圖片來源:
https://medium.com/marketingdatascience/%E5%BF%AB%E9%80%9F%E5%8F%8D%E6%87%89%E6%A9%9F%E5%88%B6-%E9%A1%9E%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-a3bbdee4a6f6

神經網路並不一定是越多層效果就越好，有時候太多層反而會造成反效果。
### 1.4 Tensorflow與Keras
### 1.5 建置開發Tensorflow和Keras的虛擬環境
### 1.6 安裝GPU版的Tensorflow和Keras

## Chapter2 多層感知器(MLP)

### 2.1 認識多層感知器(MLP)
### 2.2 認識Mnist資料集
Mnist資料集收集整理了許多人0到9的手寫數字圖片所形成的資料集，其中包含了60000筆訓練資料，10000筆的測試資料。在Mnist資料集中，每一筆資料都是由images(數字圖片)和labels(真實數字)組成的黑白圖片資料。
<img src="./image/4.png" width="75%"/>  
圖片來源:https://docs.microsoft.com/zh-tw/archive/msdn-magazine/2014/june/test-run-working-with-the-mnist-image-recognition-data-set

#### 下載&讀取Mnist資料集
##### 下載Mnist資料集
在python中可直接透過keras下載Mnist資料集，如下:
```python
from keras.datasets import mnist 
```
mnist.load_data()第一次執行時會將資料下載到使用者的目錄下的<.keras\datasets>檔名為mnist.npz。
##### 讀取Mnist資料集 
每次執行mnist.load_data()時，會先去檢查Mnist資料集是否已經存在，如果已經存在就不用重複執行下載。載入後資料分別放在(train_feature, train_label)訓練資料和(test_feature, test_label)測試資料的變數中，如下:
```python
(train_feature, train_label),(test_feature, test_label)=mnist.load_data() 
```
#### 查看訓練資料
##### 顯示訓練資料內容
訓練資料是由每張都是28*/28的單色數字圖片(images)和數字圖片為0~9的真實值(labels)各60000筆所組成。可使用len()函式查看資料長度(數量):
```python
print(len(train_feature),len(train_label))    # 60000 60000
```



