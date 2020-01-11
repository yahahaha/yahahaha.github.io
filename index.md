## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/yahahaha/yahahaha.github.io/edit/master/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/yahahaha/yahahaha.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
## **什麼是Tensorflow?**  
TensorFlow 是 Google透過使用資料流 (flow) 圖像，來進行數值演算的新一代開源機器學習工具。

## **Tensorflow安裝**  
### MacOS/Linux安裝  
用pip安裝(需先確認電腦中已安裝了pip。若電腦中已安裝了python3.X，因為pip已經自帶在python的模組裡，因此pip也已安裝了)  
#### CPU版本  
開啟terminal    
#如果安裝的是python2.X  
$pip install tensorflow  
#如果安裝的是python3.X  
$pip3 install tensorflow  
### Windows安裝  
支持python3.5(64bit)版本  
開啟command  
c:\>pip install tensorflow  

## **tensorflow數據流圖**
![image](https://github.com/yahahaha/tensorflow/blob/master/img/tensors_flowing.gif)

## **實例**
```python	
	import tensorflow as tf
	import numpy as np

	#create data
	x_data=np.random.rand(100).astype(np.float32)     #生成100個隨機數列，在tensorflow中大部分的數據的type是float32的形式
	y_data=x_data*0.1+0.3

	#create tensorflow structure start
	Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))  #tf.random_uniform(結構,左範圍,右範圍)，初始值是-1~1的數
	biases=tf.Variable(tf.zeros([1]))   #初始值是0

	y=Weights*x_data+biases

	loss=tf.reduce_mean(tf.square(y-y_data))   #誤差，預測的y和實際的y_data的差別
	optimizer=tf.train.GradientDescentOptimizer(0.5)   #利用optimizer減少誤差，GradientDescentOptimizer(學習效率)，學習效率<1
	train=optimizer.minimize(loss)

	init=tf.initiallize_all_variables()    #初始化變量
	#create tensorflow structure end

	sess=tf.Session()
	sess.run(init)         #very important

	for step in range(201):
		session.run(train)
		if step%20==0:
			print(step,sess.run(Weights),sess.run(biases))
```			
## **Session**
Tensorflow是基於圖架構進行運算的深度學習框架，Session是圖和執行者之間的媒介，首先透過Session來啟動圖，而Session.run()是用來進行操作的，Session再使用完過後需要透過close來釋放資源，或是透過with as的方式來讓他自動釋放。
```python	
	import tensorflow as tf

	matrix1=tf.constant([[3,3]])         #Constant就是不可變的常數
	matrix2=tf.constant([[2],[2]])
	product=tf.matmul(matrix1,matrix2)

	#method 1
	session=tf.Session()
	result=sess.run(product)
	print(result)
	sess.close()

	#method 2
	with tf.Session() as sess:
		result2= sess.run(product)
		print(result2)
```		
## **Variable**
將值宣告賦值給變數（Variables）讓使用者能夠動態地進行相同的計算來得到不同的結果，在TensorFlow中是以tf.Variable()來完成。  
在TensorFlow的觀念之中，宣告變數張量並不如Python那麼單純，它需要兩個步驟：  
1.宣告變數張量的初始值、類型與外觀   
2.初始化變數張量
```python	
	import tensorflow as tf

	state=tf.Variable(0,name='counter')
	#print(state.name)        #print出來的結果為counter:0
	one=tf.constant(1)

	new_value=tf.add(state,one)
	update=tf.assign(state,new_value)    #可以透過tf.assign()賦予不同的值，值得注意的地方是對變數張量重新賦值這件事對tensorflow來說也算是一個運算，必須在宣告之後放入Session中執行，否則重新賦值並不會有作用。
										
	init=tf.initialize_all_variables()   #must have if define variable

	with tf.Session() as sess:
		sess.run(init)
		for _ in range(3):
			sess.run(update)
			print(sess.run(state))
```
## **Placeholder**
我們可以將它想成是一個佔有長度卻沒有初始值的None，差異在於None不需要將資料類型事先定義，但是Placeholder必須事先定義好之後要輸入的資料類型與外觀。
```python	
	import tensorflow as tf

	input1=tf.placeholder(tf.float32)    #先定義好之後要輸入的資料類型，tf.placeholder(dtype,shape=None,name=None)
	input2=tf.placeholder(tf.float32)

	output=tf.mul(input1,input2)

	with tf.Session() as sess:
		print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))    #將資料以python dict餵進(feed)Placeholder之中，print出來的結果為[14.]
```
## **激勵函數Activation Function**
在類神經網路中使用激勵函數，主要是利用非線性方程式，解決非線性問題，若不使用激勵函數，類神經網路即是以線性的方式組合運算，因為隱藏層以及輸出層皆是將上層之結果輸入，並以線性組合計算，作為這一層的輸出，使得輸出與輸入只存在著線性關係，而現實中，所有問題皆屬於非線性問題，因此，若無使用非線性之激勵函數，則類神經網路訓練出之模型便失去意義。  
1.激勵函數需選擇可微分之函數，因為在誤差反向傳遞(Back Propagation)運算時，需要進行一次微分計算。  
2.在深度學習中，當隱藏層之層數過多時，激勵函數不可隨意選擇，因為會造成梯度消失(Vanishing Gradient)以及梯度爆炸(Exploding gradients)等問題。  
常見的激勵函數的選擇有sigmoid，tanh，ReLU，實用上最常使用ReLU。

## **例子-def add_layer()**
```python	
	import tensorflow as tf
	import numpy as np
	import matplotlib.pyplot as plt
	def add_layer(inputs,in_size,out_size,activation_function=None):     #add_layer(輸入值,輸入的大小,輸出的大小,激勵函數)
		Weights=tf.Variable(tf.random_normal([in_size,out_size]))    #在生成初始參數時，隨機變量(normal distribution)會比全部為0要好很多，所以這裡的Weights為一個in_size行，out_size列的隨機變量矩陣。
		biases=tf.Variable(tf.zeros([1,out_size])+0.1)               #在機器學習中，biases推薦的初始值不為零，所以+0.1
		Wx_plus_b=tf.matmul(inputs,Weights)+biases		
		if activation_function is None:				     #當activation_function為None時(非線性函數)，輸出就是當前的預測值Wx_plus_b，不為None時，就會把Wx_plus_b傳到activation_function()函數中得到輸出。
			outputs=Wx_plus_b
		else:
			outputs=activation_function(Wx_plus_b)
		return outputs
```
## **建造神經網路+結果可視化**
```python		
		#Make up some real data
		x_data=np.linspace(-1,1,300)[:,np.newaxis]       #定義輸入資料，為一個值域在-1到+1之間的300個數據，為300*1的形式。numpy.linspace(start, stop, num=50)在指定區間內返回均勻間隔的數字，np.newaxis的功能是增加一個新的維度，ex:[1 2 3]透過[:,np.newaxis]->[[1][2][3]]，[1 2 3]透過[np.newaxis,:]->[[1 2 3]]
		noise=np.random.normal(0,0.05,x_data.shape)      #然後定義一個與x_data形式一樣的噪音點，使得我們進行訓練的資料更像是真實的資料
		y_data=np.squre(x_data)-0.5+noise		 #定義y_data,假設為x_data的平方減去噪音點

		#define placeholder for inputs to network
		xs=tf.placeholder(tf.float32,[None,1])           #定義placeholder, 引數中none表示輸入多少個樣本都可以，x_data的屬性為1，所以輸出也為1
		ys=tf.placeholder(tf.float32,[None,1])
		l1=add_layer(x_data,1,10,activation_function=tf.nn.relu)    #我們定義一個簡單的神經網路，輸入層->隱藏層->輸出層，隱藏層我們假設只給10個神經元，輸入層是有多少個屬性就有多少個神經元，我們的x_data只有一個屬性，所以只有一個神經元，輸出層與輸入層是一樣的，輸入層有多少個神經元輸出層就有多少個。
		prediction=add_layer(l1,10,1,activation_function=None)	    #l1=add_layer(x_data,輸入層,隱藏層,activation_function=tf.nn.relu)，prediction=add_layer(l1,隱藏層,輸出層,activation_function=None)

		loss=tf.reduce_mean(tf.reduce_sum(tf.square(y_data-prediction),reduction_indices=[1]))    #計算預測值與真實值的差異
		train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

		init=tf.initialize_all_variables()
		sess=tf.Session()
		sess.run(init)
		
		fig=plt.figure()            #figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)，figure(圖的名稱,圖的大小ex=(4,3),參數指定繪圖對象的分辨率(即每英吋有多少個像素),背景顏色,邊框顏色,是否顯示邊框)
		ax=fig.add_subplot(1,1,1)   #add_subplot(1,1,1)表示1x1的網格，第1個子圖，add_subplot(A,B,C)表示AxB的網格，第C個子圖。
		ax.scatter(x_data,y_data)   #ax.scatter(x, y, z, c = 'r', marker = '^')表示產生散點圖，c表示顏色，marker表示點的形式(o是圓形的點，^是三角形)
		plt.ion()   #開啟交互模式->連續顯示圖，plt.ioff()->關閉交互模式
		plt.show()

		for i in range(1000):
			sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
			if i%50==0:
				print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
				try:
					ax.lines.remove(lines[0])
				except Exception:
					pass
				prediction_value=sess.run(prediction,feed_dict={xs:x_data})
				lines=ax.plot(x_data,prediction_value,'r-',lw=5)	#紅色，寬度為5
				plt.pause(0.1)	    #暫停0.1s
```
## **神經網路學習的優化(speed up training)**	
### 梯度下降法(gradient descent，GD)
梯度下降法是一種不斷去更新參數找「解」的方法，所以一定要先隨機產生一組初始參數的「解」，然後根據這組隨機產生的「解」開始算此「解」的梯度方向大小，然後將這個「解」去減去梯度方向，公式如下: ![image](https://github.com/yahahaha/tensorflow/blob/master/img/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%85%AC%E5%BC%8F.PNG)  
(t是第幾次更新參數，γ是學習率(Learning rate)，一次要更新多少，就是由學習率來控制的)  
參考:https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8-%E4%BA%8C-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95-gradient-descent-406e1fd001f
http://ruder.io/optimizing-gradient-descent/index.html#adam
### 隨機梯度下降法(stochastic Gradient Descent,SGD)	
在更新參數的時候，GD我們是一次用全部訓練集的數據去計算損失函數的梯度就更新一次參數。SGD就是一次跑一個樣本或是小批次(mini-batch)樣本然後算出一次梯度或是小批次梯度的平均後就更新一次，那這個樣本或是小批次的樣本是隨機抽取的，所以才會稱為隨機梯度下降法。  
SGD缺點:在當下的問題如果學習率太大，容易造成參數更新呈現鋸齒狀的更新，這是很沒有效率的路徑。
### Momentum
公式: ![image](https://github.com/yahahaha/tensorflow/blob/master/img/Momentum.PNG)   
t是第幾次更新參數，γ是學習率(Learning rate)，m是momentum項(一般設定為0.9)，主要是用在計算參數更新方向前會考慮前一次參數更新的方向(v(t-1))，如果當下梯度方向和歷史參數更新的方向一致，則會增強這個方向的梯度，若當下梯度方向和歷史參數更新的方向不一致，則梯度會衰退。然後每一次對梯度作方向微調。這樣可以增加學習上的穩定性(梯度不更新太快)，這樣可以學習的更快，並且有擺脫局部最佳解的能力。
### AdaGrad
SGD和momentum在更新參數時，都是用同一個學習率(γ)，Adagrad算法則是在學習過程中對學習率不斷的調整，這種技巧叫做「學習率衰減(Learning rate decay)」。通常在神經網路學習，一開始會用大的學習率，接著在變小的學習率。大的學習率可以較快走到最佳值或是跳出局部極值，但越後面到要找到極值就需要小的學習率。Adagrad則是針對每個參數客制化的值，![image](https://github.com/yahahaha/tensorflow/blob/master/img/adagrad.PNG)
這邊假設 g_t,i為第t次第i個參數的梯度，(ε是平滑項，主要避免分母為0的問題，一般設定為1e-7。Gt這邊定義是一個對角矩陣，對角線每一個元素是相對應每一個參數梯度的平方和。)  
Adagrad缺點是在訓練中後段時，有可能因為分母累積越來越大(因為是從第1次梯度到第t次梯度的和)導致梯度趨近於0，如果有設定early stop的，會使得訓練提前結束。early stop:在訓練中計算模型的表現開始下降的時候就會停止訓練。
### RMSProp
RMSProp和Adagrad一樣是自適應的方法，但Adagrad的分母是從第1次梯度到第t次梯度的和，所以和可能過大，而RMSprop則是算對應的平均值，因此可以緩解Adagrad學習率下降過快的問題。  
公式:i[image](https://github.com/yahahaha/tensorflow/blob/master/img/RMSProp.PNG)  
E[]在統計上就是取期望值，所以是取g_i^2的期望值，白話說就是他的平均數。ρ是過去t-1時間的梯度平均數的權重，一般建議設成0.9。
### Adam
Momentum是「計算參數更新方向前會考慮前一次參數更新的方向」， RMSprop則是「在學習率上依據梯度的大小對學習率進行加強或是衰減」。Adam則是兩者合併加強版本(Momentum+RMSprop+各自做偏差的修正)。 
![image](https://github.com/yahahaha/tensorflow/blob/master/img/adam1.PNG)  
![image](https://github.com/yahahaha/tensorflow/blob/master/img/adam2.PNG)  
mt和vt分別是梯度的一階動差函數和二階動差函數(非去中心化)。因為mt和vt初始設定是全為0的向量，Adam的作者發現算法偏量很容易區近於0，因此他們提出修正項，去消除這些偏量  
Adam更新的準則: (adam2)(建議預設值β1=0.9, β2=0.999, ε=10^(-8)。)

## **優化器optimizer**
參考:https://www.tensorflow.org/api_docs/python/tf/train  
https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/518746/

## **可視化**
```python
	import tensorflow as tf
	import numpy as np
	import matplotlib.pyplot as plt
*	def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):   
*		layer_name='layer%s'%n_layer
*		with tf.name_scope(layer_name):
*			with tf.name_scope('weight'):
*				Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')    
*				tf.histogram_summary(layer_name+'/weights',Weights)
*			with tf.name_scope('biases'):
*				biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')               
*				tf.histogram_summary(layer_name+'/biases',biases)
*			with tf.name_scope('Wx_plus_b'):
				Wx_plus_b=tf.matmul(inputs,Weights)+biases		
			if activation_function is None:				     
				outputs=Wx_plus_b
			else:
				outputs=activation_function(Wx_plus_b)
*			tf.histogram_summary(layer_name+'/outputs',outputs)
			return outputs

		#Make up some real data
		x_data=np.linspace(-1,1,300)[:,np.newaxis]       
		noise=np.random.normal(0,0.05,x_data.shape)      
		y_data=np.squre(x_data)-0.5+noise		

		#define placeholder for inputs to network
*		with tf.name_scope('inputs'):
*			xs=tf.placeholder(tf.float32,[None,1],name='x_input')           
*			ys=tf.placeholder(tf.float32,[None,1],name='x_input')
*		l1=add_layer(x_data,1,10,n_layer=1,activation_function=tf.nn.relu)    
*		prediction=add_layer(l1,10,1,n_layer=2,activation_function=None)	    

*		with tf.name_scope('loss'):
			loss=tf.reduce_mean(tf.reduce_sum(tf.square(y_data-prediction),reduction_indices=[1]))    
*		tf.scalar_summary('loss',loss)
*		with tf.name_scope('train'):
			train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

		
		init=tf.initialize_all_variables()
		sess=tf.Session()
*		merged=tf.merge_all_summaries()
*		writer=tf.train.SummaryWriter("logs/",sess.graph)
		sess.run(init)
		
		fig=plt.figure()            
		ax=fig.add_subplot(1,1,1)   
		ax.scatter(x_data,y_data)   
		plt.ion()   
		plt.show()

		for i in range(1000):
			sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
			if i%50==0:
*				result=sess.run(merged,feed_dict={xs:x_data,ys:y_data}))
*				writer.add_summary(result,i)	
	#存檔後開啟terminal，移動到檔案目錄後輸入tensorboard --logdir='logs/' 
	複製網址，在網址列上貼上搜尋
```
## **Classification分類學習**
```python
	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data   #mnist為一個手寫數字辨識資料的數據庫(有55000筆)
	
	#number 1 to 10 data
	mnist=input_data.read_data_sets('MNIST_data',one_hot=True)   #One-hot encoding 是將類別以 (0, 1) 的方式表示，之所以用 One-hot encoding 的原因是，一般來說，我們在做 Classification 時，其資料的 label 是用文字代表一個類別，例如做動物的影像辨識，label 可能會是 cat、dog、bird 等，但是類神經網路皆是輸出數值，所以我們無法判斷 34 與 cat 的差別。因此，One-hot encoding 便是在做 Classification 經常使用的一個技巧。

	def add_layer(inputs,in_size,out_size,activation_function=None):     
		Weights=tf.Variable(tf.random_normal([in_size,out_size]))    
		biases=tf.Variable(tf.zeros([1,out_size])+0.1)               
		Wx_plus_b=tf.matmul(inputs,Weights)+biases		
		if activation_function is None:				     
			outputs=Wx_plus_b
		else:
			outputs=activation_function(Wx_plus_b)
		return outputs

	def compute_accuracy(v_xs,v_ys):
		global prediction
		y_pre=sess.run(prediction,feed_dict={xs:v_xs})
		correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))   #判斷預測值與真實值是否一樣，correct_prediction 是一個 [True, False] 的陣列，再經由計算平均 (True=1，False=0)，就可以得到準確度 accuracy。
		accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
		return result

	#define placeholder for inputs to network
	xs=tf.placeholder(tf.float32,[None,784]) #每張圖的解析度是28x28，所以我們有28x28=784個像素資料		
	ys=tf.placeholder(tf.float32,[None,10])	 #每張圖都代表一個數字，有0~9，所以有10種 	

	#add output layer
	prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)   #呼叫add_layer函式搭建一個訓練的網路結構，只有輸入層和輸出層。其中輸入資料是784個特徵，輸出資料是10個特徵，激勵採用softmax函式，Softmax 回歸是邏輯回歸 (Logistic Regression) 的推廣，邏輯回歸適用於二元分類的問題，而 Softmax 回歸適用於多分類的問題。

	#the error between prediction and real data
	cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))  #loss，所使用的損失函數是交叉熵(Cross Entropy)。交叉熵是評估兩個機率分配(distribution) 有多接近，如果兩著很接近，則交叉熵的結果趨近於 0；反之，如果兩個機率分配差距較大，則交叉熵的結果趨近於 1。
	train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	
	sess=tf.Session()
	sess.run(tf.initialize_all_variables())
	
	for i in range(1000):
			batch_xs,batch_ys=mnist.train.next_batch(100)    #開始train，每次只取100張圖片，免得資料太多訓練太慢。
			sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
			if i%50==0:     #每訓練50次輸出一下預測精度
				print(compute_accuracy(mnist.test.images,mnist.test.labels))	
```

## **Overfitting**
Overfitting顧名思義就是過度學習訓練資料，變得無法順利去預測或分辨不是在訓練資料內的其他資料。  
有個方法可以偵測是否有Overfitting的情況發生:將所有的Training data坼成二部分，一個是Training Set跟Validate Set，Training Set就是真的把資料拿去訓練的，而Validate Set就是去驗證此Model在訓練資料外的資料是否可行。  
造成Overfitting的原因與解決方式:  
### 1.訓練資料太少
取得更多的資料，這個方法就是收集更多的資料，或是自行生成更多的有效資料。  
### 2.擁有太多的參數，功能太強的模型
a.減少參數或特徵或者是減少神經層數(其實就是在降低模型的大小，複雜的模型容易造成過度學習)  
b.在相同參數跟相同資料量的情況下，可以使用Regularization(正規化)  	
c.在相同參數跟相同資料量的情況下，可以使用Dropout  

### Regularization (正規化)
#### Weight decay(權重衰減)
Weight decay的意思就是對擁有較大權重的參數，課以罰金，藉此控制Overfitting的情況，因為Overfitting就是Weight 太大的時候可能會發生的問題。  
Weight decay的方式就是在loss function (損失函數)加入參數權重的L2 norm，就可以抑制權重變大，公式:![image](https://github.com/yahahaha/tensorflow/blob/master/img/weight_decay.PNG)  
(L是loss function，也就是損失函數，做Weight decay就是在loss function上加上Weight的L2 norm)  

### Dropout
在訓練的時候，隨機忽略掉一些神經元和神經聯結 ，使這個神經網絡變得”不完整”，然後用一個不完整的神經網絡訓練一次。到第二次再隨機忽略另一些, 變成另一個不完整的神經網絡。有了這些隨機drop掉的規則, 每一次預測結果都不會依賴於其中某部分特定的神經元。Dropout的方法就是一邊"隨機”消除神經元，一邊訓練的方法。

## **用dropout解決overfitting**
```python	
	from __future__ import print_function   #如果某個版本中出現了某個新的功能特性，而且這個特性和當前版本中使用的不相容，也就是它在該版本中不是語言標準，那麼我如果想要使用的話就需要從future模組匯入。 python2.X中print不需要括號，而在python3.X中則需要。
	import tensorflow as tf
	from sklearn.datasets import load_digits   #利用 Python 的機器學習套件 scikit-learn 將一個叫作 digits 的資料讀入。
	from sklearn.model_selection import train_test_split   #隨機劃分訓練集和測試集
	from sklearn.preprocessing import LabelBinarizer   #對於分類和文字屬性，需要將其轉換為離散的數值特徵才能餵給機器學習演算法。preprocessing.LabelBinarizer是一個很好用的工具，比如可以把yes和no轉化為0和1，或是把incident和normal轉化為0和1

	# load data
	digits = load_digits()
	X = digits.data
	y = digits.target
	y = LabelBinarizer().fit_transform(y)  #標準化
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


	def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    		# add one more layer and return the output of this layer
    		Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    		biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    		Wx_plus_b = tf.matmul(inputs, Weights) + biases
    		
		# here to dropout
    		Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    		if activation_function is None:
        		outputs = Wx_plus_b
    		else:
        		outputs = activation_function(Wx_plus_b, )
    		tf.summary.histogram(layer_name + '/outputs', outputs)
    		return outputs


	# define placeholder for inputs to network
	keep_prob = tf.placeholder(tf.float32)
	xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
	ys = tf.placeholder(tf.float32, [None, 10])

	# add output layer
	l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
	prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

	# the loss between prediction and real data
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))  # loss
	tf.summary.scalar('loss', cross_entropy)
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.Session()
	merged = tf.summary.merge_all()
	# summary writer goes in here
	train_writer = tf.summary.FileWriter("logs/train", sess.graph)
	test_writer = tf.summary.FileWriter("logs/test", sess.graph)

	# tf.initialize_all_variables() no long valid from
	# 2017-03-02 if using tensorflow >= 0.12
	if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    		init = tf.initialize_all_variables()
	else:
    		init = tf.global_variables_initializer()
		sess.run(init)
		for i in range(500):
    			# here to determine the keeping probability
    			sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
    			if i % 50 == 0:
        			# record loss
        			train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        			test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        			train_writer.add_summary(train_result, i)
        			test_writer.add_summary(test_result, i)
```
## **甚麼是卷積神經網路Convolutional Neural Networks(CNN)** 
傳統的DNN（即Deep neural network，泛指一般的深度學習網路）最大問題在於它會忽略資料的形狀。例如，輸入影像的資料時，該data通常包含了水平、垂直、color channel等三維資訊，但傳統DNN的輸入處理必須是平面的、也就是須一維的資料。舉例用DNN來分類MNIST手寫數字集？其影像資訊是水平28 pixels、垂直28 pixels、color channel=1，即(1, 28, 28)的形狀，但輸入DNN時，所有dataset必須轉為一維，欄位數為784的dataset。  	
因此，若去除了這些形狀資訊，就代表失去了一些重要的空間資料，像不同影像但類似的空間可能有著相似的像素值，RGB不同的channel之間也可能具有某些關連性、而遠近不同的像素彼此也應具有不同的關聯性，而這些資訊只有在三維形狀中才能保留下來。  
因此，Deep learning中的CNN較傳統的DNN多了Convolutional（卷積）及池化（Pooling） 兩層layer，用以維持形狀資訊並且避免參數大幅增加。在加入此兩層後，我們所看到的架構就如下圖分別有兩層的卷積和池化層，以及一個全連結層（即傳統的DNN），最後再使用Softmax activation function來輸出分類結果。
簡單來說，圖片經過各兩次的Convolution, Pooling, Fully Connected就是CNN的架構了。 
![image](https://github.com/yahahaha/tensorflow/blob/master/img/CNN%E6%9E%B6%E6%A7%8B.PNG)
	
### Convolutional layer卷積層
如果使用傳統的深度學習網路(例如全連接層)來識別圖像，那麼原本是二維的圖片就必須先打散成一維，然後再將每個像素視為一個特徵值丟入DNN架構進行分析，因此這些輸入的像素已經丟失了原有的空間排列資訊。然而CNN的Convolution layer的目的就是在保留圖像的空間排列並取得局部圖像作為輸入特徵。  
卷積運算就是將原始圖片的與特定的Feature Detector(filter)做卷積運算，卷積運算就是將下圖兩個3x3的矩陣作相乘後再相加，以下圖為例
![image](https://github.com/yahahaha/tensorflow/blob/master/img/Convolutional%20layer.PNG)
0 x 0 + 0 x 0 + 0 x 1 + 0 x 1 + 1 x 0 + 0 x 0 + 0 x 0 + 0 x 1 + 0 x 1 = 0      
依序做完整張表
![image](https://github.com/yahahaha/tensorflow/blob/master/img/%E4%BE%9D%E5%BA%8F%E5%81%9A%E5%AE%8C%E6%95%B4%E5%BC%B5%E8%A1%A8.PNG)
中間的Feature Detector(Filter)會隨機產生好幾種(ex:16種)，Feature Detector的目的就是幫助我們萃取出圖片當中的一些特徵(ex:形狀)，就像人的大腦在判斷這個圖片是什麼東西也是根據形狀來推測  
![image](https://github.com/yahahaha/tensorflow/blob/master/img/16%E7%A8%AE%E4%B8%8D%E5%90%8C%E7%9A%84Feature%20Detector.PNG)
然而如果我們輸入的是三層的RGB圖像而非單層的灰階呢？或是想要使用多個Feature Detector(filter)來取得不同的特徵，那麼就需要在同一卷積層中定義多個Feature Detector(filter)，此時Feature Detector(filter)的數量就代表其Feature Detector(filter)的維度。當Feature Detector(filter)維度愈大，代表使用的Feature Detector(filter)種類愈多提取的圖像特徵也就越多，圖像識別的能力也就更好。  
	
### Pooling Layer 池化層
Pooling layer稱為池化層，它的功能很單純，就是將輸入的圖片尺寸縮小（大部份為縮小一半）以減少每張feature map維度並保留重要的特徵，其好處有：  
1.減少後續layer需要參數，加快系統運作的效率。  
2.具有抗干擾的作用：圖像中某些像素在鄰近區域有微小偏移或差異時，對Pooling layer的輸出影響不大，結果仍是不變的。  
3.減少過度擬合over-fitting的情況。    
與卷積層相同，池化層會使用Feature Detector(filter)來取出各區域的值並運算，但最後的輸出並不透過Activate function（卷積層使用的function是ReLU）。另外，池化層用來縮小圖像尺寸的作法主要有三種：最大化（Max-Pooling）、平均化（Mean-Pooling）、隨機（Stochastic-Pooling）…等，以Max-pooling為例，作法說明如下：
![image](https://github.com/yahahaha/tensorflow/blob/master/img/Pooling%20layer%20%E5%81%B6%E6%95%B8.PNG)
![image](https://github.com/yahahaha/tensorflow/blob/master/img/Pooling%20layer%20%E5%A5%87%E6%95%B8.PNG)
### Full connected layer
Full connected layer指的就是一般的神經網路，基本上全連接層的部分就是將之前的結果平坦化之後接到最基本的神經網絡了。  
可以看出池化層減少了圖素的參數數量，卻保留了所有重要的特徵資訊，對於CNN的運作效率增進不少。

## **CNN實作**
```python
	from __future__ import print_function
	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	# number 1 to 10 data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	def compute_accuracy(v_xs, v_ys):   #計算準確度
	    global prediction
	    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
	    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
	    return result

	def weight_variable(shape):
	    initial = tf.truncated_normal(shape, stddev=0.1)    #tf.truncated_normal()函式是一種“截斷”方式生成正太分佈隨機值，“截斷”意思指生成的隨機數值與均值的差不能大於兩倍中誤差，否則會重新生成。
	    return tf.Variable(initial)

	def bias_variable(shape):
	    initial = tf.constant(0.1, shape=shape)
	    return tf.Variable(initial)

	def conv2d(x, W):  #這個輸出，就是我們的feature map
	    # stride [1, x_movement, y_movement, 1]
	    # Must have strides[0] = strides[3] = 1
	    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')    #tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)

	def max_pool_2x2(x):   #Pooling
	    # stride [1, x_movement, y_movement, 1]
	    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	# define placeholder for inputs to network
	xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
	ys = tf.placeholder(tf.float32, [None, 10])
	keep_prob = tf.placeholder(tf.float32)
	x_image=tf.reshape(xs,[-1,28,28,1])
	#print(x_image.shape)  #[n_samples,28,28,1]

	## conv1 layer ##
	W_conv1=weight_variable([5,5,1,32])  #patch 5x5,in size 1,out size 32
	b_conv1=bias_variable([32])
	h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #output size 28x28x32 
	h_pool1=max_pool_2x2(h_conv1)  #output size 14x14x32

	## conv2 layer ##
	W_conv2=weight_variable([5,5,32,64])  #patch 5x5,in size 32,out size 64
	b_conv2=bias_variable([64])
	h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) #output size 14x14x64 
	h_pool2=max_pool_2x2(h_conv2)  #output size 7x7x64

	## func1 layer ##
	W_fc1=weight_variable([7*7*64,1024])
	b_fcl=bias_variable([1024])
	#[n_samples,7,7,64]->>[n_samples,7*7*64]
	h_pool2_glat=tf.reshape(h_pool2,[-1,7*7*64])
	h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
	h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

	## func2 layer ##
	W_fc2=weight_variable([1024,10])
	b_fc2=bias_variable([10])
	prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

	# the error between prediction and real data
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))       # loss
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	sess = tf.Session()
	# important step
	# tf.initialize_all_variables() no long valid from
	# 2017-03-02 if using tensorflow >= 0.12
	if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	    init = tf.initialize_all_variables()
	else:
	    init = tf.global_variables_initializer()
	sess.run(init)

	for i in range(1000):
	    batch_xs, batch_ys = mnist.train.next_batch(100)
	    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
	    if i % 50 == 0:
		print(compute_accuracy(
		    mnist.test.images[:1000], mnist.test.labels[:1000]))
```python


