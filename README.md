# XGBoost-Homework

我们使用了遗传算法来求解最优参数。结果实验，发现各个参数取如下值时，在测试集上RMSE可以达到3.656079（此时，迭代次数为1191）。

> 
> 'objective': 'count:poisson',
> 
> 'max_depth': 3,
> 
> 'eta': 0.04375073666234408,
> 
> 'gamma': 0
> 


### baseline.py:　	
> 训练模型。80% 训练集，20% 验证集。 <br>
> 结果在 1103 轮达到最优，test-rmse 为 3.6608。
	
### predict.py 
> 使用全部的训练数据，训练 1103 轮，预测结果输出至 predict.txt
	
### pca.py 
> 预处理对连续属性做了 PCA 降维，失败的试验品。
	
### plot.py ·
> 对 Y 值的分布进行统计，第一幅图输出分布的频数，第二幅图输出每个数值所占百分比。
	
### predict.txt 
> 预测结果文件
