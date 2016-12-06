# XGBoost-Homework
* baseline.py:　	<br>
> 训练模型。80% 训练集，20% 验证集。 <br>
> 结果在 1103 轮达到最优，test-rmse 为 3.6608。
	
* predict.py <br>
> 使用全部的训练数据，训练 1103 轮，预测结果输出至 predict.txt
	
* pca.py <br>
> 预处理对连续属性做了 PCA 降维，失败的试验品。
	
* plot.py <br>
> 对 Y 值的分布进行统计，第一幅图输出分布的频数，第二幅图输出每个数值所占百分比。
	
* predict.txt <br>
> 预测结果文件
