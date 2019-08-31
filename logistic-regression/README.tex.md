#draft
#
Logistic Regression的损失函数是交叉熵，但是为什么不能是均方误差的损失函数呢？
首先，我们回顾一下LR的模型。LR的模型非常简单，就是对feature的线性组合之后用sigmoid函数将输出的值限制在[0,1]。
$$\hat{y}=\frac{1}{1+e^{-(w^Tx+b)}} $$
其损失函数为交叉熵损失函数，也就是用极大似然估计去得到参数w和b，因为这边输出的$\hat{y}$其实对应了属于1的概率。
$$L = y \log(\hat{y}) + (1-y)\log(1-\hat{y})  $$
接下来就是要反向传播梯度了，因为sigmoid函数有如下的特性
$$y = \sigma(x)\\ \frac{\partial y}{\partial x}=\sigma(x)(1-\sigma(x)) $$
因此，我们将损失函数对w和b求导，可以得到update的公式
$$\frac{\partial L}{\partial w_i}=\frac{y}{\hat{y}}\frac{\partial \hat{y}}{\partial w_i} - \frac{1-y}{1-\hat{y}}\frac{\partial \hat{y}}{\partial w_i}\\ =\frac{y}{\hat{y}}\hat{y}(1-\hat{y})x_i - \frac{1-y}{1-\hat{y}}\hat{y}(1-\hat{y})x_i\\ =y(1-\hat{y})x_i - (1-y)\hat{y}x_i\\ =(y-\hat{y})x_i $$
这个update的公式反应了，如果y和ŷ之间的很大，也就是预测的值和真实的值之间很大，那么每次更新的时候，将会有比较大的梯度方向去更新，因此是符合预期的。这里的推导简单起见都只考虑了单样本的情况。多样本的情况实际上就是对每一个样本的梯度求和平均起来去更新参数。
那么如果用均方损失误差会有什么后果呢？
均方损失误可以写成如下
$$L = (y-\hat{y})^2 $$
同样的，反向传播参数的时候后如下公式
$$\frac{\partial L}{\partial w_i} = 2(y-\hat{y})\frac{\partial \hat{y}}{\partial w_i} \\ =2(y-\hat{y})\hat{y}(1-\hat{y})x_i $$
这个公式有什么问题呢？
可以看出来$(y−ŷ )(y−y^)(y-\hat{y})$是反应了预测和真实值的一个误差，这点符合直观要求。但是后面这个乘积项是不合理的。
$\hat{y}(1-\hat{y})$
为什么呢？取个极端的例子，如果真实值是1，预测值是0，那么$ŷ (1−ŷ )=0$，此时梯度为0，无法再去更新参数，所以导致问题收敛到一个非常差的点。
综上，用均方误差去作为LR的损失函数是不合理的。
当然，直接用均方误差作为损失函数，该问题就不是一个凸问题了，这就是为什么会收敛到局部最优解的原因。

# kernal logistic regression
LR其实也是可以采取kernal method的，那么具体是如何采取的呢？回想一下，在svm的时候，我们在计算两个样本的内积的时候，定义了kernal function，将在原来维度的一个内积，变成了映射到另一个维度的内积，而且不需要在另一个维度具体的feature值。那么如何在LR中出现内积这一个term呢？

回想一下，为什么svm可以采用kernal function呢？可以从下述角度来理解。
svm的model是$f(x)=sign(wx+b)$，之所以会出现内积项，是因为参数$w$最后其实是一系列样本点的线性组合，也就是说$w=\sum_{n} \alpha_n y_n x_n$。从另一个角度来说，最优解是在所有样本张成的一个平面上的。那么LR是不是也是这样的呢？

一种解释可以从每一步的梯度更新来看，每次LR模型更新参数的时候，$w_i = w_{i-1} - (y-y_i)x_i$，每次都是加减一个权值为$y-y_i$的样本$x_i$，那么可以想象，在任意次更新之后，依然是一系列样本的线性组合。

广义上来看，只要是线性模型并且以L2norm作为正则项，其最有的参数w都是样本的线性组合，那么我们可以把LR的优化目标从
$$\min_{w} \frac{\lambda}{N}w^Tw+\frac{1}{N}\sum_{n}\log(1+\exp(-y_nw^Tx_n))$$变成
$$\min_{\beta} \frac{\lambda}{N}\sum_n\sum_m\beta_n\beta_m x_n^T x_m+\frac{1}{N}\sum_{n}\log(1+\exp(-y_n\sum_m \beta_m x_m^Tx_n))$$
那么，既然出现了样本内积项，就可以用kernal function映射得到高维的内积值，至此我们只需要优化参数$\beta$即可。

区别与svm，在svm中，其$\alpha$会是比较稀疏的，但是在kernal lr中，不像hinge loss在大于1的部分loss就是0了，所以$\beta$不是一个稀疏的。
