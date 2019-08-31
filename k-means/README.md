# k means

## 1. k means算法过程

1. 数据预处理，归一化，离群点处理。\
  因为k means算法对异常值敏感。如果有时候不作归一化，均值和方差大大维度将对数据的聚类结果产生决定性的影响

2. 随机选取K个簇中心，记为 <img src="/k-means/tex/70ae9dfe61f777b0f01b4626ad43a696.svg?invert_in_darkmode&sanitize=true" align=middle width=102.89031719999998pt height=26.76175259999998pt/>.

3. 优化目标其实是下述<p align="center"><img src="/k-means/tex/3e18d84ae7f07776ff805b836782e4b9.svg?invert_in_darkmode&sanitize=true" align=middle width=198.546942pt height=37.775108249999995pt/></p>
  对一系列凸函数取min其实是一个非凸的函数,因此k means算法得到的通常也是局部最优解。

4. 最后重复下述两个步骤，直到目标函数收敛
4.1 对于每一个样本<img src="/k-means/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/>,将其分配到距离最近的簇
4.2 对于每一类簇，重新计算该类簇的中心

##2. k means算法的优缺点
- 优点
   - 计算复杂度接近于线性，$\mathcal{O}(NKt)$，N是样本数量，K是簇的个数，t是迭代轮数。
   >4.1步骤，对于样本，需要计算K个距离，然后得到分配结果，那么，N个样本的计算复杂度就是$\mathcal{O}(NK)$\
   4.2步骤是重新分配簇中心，也就是扫描整个样本一次即可。

- 缺点
   - 受初始值、离群值的影响，每次结果不稳定
   - 通常是局部最优解
   - K值的选取不易

##3. K值选取的方案
1. 手肘法，就是画出总的误差随着k的一个曲线，然后取肘部位置的k，也就是当k的增大带来的增益较小的时候，可以认为当前的k比较合理
2. gap statistic: 用一个gap函数来刻画当前数据的聚类的损失函数和随机数据聚类的损失函数的差别。一般认为当前数据的损失和随机误差的损失差距越大，聚类的越好。
3. ISODATA算法自动选取k值

##4. k means算法的变种
1. k-means++：这个算法只改进了每次选取初始簇中心。在原有的算法是随机选取K的簇中心。k means++算法，依次迭代的取选取k个中心。假设已经选取了k-1个簇中心，那么计算剩下的每一个样本分别到这k-1个簇中心的距离，并最小的距离作为选择到该点的概率，计算完所有点之后需要归一化概率。那么距离越远的点就有更高的概率被选中，比较符合直觉。
2. ISODATA:![RUNOOB 图标](https://pic3.zhimg.com/v2-ebe6d577c8a70cd4e639a8a5621248be_b.jpg)