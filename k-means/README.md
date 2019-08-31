# k means

## 1. k means算法过程

1. 数据预处理，归一化，离群点处理。\
  因为k means算法对异常值敏感。如果有时候不作归一化，均值和方差大大维度将对数据的聚类结果产生决定性的影响

2. 随机选取K个簇中心，记为 <img src="/k-means/tex/70ae9dfe61f777b0f01b4626ad43a696.svg?invert_in_darkmode&sanitize=true" align=middle width=102.89031719999998pt height=26.76175259999998pt/>.

3. 优化目标<p align="center"><img src="/k-means/tex/a85c5edeb5b683a1def1993baa6689f5.svg?invert_in_darkmode&sanitize=true" align=middle width=203.13258075pt height=37.775108249999995pt/></p>
