## The loss function is `$J(\theta)$`, leanring rate `$\eta$`

# Gradient Descent
参数更新的公式
```math
\theta = \theta - \eta \nabla J(\theta)
```
通过函数的一阶性质，选取在当前参数下下降最多的方向，作为梯度的方向

# Variants of Gradient Descent
传统的梯度下降的缺点是当数据较大的时候，计算一个梯度是十分耗时的并且扫描一遍数据集之后只用来更新一次参数。
> 传统的梯度下降在数据集较大的时候十分低效。

## Stochasitc gradient descent
参数更新公式
```math
\theta = \theta - \eta \nabla J(\theta;x_i, y_i), \{x_i,y_i\} is~one~samlpe~in~training~set
```
在每次更新参数的时候，我们只需要计算`$J(\theta)$`对于当前样本的梯度，然后再更新参数。

显然，相比传统的梯度下降来说，参数更新非常的频繁，因此每一步参数更新时候的方差都很大。不过，这并不是一个坏事，因为方差大反而帮助了我们能够探索更多的空间，有可能能够得到更好的局部最优解。

不过方差的弊端也是十分突出的，频繁的更新使得SGD最后收敛非常不稳定。
> 即使我们通过逐渐减小`$\eta$`可以保证最后的收敛，但是收敛不稳定问题依然十分的突出

![image](https://miro.medium.com/max/484/1*BS5UuWEE_qXzoWBDQumgDA.png)

## Mini Batch Gradient Descent
Mini Batch Gradient Descent介于gd和sgd之间，通过选取一定数量的batch计算梯度，保证了计算效率以及一定的随机性。

通常的batch size设置为50-256之间

# Challenges faced while using Gradient Descent and its variants 
- 学习率的选取。太小的学习率使得收敛十分的慢，过大的学习率又可能导致收敛不稳定（在局部最优解之间波动等）。
- 所有参数都用了相同的学习率。如果数据十分的稀疏，我们更倾向于用更大的学习率去更新较少出现的feature。
- 如果`$J(\theta)$`非凸程度很大，经常收敛到saddle points而不是sub-optimal local minima。

# Optimizing the Gradient Descent
下面介绍一些对梯度下降系列方法的改进措施

## Momentum
SGD的高方差使得其很难收敛，因此Momentum通过综合考虑之前的梯度以及当前的梯度，来减小方差。
```math
V(t) = \gamma V(t-1) + \eta \nabla J(\theta)

\theta = \theta - V(t)
```

通常来说，`$\gamma$`设置为0.9。越大表示考虑更多之前的梯度信息。

优点
- 相比sgd更快且更稳定的收敛
- 减小了发生振荡的可能性

缺点
- 不会考虑之前梯度的优劣性，例如：如果函数出现比较陡峭的上升，那么基于动量的会导致不断的增加loss function的值。

## Nesterov accelerated gradient

## Adagrad
Adagrad根据参数调整了学习率，也就是说，对于较小更新的参数，会用较大的学习率，对于较多更新的参数，会用较小的学习率。因此Adagrad更加适合处理稀疏的数据。
```math
\theta_{i, t+1} = \theta_{i,t} - \frac{\eta}{\sqrt{G_{i,t}+\epsilon}} \nabla_{\theta_{i,t}} J(\theta)

G_{i,t} = G_{i,t-1} + \nabla_{\theta_{i,t}} J(\theta)
```
与传统的梯度下降相比，adagrad通过一个分母项来动态的调整对于每一个参数的学习率。

`$\sqrt{G_{i,t}+\epsilon}$`是一个梯度的积累项，如果`$\theta_i$`频繁更新，那么之前的梯度都会被积累下来，其学习率会更大，反之，其学习率会比较大。

优点
- 不用调节学习率，因为会动态的根据更新频繁程度以及梯度大小去调节

缺点
- 学习率一直都在减小，收敛较慢

## AdaDelta
Adagrad的修正版本，避免了学习率不断减小。 AdaDelta避免了不断的累计之前所有的梯度，只计算在一个窗口w中存放的梯度。如果存放之前的梯度是比较低效的，因此通过一个累计计算公式是比较合理的。
```math
\theta_{i, t+1} = \theta_{i,t} - \frac{\eta}{\sqrt{E[g_i^2]_t+\epsilon}} \nabla_{\theta_{i,t}} J(\theta)

g_{i,t} = \gamma*E[g_i^2]_{t-1} +(1-\gamma)(\nabla_{\theta_{i,t}} J(\theta))^2
```

## Adam
Adam 全称是 Adaptive Moment Estimation.
```math
g_t = \nabla_{\theta} f_t(\theta_{t-1})

m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t

v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2

\hat{m}_t = \frac{m_t}{1-\beta_1^t}

\hat{v}_t = \frac{v_t}{1-\beta_2^t}

\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t + \epsilon}}
```
adam 同时考虑了动量以及自适应的调节学习率


# 总结
method | update formula
---|---
Gradient Descent |`$\theta = \theta - \eta \nabla J(\theta)$`
Stochasitc gradient descent | `$\theta = \theta - \eta \nabla J(\theta;x_i, y_i), \{x_i,y_i\} is~one~samlpe~in~training~set$`
Momentum | `$V(t) = \gamma V(t-1) + \eta \nabla J(\theta),\quad \theta = \theta - V(t)$`
Adagrad | `$\theta_{i, t+1} = \theta_{i,t} - \frac{\eta}{\sqrt{G_{i,t}+\epsilon}} \nabla_{\theta_{i,t}} J(\theta), \quad G_{i,t} = G_{i,t-1} + \nabla_{\theta_{i,t}} J(\theta)$`
AdaDelta| `$\theta_{i, t+1} = \theta_{i,t} - \frac{\eta}{\sqrt{E[g_i^2]_t+\epsilon}} \nabla_{\theta_{i,t}} J(\theta), \quad g_{i,t} = \gamma*E[g_i^2]_{t-1} +(1-\gamma)(\nabla_{\theta_{i,t}} J(\theta))^2$`
Adam | `$g_t = \nabla_{\theta} f_t(\theta_{t-1}), m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,$` `$\hat{m}_t = \frac{m_t}{1-\beta_1^t},\hat{v}_t = \frac{v_t}{1-\beta_2^t}, \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t + \epsilon}}$`
