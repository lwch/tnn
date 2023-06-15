# xor

该示例使用一个简单的MLP模型来学习如何进行异或运算，以及如何保存和加载模型，模型训练参数如下：

```go
const lr = 1e-4
const hiddenSize = 10
const epoch = 10000
```

训练后的loss曲线如下：

![loss](loss.png)