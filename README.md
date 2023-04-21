# tnn

go版本神经网络框架，支持模型训练和预估

- [mlp](example/xor/): 该示例是一个四层的神经网络，用于训练异或运算
- [cnn](example/mnist/): 该示例是著名的手写数字识别示例

## 构造网络

首先定义网络的每个层

```go
initializer := initializer.NewXavierUniform(1)
var net net.Net
net.Set(
    layer.NewDense(16, initializer),
    activation.NewSigmoid(),
    layer.NewDense(8, initializer),
    activation.NewSigmoid(),
    layer.NewDense(4, initializer),
    activation.NewSigmoid(),
    layer.NewDense(2, initializer),
    activation.NewSigmoid(),
    layer.NewDense(1, initializer),
)
```

选定一个loss函数和优化器

```go
loss := loss.NewMSE()
optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
```

最后构造出模型并进行模型训练

```go
m := model.New(&net, loss, optimizer)
for i := 0; i < 10; i++ {
    m.Train(input, output)
    loss := m.Loss(input, output)
    fmt.Printf("Epoch: %d, Loss: %.05f\n", i, loss)
}
```

模型预测方法如下

```go
pred := model.Predict(input)
```

## 感谢

- [tinynn](https://github.com/borgwang/tinynn)
- [gonum](https://github.com/gonum/gonum)