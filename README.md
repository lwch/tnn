# tnn

go版本神经网络框架，支持模型训练和推理，libgotorch库的安装方式请看[libgotorch安装](https://github.com/lwch/gotorch#%E5%AE%89%E8%A3%85)

## 工具

- ~~[minfo](cmd/minfo/): 这是tnn框架中的一个工具，用于查看保存模型的定义信息~~

## 示例

- [xor](example/xor/): 该示例是一个四层的神经网络，用于训练异或运算（回归）
- [sin](example/sin/): 该示例使用rnn网络来进行sin函数的时序任务训练（回归）
- [sin_attention](example/sin_attention/): 使用transformer来实现sin曲线的预测（回归）
- [couplet](example/couplet/): 使用GPT模型来对对联（回归）

## 构造网络

首先定义网络的每个层

```go
var net net.Net
net.Add(
    layer.NewDense(16),
    activation.NewSigmoid(),
    layer.NewDense(8),
    activation.NewSigmoid(),
    layer.NewDense(4),
    activation.NewSigmoid(),
    layer.NewDense(2),
    activation.NewSigmoid(),
    layer.NewDense(1),
)
```

选定一个优化器

```go
optimizer := optimizer.NewAdam()
```

最后构造出模型并进行模型训练

```go
for i := 0; i < 10; i++ {
    pred := net.Forward(input, true)
    loss := loss.NewMse(pred, output)
    loss.Backward()
    optimizer.Step(net.Params())
    m.Apply()
    fmt.Printf("Epoch: %d, Loss: %.05f\n", i, loss.Float32Value()[0])
}
```

模型预测方法如下

```go
pred := model.Forward(input, false)
```

## 感谢

- [tinynn](https://github.com/borgwang/tinynn)
- [gonum](https://github.com/gonum/gonum)
- [pytorch](https://pytorch.org)