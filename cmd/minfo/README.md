# minfo

该工具用于查看模型配置信息，如层定义、loss函数定义、optimizer定义等。

## 安装

```shell
go install github.com/lwch/tnn/cmd/minfo@latest
```

## 使用方法

```shell
minfo <xxx.model>
```

输出内容如下：

```yaml
Model: <unset>
Train Count: 40000
Param Count: 233
Loss Func: mse
Optimizer: adam
  - Learning Rate: 0.0001
  - Weight Decay: 0
  - beta1: 0.9
  - beta2: 0.999
  - epsilon: 1e-08
Layers:
  - Class: dense
    Name: hidden1
    Output Count: 16
    Params:
      - w: 2x16
      - b: 1x16
  - Class: sigmoid
    Name: sigmoid
  - Class: dense
    Name: hidden2
    Output Count: 8
    Params:
      - w: 16x8
      - b: 1x8
  - Class: sigmoid
    Name: sigmoid
  - Class: dense
    Name: hidden3
    Output Count: 4
    Params:
      - w: 8x4
      - b: 1x4
  - Class: sigmoid
    Name: sigmoid
  - Class: dense
    Name: hidden4
    Output Count: 2
    Params:
      - b: 1x2
      - w: 4x2
  - Class: sigmoid
    Name: sigmoid
  - Class: dense
    Name: output
    Output Count: 1
    Params:
      - w: 2x1
      - b: 1x1
```