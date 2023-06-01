package model

import (
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/tensor"
)

// forward 正向迭代
func (m *Model) forward(x *tensor.Tensor, paddingMasks []*tensor.Tensor, train bool) *tensor.Tensor {
	batchSize, _ := x.Dims()
	x = x.Add(buildPositionEmbedding(batchSize)) // 添加位置信息
	i := 0
	var y *tensor.Tensor
	for j := 0; j < transformerSize; j++ {
		y, i = m.forwardTransformer(i, x, paddingMasks, train)
	}
	y = m.layers[i].Forward(y, train)   // relu
	y = m.layers[i+1].Forward(y, train) // output
	y = y.Softmax(1)                    // softmax
	return y
}

// forwardTransformer 运行transformer层
func (m *Model) forwardTransformer(i int, x *tensor.Tensor, paddingMasks []*tensor.Tensor, train bool) (*tensor.Tensor, int) {
	// masks := make([]*tensor.Tensor, len(paddingMasks))
	// for i, m := range paddingMasks {
	// 	masks[i] = m.Add(featureMask) // 添加掩码隐藏未来信息
	// }
	y := m.layers[i].(*layer.SelfAttention).ForwardQKV(x, x, x, paddingMasks, train)
	y = y.Add(x)
	selfOut := m.layers[i+1].Forward(y, train) // nor
	y = m.layers[i+2].Forward(selfOut, train)  // dense
	y = m.layers[i+3].Forward(y, train)        // relu
	y = m.layers[i+4].Forward(y, train)        // dense
	y = y.Add(selfOut)
	y = m.layers[i+5].Forward(y, train) // nor
	return y, i + 6
}
