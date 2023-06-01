package model

import (
	"math"

	"github.com/lwch/tnn/nn/tensor"
)

// 位置信息编码，由于padding size和embedding size固定，因此每一个样本的位置编码信息固定
var positionEmbedding []float64

func init() {
	positionEmbedding = make([]float64, unitSize)
	for k := 0; k < paddingSize; k++ {
		start := k * embeddingDim
		for i := 0; i < embeddingDim/2; i++ {
			n := float64(k) / math.Pow(10000, 2*float64(i)/float64(embeddingDim))
			positionEmbedding[start+i*2] = math.Sin(n)
			positionEmbedding[start+i*2+1] = math.Cos(n)
		}
	}
}

// buildPositionEmbedding 为每一个样本生成位置编码
func buildPositionEmbedding(batchSize int) *tensor.Tensor {
	data := make([]float64, batchSize*unitSize)
	for i := 0; i < batchSize; i++ {
		start := i * unitSize
		copy(data[start:start+unitSize], positionEmbedding)
	}
	return tensor.New(data, batchSize, unitSize)
}

// 未来信息掩码，上三角矩阵，注意：此处添加了对角线的掩码来降低该词与自身的权重
var featureMask *tensor.Tensor

func init() {
	featureMask = tensor.New(nil, paddingSize, paddingSize)
	for i := 0; i < paddingSize; i++ {
		for j := i; j < paddingSize; j++ {
			featureMask.Set(i, j, -1e9)
		}
	}
}

// buildPaddingMasks 生成padding的掩码
func buildPaddingMasks(masks [][]bool) []*tensor.Tensor {
	ret := make([]*tensor.Tensor, 0, len(masks))
	for batch := 0; batch < len(masks); batch++ {
		size := len(masks[batch])
		mask := tensor.New(nil, size, size)
		for i, b := range masks[batch] {
			if !b {
				continue
			}
			for j := 0; j < size; j++ {
				mask.Set(j, i, -1e9)
			}
		}
		ret = append(ret, mask)
	}
	return ret
}
