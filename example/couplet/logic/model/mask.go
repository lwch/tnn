package model

import (
	"math"

	"github.com/lwch/gotorch/tensor"
)

// 位置信息编码，由于padding size和embedding size固定，因此每一个样本的位置编码信息固定
var positionEmbedding []float32

func init() {
	positionEmbedding = make([]float32, unitSize)
	for k := 0; k < paddingSize; k++ {
		start := k * embeddingDim
		for i := 0; i < embeddingDim/2; i++ {
			n := float32(k) / float32(math.Pow(10000, 2*float64(i)/float64(embeddingDim)))
			positionEmbedding[start+i*2] = float32(math.Sin(float64(n)))
			positionEmbedding[start+i*2+1] = float32(math.Cos(float64(n)))
		}
	}
}

// buildPositionEmbedding 为每一个样本生成位置编码
func buildPositionEmbedding(batchSize int64) *tensor.Tensor {
	data := make([]float32, batchSize*unitSize)
	for i := int64(0); i < batchSize; i++ {
		start := i * unitSize
		copy(data[start:start+unitSize], positionEmbedding)
	}
	return tensor.FromFloat32(storage, data, batchSize, paddingSize, embeddingDim)
}

// 未来信息掩码，上三角矩阵，注意：此处添加了对角线的掩码来降低该词与自身的权重
var featureMask []float32

func init() {
	featureMask = make([]float32, maskSize)
	for i := 0; i < paddingSize; i++ {
		start := i * paddingSize
		for j := i; j < paddingSize; j++ {
			featureMask[start+j] = -1e9
		}
	}
}

// buildFeatureMasks 为每一个样本生成未来信息掩码矩阵
func buildFeatureMasks(batchSize int64) *tensor.Tensor {
	data := make([]float32, batchSize*maskSize)
	for i := int64(0); i < batchSize; i++ {
		start := i * maskSize
		copy(data[start:start+maskSize], featureMask)
	}
	return tensor.FromFloat32(storage, data, batchSize, paddingSize, paddingSize)
}

// buildPaddingMasks 生成padding的掩码
func buildPaddingMasks(masks [][]bool) *tensor.Tensor {
	ret := make([]float32, 0, len(masks)*maskSize)
	for batch := 0; batch < len(masks); batch++ {
		mask := make([]float32, maskSize)
		for i, b := range masks[batch] {
			if !b {
				continue
			}
			for j := 0; j < paddingSize; j++ {
				// 行
				mask[i*paddingSize+j] = -1e9
				// 列
				mask[j*paddingSize+i] = -1e9
			}
		}
		ret = append(ret, mask...)
	}
	return tensor.FromFloat32(storage, ret, int64(len(masks)), paddingSize, paddingSize)
}