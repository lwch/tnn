package model

import "github.com/lwch/gotorch/consts"

const embeddingDim = 128 // 128个float64表示一个字向量
const paddingSize = 34   // 最长为34
const heads = 8
const maskSize = paddingSize * paddingSize
const batchSize = 128
const epoch = 200
const lr = 0.001
const transformerSize = 4
const device = consts.KCPU
