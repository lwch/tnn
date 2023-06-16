package model

const embeddingDim = 128 // 128个float32表示一个字向量
const paddingSize = 34   // 最长为34
const heads = 8
const maskSize = paddingSize * paddingSize
const batchSize = 32
const epoch = 1000
const warmup = 200
const transformerSize = 2
