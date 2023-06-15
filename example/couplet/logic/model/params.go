package model

const embeddingDim = 32 // 32个float32表示一个字向量
const paddingSize = 70  // 最长为34*2，因此padding长度必须大于68
const heads = 4
const unitSize = paddingSize * embeddingDim
const maskSize = paddingSize * paddingSize
const batchSize = 128
const epoch = 200
const lr = 0.001
const transformerSize = 2
