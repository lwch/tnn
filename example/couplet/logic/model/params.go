package model

const embeddingDim = 8 // 8个float64表示一个字向量
const paddingSize = 74 // 最长为34*2，因此padding长度必须大于68+2
const unitSize = paddingSize * embeddingDim
const batchSize = 64
const epoch = 10000
const lr = 0.001
const transformerSize = 2
