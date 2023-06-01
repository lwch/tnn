package model

const embeddingDim = 2 // 2个float64表示一个字向量
const paddingSize = 74 // 最长为34*2，因此padding长度必须大于68+2
const unitSize = paddingSize * embeddingDim
const head = 8
const batchSize = 8
const epoch = 10000
const lr = 0.0001
const transformerSize = 16
