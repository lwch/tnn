package model

// +------------------------+---------+
// |          NAME          |  COUNT  |
// +------------------------+---------+
// | transformer0_attention |    1968 |
// | transformer0_dense     | 1404224 |
// | transformer0_output    | 1402448 |
// | transformer1_attention |    1968 |
// | transformer1_dense     | 1404224 |
// | transformer1_output    | 1402448 |
// | output                 | 2630548 |
// | total                  | 8247828 |
// +------------------------+---------+
// 以下参数配置总计824万个参数

const embeddingDim = 8 // 4个float32表示一个字向量
const paddingSize = 74 // 最长为34*2，因此padding长度必须大于68+2
const heads = 4
const unitSize = paddingSize * embeddingDim
const maskSize = paddingSize * paddingSize
const batchSize = 128
const epoch = 200
const lr = 0.001
const transformerSize = 4
