package model

import (
	"math"
	"math/rand"
	rt "runtime"
	"sync"

	"github.com/lwch/tnn/example/couplet/logic/sample"
	"github.com/lwch/tnn/nn/tensor"
)

// trainWorker 训练协程
func (m *Model) trainWorker(ch chan []int) {
	for {
		idx, ok := <-ch
		if !ok {
			return
		}
		x := make([]float64, 0, len(idx)*unitSize)
		y := make([]float64, 0, len(idx)*embeddingDim)
		paddingMask := make([][]bool, 0, batchSize)
		for _, idx := range idx {
			i := math.Floor(float64(idx) / float64(paddingSize))
			j := idx % paddingSize
			dx := m.trainX[int(i)]
			dy := m.trainY[int(i)]
			dy = append([]int{0}, dy...) // <s> ...
			var dz int
			if j < len(dy) {
				dz = dy[j]
				dy = dy[:j]
			} else {
				dz = -1
			}
			xTrain, zTrain, pm := sample.Build(append(dx, dy...), dz, paddingSize, m.embedding, m.vocabs)
			x = append(x, xTrain...)
			y = append(y, zTrain...)
			paddingMask = append(paddingMask, pm)
		}
		xIn := tensor.New(x, len(idx), unitSize)
		zOut := tensor.New(y, len(idx), len(m.vocabs))
		pred := m.forward(xIn, buildPaddingMasks(paddingMask), true)
		grad := m.loss.Loss(pred, zOut)
		grad.Backward(grad)
		m.current.Add(uint64(len(idx)))
	}
}

// trainEpoch 运行一个批次
func (m *Model) trainEpoch() {
	m.status = statusTrain
	m.current.Store(0)

	// 生成索引序列
	idx := make([]int, len(m.trainX)*paddingSize)
	for i := 0; i < len(idx); i++ {
		idx[i] = i
	}
	rand.Shuffle(len(idx), func(i, j int) {
		idx[i], idx[j] = idx[j], idx[i]
	})

	// 创建训练协程并行训练
	workerCount := rt.NumCPU()
	// workerCount = 1

	ch := make(chan []int)
	var wg sync.WaitGroup
	wg.Add(workerCount)
	for i := 0; i < workerCount; i++ {
		go func() {
			defer wg.Done()
			m.trainWorker(ch)
		}()
	}

	for i := 0; i < len(idx); i += batchSize {
		list := make([]int, 0, batchSize)
		for j := 0; j < batchSize; j++ {
			if i+j >= len(idx) {
				break
			}
			list = append(list, idx[i+j])
		}
		ch <- list
	}
	close(ch)
	wg.Wait()

	// 触发梯度更新
	m.chUpdate <- struct{}{}
}
