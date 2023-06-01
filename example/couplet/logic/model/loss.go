package model

import (
	"math"
	rt "runtime"
	"sync"

	"github.com/lwch/tnn/example/couplet/logic/sample"
	"github.com/lwch/tnn/nn/tensor"
)

// lossWorker loss计算协程
func (m *Model) lossWorker(ch chan []int, sum *float64) {
	for {
		idx, ok := <-ch
		if !ok {
			return
		}
		x := make([]float64, 0, len(idx)*unitSize)
		y := make([]float64, 0, len(idx)*embeddingDim)
		paddingMask := make([][]bool, 0, batchSize)
		for _, idx := range idx {
			i := math.Floor(float64(idx) / float64(paddingSize/2))
			j := idx % (paddingSize / 2)
			dx := m.trainX[int(i)]
			dy := m.trainY[int(i)]
			dy = append([]int{0}, dy...) // <s> ...
			var dz int
			if j < len(dy) {
				dz = dy[j]
				dy = dy[:j]
			} else {
				dz = -1 // padding
			}
			xTrain, zTrain, pm := sample.Build(append(dx, dy...), dz, paddingSize, m.embedding, m.vocabs)
			x = append(x, xTrain...)
			y = append(y, zTrain...)
			paddingMask = append(paddingMask, pm)
		}
		xIn := tensor.New(x, len(idx), unitSize)
		zOut := tensor.New(y, len(idx), len(m.vocabs))
		pred := m.forward(xIn, buildPaddingMasks(paddingMask), false)
		loss := m.loss.Loss(pred, zOut).Value()
		*sum += loss.At(0, 0)
		m.current.Add(uint64(len(idx)))
	}
}

func (m *Model) avgLoss() float64 {
	m.status = statusEvaluate
	m.current.Store(0)
	sum := 0.

	idx := make([]int, len(m.trainX)*paddingSize/2)
	for i := 0; i < len(idx); i++ {
		idx[i] = i
	}

	ch := make(chan []int)
	var wg sync.WaitGroup
	wg.Add(rt.NumCPU())
	for i := 0; i < rt.NumCPU(); i++ {
		go func() {
			defer wg.Done()
			m.lossWorker(ch, &sum)
		}()
	}

	var size float64
	var list []int
	for _, i := range idx {
		list = append(list, i)
		if len(list) < batchSize {
			continue
		}
		dup := make([]int, len(list))
		copy(dup, list)
		ch <- dup
		list = list[:0]
		size++
	}
	if len(list) > 0 {
		ch <- list
		size++
	}
	close(ch)
	wg.Wait()
	return sum / size
}
