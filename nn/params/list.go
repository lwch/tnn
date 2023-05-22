package params

import (
	"sync"

	"github.com/lwch/tnn/nn/tensor"
)

type List struct {
	m    sync.RWMutex
	data []*tensor.Tensor
}

func NewList() *List {
	return &List{}
}

func (l *List) Add(t *tensor.Tensor) {
	l.m.Lock()
	defer l.m.Unlock()
	l.data = append(l.data, t)
}

func (l *List) Range(fn func(i int, t *tensor.Tensor)) {
	data := make([]*tensor.Tensor, len(l.data))
	l.m.RLock()
	for i, t := range l.data {
		data[i] = t
	}
	l.m.RUnlock()
	for i, t := range data {
		fn(i, t)
	}
}

func (l *List) Set(i int, t *tensor.Tensor) {
	l.m.Lock()
	defer l.m.Unlock()
	l.data[i] = t
}

func (l *List) Get(i int) *tensor.Tensor {
	l.m.RLock()
	defer l.m.RUnlock()
	return l.data[i]
}
