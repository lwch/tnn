package optimizer

import (
	"math"
	"sync"
	"sync/atomic"

	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/tensor"
)

type Adam struct {
	*base
	beta1, beta2 float64
	epsilon      float64

	mu   sync.Mutex
	init bool
	t    atomic.Uint64
	m, v *params.List
}

func NewAdam(lr, weightDecay, beta1, beta2, epsilon float64) *Adam {
	var adam Adam
	adam.base = new("adam", lr, weightDecay, adam.compute)
	adam.beta1 = beta1
	adam.beta2 = beta2
	adam.epsilon = epsilon
	adam.m = params.NewList()
	adam.v = params.NewList()
	return &adam
}

func (adam *Adam) initParams(grads *params.List) {
	adam.mu.Lock()
	defer adam.mu.Unlock()
	if adam.init {
		return
	}
	grads.Range(func(_ int, t *tensor.Tensor) {
		rows, cols := t.Dims()
		adam.m.Add(tensor.New(nil, rows, cols))
		adam.v.Add(tensor.New(nil, rows, cols))
	})
	adam.init = true
}

func (adam *Adam) compute(grads *params.List) *params.List {
	if !adam.init {
		adam.initParams(grads)
	}
	t := float64(adam.t.Add(1))
	ret := params.NewList()
	grads.Range(func(i int, ts *tensor.Tensor) {
		m := adam.m.Get(i)
		v := adam.v.Get(i)

		dm := ts.Grad().Sub(m).Scale(1 - adam.beta1)
		dv := ts.Grad().Pow(2).Sub(v).Scale(1 - adam.beta2)
		m.AddValue(dm.Value())
		v.AddValue(dv.Value())

		m = m.Scale(1 / (1 - math.Pow(adam.beta1, t)))
		v = v.Scale(1 / (1 - math.Pow(adam.beta2, t)))

		rows, cols := v.Dims()
		a := m.Scale(-adam.lr)
		b := v.Pow(0.5).Add(tensor.Numbers(rows, cols, adam.epsilon))
		ret.Add(a.DivElem(b))
	})
	return ret
}
