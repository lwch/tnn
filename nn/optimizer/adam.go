package optimizer

import (
	"math"
	"sync"
	"sync/atomic"

	"github.com/lwch/tnn/internal/pb"
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
	m, v []*params.Params
}

func NewAdam(lr, weightDecay, beta1, beta2, epsilon float64) *Adam {
	var adam Adam
	adam.base = new("adam", lr, weightDecay, adam.compute)
	adam.beta1 = beta1
	adam.beta2 = beta2
	adam.epsilon = epsilon
	return &adam
}

func (adam *Adam) initParams(grads []*params.Params) {
	adam.mu.Lock()
	defer adam.mu.Unlock()
	if adam.init {
		return
	}
	adam.m = make([]*params.Params, len(grads))
	adam.v = make([]*params.Params, len(grads))
	for i := 0; i < len(grads); i++ {
		ps := grads[i]
		adam.m[i] = params.New()
		adam.v[i] = params.New()
		ps.Range(func(name string, t *tensor.Tensor) {
			rows, cols := t.Dims()
			adam.m[i].Set(name, tensor.New(nil, rows, cols))
			adam.v[i].Set(name, tensor.New(nil, rows, cols))
		})
	}
	adam.init = true
}

func (adam *Adam) compute(grads []*params.Params) []*params.Params {
	if !adam.init {
		adam.initParams(grads)
	}
	t := float64(adam.t.Add(1))
	ret := make([]*params.Params, len(grads))
	for i := 0; i < len(grads); i++ {
		ret[i] = params.New()
		ps := grads[i]
		ps.Range(func(name string, ts *tensor.Tensor) {
			m := adam.m[i].Get(name)
			v := adam.v[i].Get(name)

			dm := ts.Grad().Sub(m).Scale(1 - adam.beta1)
			dv := ts.Grad().Pow(2).Sub(v).Scale(1 - adam.beta2)
			m.AddValue(dm.Value())
			v.AddValue(dv.Value())

			m = m.Scale(1 / (1 - math.Pow(adam.beta1, t)))
			v = v.Scale(1 / (1 - math.Pow(adam.beta2, t)))

			rows, cols := v.Dims()
			a := m.Scale(-adam.lr)
			b := v.Pow(0.5).Add(tensor.Numbers(rows, cols, adam.epsilon))
			ret[i].Set(name, a.DivElem(b))
		})
	}
	return ret
}

func (adam *Adam) Save() *pb.Optimizer {
	ret := adam.base.Save()
	ret.Params = make(map[string]float64)
	ret.Params["beta1"] = adam.beta1
	ret.Params["beta2"] = adam.beta2
	ret.Params["epsilon"] = adam.epsilon
	return ret
}
