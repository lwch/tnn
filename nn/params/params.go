package params

import (
	"sync"

	"github.com/lwch/tnn/nn/tensor"
)

type Params struct {
	m    sync.RWMutex
	data map[string]*tensor.Tensor
}

func New() *Params {
	return &Params{
		data: make(map[string]*tensor.Tensor),
	}
}

func (p *Params) Set(name string, data *tensor.Tensor) {
	p.m.Lock()
	defer p.m.Unlock()
	p.data[name] = data
}

func (p *Params) Get(name string) *tensor.Tensor {
	p.m.RLock()
	defer p.m.RUnlock()
	return p.data[name]
}

func (params *Params) Range(fn func(name string, dense *tensor.Tensor)) {
	if params == nil {
		return
	}
	data := make(map[string]*tensor.Tensor, len(params.data))
	params.m.RLock()
	for k, v := range params.data {
		data[k] = v
	}
	params.m.RUnlock()
	for name, dense := range data {
		fn(name, dense)
	}
}
