package tensor

type Operator interface {
	Forward() *Tensor
	Backward(grad *Tensor) []*Tensor
	Dims() (int, int)
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	op := &add{
		a: t,
		b: t2,
	}
	return &Tensor{op: op}
}

func (t *Tensor) Sub(t2 *Tensor) *Tensor {
	op := &sub{
		a: t,
		b: t2,
	}
	return &Tensor{op: op}
}

func (t *Tensor) Scale(n float64) *Tensor {
	op := &scale{
		a: n,
		b: t,
	}
	return &Tensor{op: op}
}
