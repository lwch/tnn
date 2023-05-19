package tensor

type Operator interface {
	Forward() *Tensor
	Backward(grad *Tensor) []*Tensor
}

func (t *Tensor) Add(t2 *Tensor) Operator {
	if !t.IsSameShape(t2) {
		panic("invalid shape")
	}
	return &add{
		a: t,
		b: t2,
	}
}
