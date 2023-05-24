package tensor

type Operator interface {
	Forward() *Tensor
	Backward(grad *Tensor)
	Dims() (int, int)
	ZeroGrad()
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	op := &add{
		a: t,
		b: t2,
	}
	return &Tensor{op: op}
}

func (t *Tensor) AddVector(t2 *Tensor) *Tensor {
	if rows, _ := t2.Dims(); rows != 1 {
		panic("t2 must be a vector")
	}
	op := &addVector{
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

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	op := &mul{
		a: t,
		b: t2,
	}
	return &Tensor{op: op}
}

func (t *Tensor) MulElem(t2 *Tensor) *Tensor {
	op := &mulElem{
		a: t,
		b: t2,
	}
	return &Tensor{op: op}
}

func (t *Tensor) DivElem(t2 *Tensor) *Tensor {
	op := &divElem{
		a: t,
		b: t2,
	}
	return &Tensor{op: op}
}

func (t *Tensor) Exp() *Tensor {
	op := &exp{
		a: t,
	}
	return &Tensor{op: op}
}

func (t *Tensor) Log() *Tensor {
	op := &log{
		a: t,
	}
	return &Tensor{op: op}
}

func (t *Tensor) Inv() *Tensor {
	op := &inv{
		a: t,
	}
	return &Tensor{op: op}
}

func (t *Tensor) Sum() *Tensor {
	op := &sum{
		a: t,
	}
	return &Tensor{op: op}
}

func (t *Tensor) Pow(n float64) *Tensor {
	op := &pow{
		a: t,
		b: n,
	}
	return &Tensor{op: op}
}

func (t *Tensor) Tanh() *Tensor {
	op := &tanh{
		a: t,
	}
	return &Tensor{op: op}
}

func (t *Tensor) Slice(topY, bottomY, leftX, rightX int) *Tensor {
	op := &slice{
		a:       t,
		topY:    topY,
		bottomY: bottomY,
		leftX:   leftX,
		rightX:  rightX,
	}
	return &Tensor{op: op}
}

func (t *Tensor) T() *Tensor {
	op := &transpose{
		a: t,
	}
	return &Tensor{op: op}
}
