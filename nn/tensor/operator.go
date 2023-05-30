package tensor

import "gonum.org/v1/gonum/mat"

type Operator interface {
	f() *mat.Dense
	df(grad *Tensor)
	needGrad() bool
	ZeroGrad()
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	op := &add{
		a: t,
		b: t2,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Sub(t2 *Tensor) *Tensor {
	op := &sub{
		a: t,
		b: t2,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Scale(n float64) *Tensor {
	op := &scale{
		a: n,
		b: t,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	op := &mul{
		a: t,
		b: t2,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) MulElem(t2 *Tensor) *Tensor {
	op := &mulElem{
		a: t,
		b: t2,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) DivElem(t2 *Tensor) *Tensor {
	op := &divElem{
		a: t,
		b: t2,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Exp() *Tensor {
	op := &exp{
		a: t,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Log() *Tensor {
	op := &log{
		a: t,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Inv() *Tensor {
	op := &inv{
		a: t,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Sum() *Tensor {
	op := &sum{
		a: t,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) SumAxis(axis int) *Tensor {
	op := &sumAxis{
		a:    t,
		axis: axis,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Pow(n float64) *Tensor {
	op := &pow{
		a: t,
		b: n,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Tanh() *Tensor {
	op := &tanh{
		a: t,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Slice(topY, bottomY, leftX, rightX int) *Tensor {
	op := &slice{
		a:       t,
		topY:    topY,
		bottomY: bottomY,
		leftX:   leftX,
		rightX:  rightX,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) T() *Tensor {
	op := &transpose{
		a: t,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Sqrt() *Tensor {
	op := &sqrt{
		a: t,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) MaxAxis(axis int) *Tensor {
	op := &maxAxis{
		a:    t,
		axis: axis,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Stack(t2 *Tensor) *Tensor {
	op := &stack{
		a: t,
		b: t2,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) MeanAxis(axis int) *Tensor {
	op := &meanAxis{
		a:    t,
		axis: axis,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) VarianceAxis(axis int, unbiased bool) *Tensor {
	op := &varianceAxis{
		a:        t,
		axis:     axis,
		unbiased: unbiased,
	}
	return &Tensor{op: op, data: op.f()}
}

func (t *Tensor) Softmax(axis int) *Tensor {
	op := &softmax{
		a:    t,
		axis: axis,
	}
	return &Tensor{op: op, data: op.f()}
}
