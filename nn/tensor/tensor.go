package tensor

import "gonum.org/v1/gonum/mat"

type Tensor struct {
	shapes []int
	data   *mat.VecDense
}

func New(data []float64, shapes ...int) *Tensor {
	if len(shapes) == 0 {
		panic("invalid shapes")
	}
	size := shapes[0]
	for i := 1; i < len(shapes); i++ {
		size *= shapes[i]
	}
	return &Tensor{
		shapes: shapes,
		data:   mat.NewVecDense(size, data),
	}
}

func FromVec(v *mat.VecDense, shapes ...int) *Tensor {
	if len(shapes) == 0 {
		panic("invalid shapes")
	}
	size := shapes[0]
	for i := 1; i < len(shapes); i++ {
		size *= shapes[i]
	}
	if v.Len() != size {
		panic("invalid size")
	}
	return &Tensor{
		shapes: shapes,
		data:   v,
	}
}

func (t *Tensor) Dims() int {
	return len(t.shapes)
}

func (t *Tensor) IsSameShape(t2 *Tensor) bool {
	if t.Dims() != t2.Dims() {
		return false
	}
	for i := 0; i < t.Dims(); i++ {
		if t.Shape(i) != t2.Shape(i) {
			return false
		}
	}
	return true
}

func (t *Tensor) Shapes() []int {
	return t.shapes
}

func (t *Tensor) Shape(i int) int {
	return t.shapes[i]
}

func (t *Tensor) Value() *mat.VecDense {
	return t.data
}

func (t *Tensor) Clone() *Tensor {
	var data mat.VecDense
	data.CloneFromVec(t.data)
	return FromVec(&data, t.shapes...)
}

func Ones(shapes ...int) *Tensor {
	size := shapes[0]
	for i := 1; i < len(shapes); i++ {
		size *= shapes[i]
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = 1
	}
	return New(data, shapes...)
}
