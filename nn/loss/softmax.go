package loss

import (
	"math"

	"github.com/lwch/tnn/internal/pb"
	"gonum.org/v1/gonum/mat"
)

type Softmax struct {
	t float64
}

func NewSoftmax(t float64) *Softmax {
	return &Softmax{t: t}
}

func (*Softmax) Name() string {
	return "softmax"
}

func logSoftmax(data mat.Matrix, t float64) *mat.Dense {
	var x mat.Dense
	x.Scale(1/t, data)

	rows, cols := data.Dims()
	max := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		value := mat.Max(x.RowView(i))
		for j := 0; j < cols; j++ {
			max.Set(i, j, value)
		}
	}
	var exps mat.Dense
	exps.Sub(&x, max)
	exps.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v)
	}, &exps)
	sum := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		value := mat.Sum(exps.RowView(i))
		value = math.Log(value)
		for j := 0; j < cols; j++ {
			sum.Set(i, j, value)
		}
	}
	var ret mat.Dense
	ret.Sub(&x, max)
	ret.Sub(&ret, sum)
	return &ret
}

func softmax(data mat.Matrix, t float64) *mat.Dense {
	var x mat.Dense
	x.Scale(1/t, data)

	rows, cols := data.Dims()
	max := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		value := mat.Max(x.RowView(i))
		for j := 0; j < cols; j++ {
			max.Set(i, j, value)
		}
	}
	var exps mat.Dense
	exps.Sub(&x, max)
	exps.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v)
	}, &exps)
	sum := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		value := 1 / mat.Sum(exps.RowView(i))
		for j := 0; j < cols; j++ {
			sum.Set(i, j, value)
		}
	}
	var ret mat.Dense
	ret.MulElem(&exps, sum)
	return &ret
}

func (loss *Softmax) Loss(predict, targets mat.Matrix) float64 {
	softmax := logSoftmax(predict, loss.t)
	softmax.MulElem(softmax, targets)
	softmax.Scale(-1, softmax)
	rows, _ := softmax.Dims()
	sum := mat.NewVecDense(rows, nil)
	for i := 0; i < rows; i++ {
		sum.SetVec(i, mat.Sum(softmax.RowView(i)))
	}
	return mat.Sum(sum) / float64(rows)
}

func (loss *Softmax) Grad(predict, targets mat.Matrix) mat.Matrix {
	softmax := softmax(predict, loss.t)
	var grad mat.Dense
	grad.Sub(softmax, targets)
	rows, _ := predict.Dims()
	grad.Scale(1/float64(rows), &grad)
	return &grad
}

func (loss *Softmax) Save() *pb.Loss {
	return &pb.Loss{
		Name: "softmax",
		Params: map[string]float64{
			"t": loss.t,
		},
	}
}
