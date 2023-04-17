package loss

import (
	"math"

	"github.com/lwch/tnn/nn/pb"
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

	rows, _ := data.Dims()
	max := make([]float64, rows)
	for i := 0; i < rows; i++ {
		max[i] = mat.Max(x.RowView(i))
	}
	var exps mat.Dense
	exps.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v - max[i])
	}, &x)
	sum := make([]float64, rows)
	for i := 0; i < rows; i++ {
		sum[i] = mat.Sum(exps.RowView(i))
	}
	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		return v - max[i] - math.Log(sum[i])
	}, data)
	return &ret
}

func softmax(data mat.Matrix, t float64) *mat.Dense {
	var x mat.Dense
	x.Scale(1/t, data)

	rows, _ := data.Dims()
	max := make([]float64, rows)
	for i := 0; i < rows; i++ {
		max[i] = mat.Max(x.RowView(i))
	}
	var exps mat.Dense
	exps.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v - max[i])
	}, &x)
	sum := make([]float64, rows)
	for i := 0; i < rows; i++ {
		sum[i] = mat.Sum(exps.RowView(i))
	}
	var ret mat.Dense
	ret.Apply(func(i, j int, v float64) float64 {
		return v / sum[i]
	}, &exps)
	return &ret
}

func (loss *Softmax) Loss(predict, targets mat.Matrix) float64 {
	softmax := logSoftmax(predict, loss.t)
	softmax.Apply(func(i, j int, v float64) float64 {
		return -v * targets.At(i, j)
	}, softmax)
	rows, _ := softmax.Dims()
	sum := mat.NewDense(rows, 1, nil)
	for i := 0; i < rows; i++ {
		sum.Set(i, 0, mat.Sum(softmax.RowView(i)))
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
