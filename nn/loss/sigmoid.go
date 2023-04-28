package loss

import (
	"math"

	"github.com/lwch/tnn/internal/pb"
	"gonum.org/v1/gonum/mat"
)

type Sigmoid struct{}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (*Sigmoid) Name() string {
	return "sigmoid"
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (*Sigmoid) Loss(predict, targets mat.Matrix) float64 {
	rows, cols := predict.Dims()
	one := make([]float64, rows*cols)
	for i := 0; i < rows*cols; i++ {
		one[i] = 1
	}
	var a mat.Dense
	a.Sub(mat.NewDense(rows, cols, one), targets)
	a.MulElem(&a, predict)
	a.Add(&a, mat.NewDense(rows, cols, one))
	var b mat.Dense
	b.Apply(func(i, j int, v float64) float64 {
		return math.Log(sigmoid(v))
	}, predict)
	a.MulElem(&a, &b)
	return mat.Sum(&a) / float64(rows)
}

func (*Sigmoid) Grad(predict, targets mat.Matrix) mat.Matrix {
	var grad mat.Dense
	grad.Apply(func(i, j int, v float64) float64 {
		return sigmoid(v)
	}, predict)
	grad.Sub(&grad, targets)
	rows, _ := predict.Dims()
	grad.Scale(1/float64(rows), &grad)
	return &grad
}

func (loss *Sigmoid) Save() *pb.Loss {
	return &pb.Loss{Name: "sigmoid"}
}
