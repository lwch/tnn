package loss

import (
	"math"
	"tnn/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type Huber struct {
	delta float64
}

func NewHuber(delta float64) *Huber {
	return &Huber{delta: delta}
}

func (*Huber) Name() string {
	return "huber"
}

func (loss *Huber) Loss(predict, targets mat.Matrix) float64 {
	var tmp mat.Dense
	var sum float64
	tmp.Apply(func(i, j int, v float64) float64 {
		dist := math.Abs(v - targets.At(i, j))
		mseMask := dist < loss.delta
		maeMask := !mseMask
		mse := 0.5 * math.Pow(v-targets.At(i, j), 2)
		mae := loss.delta*dist - 0.5*math.Pow(loss.delta, 2)
		if mseMask {
			sum += mse
		}
		if maeMask {
			sum += mae
		}
		return 0
	}, predict)
	rows, _ := predict.Dims()
	return sum / float64(rows)
}

func (loss *Huber) Grad(predict, targets mat.Matrix) mat.Matrix {
	rows, _ := predict.Dims()
	var grad mat.Dense
	grad.Apply(func(i, j int, v float64) float64 {
		err := v - targets.At(i, j)
		mseMask := math.Abs(err) < loss.delta
		maeMask := !mseMask
		mseGrad := err
		var maeGrad float64
		if err > 0 {
			maeGrad = loss.delta
		} else if err < 0 {
			maeGrad = -loss.delta
		}
		var grad float64
		if mseMask {
			grad += mseGrad
		}
		if maeMask {
			grad += maeGrad
		}
		return grad / float64(rows)
	}, predict)
	return &grad
}

func (loss *Huber) Save() *pb.Loss {
	return &pb.Loss{
		Name: "huber",
		Params: map[string]float64{
			"delta": loss.delta,
		},
	}
}
