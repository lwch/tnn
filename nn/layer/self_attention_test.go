package layer

import (
	"fmt"
	"math"
	"testing"

	m "github.com/lwch/tnn/internal/math"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

func TestSelfAttention(t *testing.T) {
	input := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	layer := NewSelfAttention(3, 1, 1, initializer.NewNumber(1))
	output := layer.Forward(input, false)
	fmt.Println(mat.Formatted(output.Value()))
	output.Backward(tensor.Ones(output.Dims()))
	layer.Params().Range(func(name string, value *tensor.Tensor) {
		fmt.Printf("===== param [%s] grads =====\n", name)
		fmt.Println(mat.Formatted(value.Grad().Value()))
	})
}

func TestSelfAttentionCustom(t *testing.T) {
	x := tensor.New([]float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, 2, 4)
	x.SetRequireGrad(true)
	y := tensor.New([]float64{
		1, 0, 0, 0,
		0, 0, 0, 0,
	}, 2, 4)
	y.SetRequireGrad(true)
	layer := NewSelfAttention(2, 2, 2, initializer.NewNumber(1))
	output := layer.(*SelfAttention).ForwardQKV(x, y, y, false, false)
	fmt.Println(mat.Formatted(output.Value(), mat.Squeeze()))
	output.Backward(tensor.Ones(output.Dims()))
	// layer.Params().Range(func(name string, value *tensor.Tensor) {
	// 	fmt.Printf("===== param [%s] grads =====\n", name)
	// 	fmt.Println(mat.Formatted(value.Grad().Value()))
	// })
	fmt.Println(mat.Formatted(x.Grad().Value(), mat.Squeeze()))
	fmt.Println(mat.Formatted(y.Grad().Value(), mat.Squeeze()))
}

func TestSelfAttentionRun(t *testing.T) {
	x := tensor.New([]float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, 3, 3)
	x.SetRequireGrad(true)
	y := tensor.New([]float64{
		1, 0, 0,
		0, 0, 0,
		0, 0, 0,
	}, 3, 3)
	y.SetRequireGrad(true)
	w := tensor.New([]float64{-0.0068, -0.5668, 0.3913,
		0.5622, -0.3745, -0.5589,
		0.4948, 0.5572, 0.5381}, 3, 3)
	w.SetRequireGrad(true)
	q := x.Mul(w)
	k := y.Mul(w)
	v := y.Mul(w)
	score1 := q.Mul(k.T())
	score1.SetRequireGrad(true)
	score2 := score1.Scale(1 / math.Sqrt(float64(3)))
	score2.SetRequireGrad(true)
	score3 := m.Softmax(score2, 1)
	score3.SetRequireGrad(true)
	z := score3.Mul(v)
	z.Backward(tensor.Ones(z.Dims()))
	fmt.Println(mat.Formatted(x.Grad().Value()))
	fmt.Println(mat.Formatted(y.Grad().Value()))
}
