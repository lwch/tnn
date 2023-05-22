package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

func TestDense(t *testing.T) {
	input := tensor.New([]float64{1, 2, 3, 4}, 2, 2)
	layer := NewDense(3, initializer.NewXavierUniform(1))
	output := layer.Forward(input, nil, false)
	fmt.Println(mat.Formatted(output.Value()))
	output.Backward(tensor.Ones(output.Dims()))
	layer.Params().Range(func(name string, value *tensor.Tensor) {
		fmt.Printf("===== param [%s] grads =====\n", name)
		fmt.Println(mat.Formatted(value.Grad().Value()))
	})
}
