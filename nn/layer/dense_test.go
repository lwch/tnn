package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/tensor"
)

func TestDense(t *testing.T) {
	input := tensor.New([]float32{1, 2, 3, 4}, 2, 2)
	layer := NewDense(3, initializer.NewXavierUniform(1))
	output := layer.Forward(input, false)
	fmt.Println(mat32.Formatted(output.Value()))
	output.Backward(tensor.Ones(output.Dims()))
	layer.Params().Range(func(name string, value *tensor.Tensor) {
		fmt.Printf("===== param [%s] grads =====\n", name)
		fmt.Println(mat32.Formatted(value.Grad().Value()))
	})
}
