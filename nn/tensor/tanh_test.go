package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestTanh(t *testing.T) {
	x := New([]float32{1, 2, 3, 4}, 2, 2)
	y := x.Tanh()
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}
