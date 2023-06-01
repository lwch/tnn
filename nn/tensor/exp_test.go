package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestExp(t *testing.T) {
	x1 := New([]float32{1, 2, 3, 4}, 2, 2)
	y := x1.Exp()
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x1.Grad().Value()))
}
