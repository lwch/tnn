package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestMaxAxis(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	y := x.MaxAxis(1)
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(y.Value()))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}
