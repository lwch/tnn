package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gonum/mat32"
)

func TestSlice(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	y := x.Slice(0, 2, 0, 2)
	fmt.Println(mat32.Formatted(y.Value()))
	y.Backward(Ones(y.Dims()))
	fmt.Println(mat32.Formatted(x.Grad().Value()))
}
