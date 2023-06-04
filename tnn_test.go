package tnn

import (
	"fmt"
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestAdd(t *testing.T) {
	g := gorgonia.NewGraph()
	x := gorgonia.NodeFromAny(g, tensor.New(
		tensor.WithShape(2, 2, 3),
		tensor.WithBacking([]float32{
			1, 2, 3, 4, 5, 6,
			7, 8, 9, 10, 11, 12,
		})))
	// y := gorgonia.NodeFromAny(g, tensor.New(
	// 	tensor.WithShape(1, 2, 3),
	// 	tensor.WithBacking([]float64{
	// 		1, 2, 3, 4, 5, 6,
	// 	})))
	y := gorgonia.NodeFromAny(g, gorgonia.NewF32(2))
	z := gorgonia.Must(gorgonia.Mul(y, x))

	m := gorgonia.NewTapeMachine(g)
	defer m.Close()
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	fmt.Println(z.Value())
}
