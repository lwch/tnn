package main

import (
	"fmt"
	"time"
	"tnn/internal/initializer"
	"tnn/internal/nn/layer"
	"tnn/internal/nn/layer/activation"
	"tnn/internal/nn/loss"
	"tnn/internal/nn/model"
	"tnn/internal/nn/net"
	"tnn/internal/nn/optimizer"

	"gonum.org/v1/gonum/mat"
)

const lr = 0.01
const epoch = 100000

func main() {
	input := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	output := mat.NewDense(4, 1, []float64{
		0,
		1,
		1,
		0,
	})

	const hidden = 16

	initializer := initializer.NewNormal(1, 0.5)

	var net net.Net
	net.Set(
		layer.NewDense(hidden, initializer),
		activation.NewSigmoid(),
		layer.NewDense(1, initializer),
	)
	loss := loss.NewMSE()
	optimizer := optimizer.NewSGD(lr, 0)
	// optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	m := model.New(&net, loss, optimizer)
	begin := time.Now()
	for i := 0; i < epoch; i++ {
		m.Train(input, output)
		if i%1000 == 0 {
			loss := m.Loss(input, output)
			fmt.Printf("Epoch: %d, Loss: %.05f\n", i, loss)
		}
	}
	fmt.Printf("train cost: %s\n", time.Since(begin).String())
	fmt.Println("predict:")
	pred := m.Predict(input)
	for i := 0; i < 4; i++ {
		fmt.Printf("%d xor %d: %f\n",
			int(input.At(i, 0)), int(input.At(i, 1)),
			pred.At(i, 0))
	}
}
