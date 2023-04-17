package main

import (
	"fmt"
	"os"

	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/model"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("missing model file")
		return
	}
	var m model.Model
	runtime.Assert(m.Load(os.Args[1]))
	m.Print()
}
