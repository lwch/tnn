package net

import (
	"archive/tar"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"google.golang.org/protobuf/proto"
)

type loadFunc func(device consts.DeviceType, name string, params map[string]*tensor.Tensor, args map[string]float32) layer.Layer

var loadFuncs = map[string]loadFunc{
	"linear":     layer.LoadLinear,
	"dropout":    layer.LoadDropout,
	"conv1d":     layer.LoadConv1D,
	"conv2d":     layer.LoadConv2D,
	"maxpool1d":  layer.LoadMaxPool1D,
	"rnn":        layer.LoadRnn,
	"lstm":       layer.LoadLstm,
	"attention":  layer.LoadAttention,
	"attention1": layer.LoadAttention1,
	"layer_norm": layer.LoadLayerNorm,
	"flatten":    layer.LoadFlatten,
	"embedding":  layer.LoadEmbedding,
	"rezero":     layer.LoadReZero,
	// activation
	"sigmoid": activation.LoadSigmoid,
	"tanh":    activation.LoadTanh,
	"relu":    activation.LoadRelu,
	"gelu":    activation.LoadGelu,
}

type Net struct {
	layers []layer.Layer
	device consts.DeviceType
}

func New(device consts.DeviceType) *Net {
	return &Net{device: device}
}

func (n *Net) SetDevice(device consts.DeviceType) {
	n.device = device
}

func (n *Net) Add(layers ...layer.Layer) {
	n.layers = append(n.layers, layers...)
}

func (n *Net) Params() []*tensor.Tensor {
	var ret []*tensor.Tensor
	for _, l := range n.layers {
		for _, p := range l.Params() {
			ret = append(ret, p)
		}
	}
	return ret
}

func (n *Net) ParamCount() uint64 {
	var ret uint64
	for _, l := range n.layers {
		for _, p := range l.Params() {
			ret += uint64(p.ElemCount())
		}
	}
	return ret
}

func (n *Net) Save(dir string, compress bool) error {
	f, err := os.Create(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = n.WriteTo(f, compress)
	return err
}

func (n *Net) WriteTo(w io.Writer, compress bool) (int64, error) {
	if compress {
		var err error
		w, err = zstd.NewWriter(w)
		if err != nil {
			return 0, err
		}
		defer w.(*zstd.Encoder).Close()
	}
	tw := tar.NewWriter(w)
	defer tw.Close()
	var net pb.Net
	net.Layers = make([]*pb.Layer, len(n.layers))
	var params []*tensor.Tensor
	for i := 0; i < len(n.layers); i++ {
		net.Layers[i] = new(pb.Layer)
		net.Layers[i].Class = n.layers[i].Class()
		net.Layers[i].Name = n.layers[i].Name()
		net.Layers[i].Params = make(map[string]*pb.Param)
		for name, p := range n.layers[i].Params() {
			var param pb.Param
			param.Name = name
			shape := p.Shapes()
			param.Shape = make([]int64, len(shape))
			copy(param.Shape, shape)
			param.File = fmt.Sprintf("%d.bin", len(params))
			params = append(params, p)
			net.Layers[i].Params[name] = &param
		}
		net.Layers[i].Args = n.layers[i].Args()
	}
	data, err := proto.Marshal(&net)
	if err != nil {
		return 0, err
	}
	err = tw.WriteHeader(&tar.Header{
		Typeflag:   tar.TypeReg,
		Name:       "SPEC",
		Size:       int64(len(data)),
		Mode:       0644,
		ModTime:    time.Now(),
		AccessTime: time.Now(),
		ChangeTime: time.Now(),
	})
	if err != nil {
		return 0, err
	}
	cnt, err := io.Copy(tw, bytes.NewReader(data))
	if err != nil {
		return 0, err
	}
	for i := 0; i < len(params); i++ {
		err = tw.WriteHeader(&tar.Header{
			Typeflag:   tar.TypeReg,
			Name:       fmt.Sprintf("%d.bin", i),
			Size:       int64(params[i].ElemCount() * 4),
			Mode:       0644,
			ModTime:    time.Now(),
			AccessTime: time.Now(),
			ChangeTime: time.Now(),
		})
		if err != nil {
			return 0, err
		}
		err = binary.Write(tw, binary.BigEndian, params[i].Float32Value())
		if err != nil {
			return 0, err
		}
		cnt += int64(params[i].ElemCount() * 4)
	}
	return cnt, nil
}

func (n *Net) Load(dir string, compressed bool) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = n.ReadFrom(f, compressed)
	return err
}

func (n *Net) loadSpec(r io.Reader) (*pb.Net, error) {
	return nil, nil
}

func (n *Net) ReadFrom(r io.ReadSeeker, compressed bool) (int64, error) {
	return 0, nil
	// tr := tar.NewReader(r)
	// var net pb.Net
	// if err = proto.Unmarshal(data, &net); err != nil {
	// 	return 0, err
	// }
	// layers := net.GetLayers()
	// n.layers = make([]layer.Layer, len(layers))
	// for i := 0; i < len(layers); i++ {
	// 	class := layers[i].GetClass()
	// 	fn := loadFuncs[class]
	// 	if fn == nil {
	// 		panic("unsupported " + class + " layer")
	// 	}
	// 	name := layers[i].GetName()
	// 	n.layers[i] = fn(n.device, name, layers[i].GetParams(), layers[i].GetArgs())
	// }
	// return int64(len(data)), nil
}

func (n *Net) Layers() []layer.Layer {
	return n.layers
}
