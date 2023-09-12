package net

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/runtime"
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

func (n *Net) Save(dir string) error {
	f, err := os.Create(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = n.WriteTo(f)
	return err
}

func (n *Net) WriteTo(w io.Writer) (int64, error) {
	zw := zip.NewWriter(w)
	defer zw.Close()
	zw.RegisterCompressor(zip.Deflate, func(w io.Writer) (io.WriteCloser, error) {
		return zstd.NewWriter(w)
	})
	var net pb.Net
	net.Layers = make([]*pb.Layer, len(n.layers))
	type layer struct {
		names  []string
		params []*tensor.Tensor
	}
	var layers []layer
	for i := 0; i < len(n.layers); i++ {
		net.Layers[i] = new(pb.Layer)
		net.Layers[i].Class = n.layers[i].Class()
		net.Layers[i].Name = n.layers[i].Name()
		net.Layers[i].Params = make(map[string]*pb.Param)
		var layer layer
		for name, p := range n.layers[i].Params() {
			var param pb.Param
			param.Name = name
			shape := p.Shapes()
			param.Shape = make([]int64, len(shape))
			copy(param.Shape, shape)
			param.File = fmt.Sprintf("layer_%d_param_%s.bin", i, name)
			layer.names = append(layer.names, name)
			layer.params = append(layer.params, p)
			net.Layers[i].Params[name] = &param
		}
		layers = append(layers, layer)
		net.Layers[i].Args = n.layers[i].Args()
	}
	data, err := proto.Marshal(&net)
	if err != nil {
		return 0, err
	}
	f, err := zw.CreateHeader(&zip.FileHeader{
		Name:     "SPEC",
		Method:   zip.Deflate,
		Modified: time.Now(),
	})
	if err != nil {
		return 0, err
	}
	cnt, err := io.Copy(f, bytes.NewReader(data))
	if err != nil {
		return 0, err
	}
	for i, layer := range layers {
		for j := 0; j < len(layer.names); j++ {
			name := layer.names[j]
			param := layer.params[j]
			err = func() error {
				f, err := zw.CreateHeader(&zip.FileHeader{
					Name:     fmt.Sprintf("layer_%d_param_%s.bin", i, name),
					Method:   zip.Deflate,
					Modified: time.Now(),
				})
				if err != nil {
					return err
				}
				return binary.Write(f, binary.BigEndian, param.Float32Value())
			}()
			if err != nil {
				return 0, err
			}
			cnt += int64(param.ElemCount() * 4)
		}
	}
	return cnt, nil
}

func (n *Net) Load(dir string) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return err
	}
	_, err = n.ReadFrom(f, fi.Size())
	return err
}

func (n *Net) readSpec(r *zip.Reader) (*pb.Net, error) {
	f, err := r.Open("SPEC")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}
	var net pb.Net
	if err = proto.Unmarshal(data, &net); err != nil {
		return nil, err
	}
	return &net, nil
}

func (n *Net) loadParam(r *zip.Reader, file string, shape []int64) (*tensor.Tensor, error) {
	f, err := r.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	data := make([]float32, fi.Size()/4)
	if err = binary.Read(f, binary.BigEndian, data); err != nil {
		return nil, err
	}
	t := tensor.FromFloat32(nil, data,
		tensor.WithShapes(shape...),
		tensor.WithDevice(n.device))
	t.SetRequiresGrad(true)
	return t, nil
}

func (n *Net) ReadFrom(r io.ReaderAt, size int64) (int64, error) {
	zr, err := zip.NewReader(r, size)
	if err != nil {
		return 0, err
	}
	zr.RegisterDecompressor(zip.Deflate, func(r io.Reader) io.ReadCloser {
		zr, err := zstd.NewReader(r)
		runtime.Assert(err)
		return io.NopCloser(zr)
	})
	spec, err := n.readSpec(zr)
	if err != nil {
		return 0, err
	}
	layers := spec.GetLayers()
	n.layers = make([]layer.Layer, len(layers))
	for i := 0; i < len(layers); i++ {
		class := layers[i].GetClass()
		fn := loadFuncs[class]
		if fn == nil {
			panic("unsupported " + class + " layer")
		}
		params := make(map[string]*tensor.Tensor)
		for _, param := range layers[i].GetParams() {
			shape := param.GetShape()
			params[param.GetName()], err = n.loadParam(zr, param.GetFile(), shape)
			if err != nil {
				return 0, err
			}
		}
		n.layers[i] = fn(n.device, layers[i].GetName(), params, layers[i].GetArgs())
	}
	return size, nil
}

func (n *Net) Layers() []layer.Layer {
	return n.layers
}
