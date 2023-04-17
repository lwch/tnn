package prof

import (
	"os"
	"runtime/pprof"
	"time"

	"github.com/lwch/runtime"
)

func CpuProfile(dir string, timeout time.Duration) {
	f, err := os.Create(dir)
	runtime.Assert(err)
	defer f.Close()
	runtime.Assert(pprof.StartCPUProfile(f))
	time.Sleep(timeout)
	pprof.StopCPUProfile()
	f.Close()
	os.Exit(0)
}
