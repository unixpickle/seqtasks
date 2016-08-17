package main

import (
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
)

const stackFlagCount = 4

func init() {
	var s StackActivation
	serializer.RegisterTypedDeserializer(s.SerializerType(), DeserializeStackActivation)
}

type StackActivation struct {
	VecSize int
}

func DeserializeStackActivation(d []byte) (*StackActivation, error) {
	obj, err := serializer.DeserializeWithType(d)
	if err != nil {
		return nil, err
	}
	intObj, ok := obj.(serializer.Int)
	if !ok {
		return nil, fmt.Errorf("invalid type: %T (expected serializer.Int)", intObj)
	}
	return &StackActivation{VecSize: int(intObj)}, nil
}

func (s *StackActivation) Apply(in autofunc.Result) autofunc.Result {
	if len(in.Output()) < stackFlagCount+s.VecSize {
		panic("activation input too small")
	}
	return autofunc.Pool(in, func(inPool autofunc.Result) autofunc.Result {
		flags := autofunc.Slice(inPool, 0, stackFlagCount)
		data := autofunc.Slice(inPool, stackFlagCount, s.VecSize+stackFlagCount)
		rest := autofunc.Slice(inPool, s.VecSize+stackFlagCount, len(inPool.Output()))
		sig := autofunc.Sigmoid{}
		return autofunc.Concat(flags, sig.Apply(data), rest)
	})
}

func (s *StackActivation) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	if len(in.Output()) < stackFlagCount+s.VecSize {
		panic("activation input too small")
	}
	return autofunc.PoolR(in, func(inPool autofunc.RResult) autofunc.RResult {
		flags := autofunc.SliceR(inPool, 0, stackFlagCount)
		data := autofunc.SliceR(inPool, stackFlagCount, s.VecSize+stackFlagCount)
		rest := autofunc.SliceR(inPool, s.VecSize+stackFlagCount, len(inPool.Output()))
		sig := autofunc.Sigmoid{}
		return autofunc.ConcatR(flags, sig.ApplyR(rv, data), rest)
	})
}

func (s *StackActivation) SerializerType() string {
	return "github.com/unixpickle/seqtasks/struct_lstm.StackActivation"
}

func (s *StackActivation) Serialize() ([]byte, error) {
	return serializer.SerializeWithType(serializer.Int(s.VecSize))
}
