package main

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/neuralstruct"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

const (
	StepSize  = 0.001
	BatchSize = 1
)

// NewSeqFunc creates a neuralstruct.SeqFunc with an
// underlying LSTM.
// The LSTM has no output activation function by default.
func NewSeqFunc(s neuralstruct.RStruct, inSize, hiddenSize,
	outHiddenSize, outCount int) *neuralstruct.SeqFunc {
	return NewDeepSeqFunc(s, inSize, hiddenSize, 1, outHiddenSize, outCount)
}

// NewDeepSeqFunc is like NewSeqFunc, except that it
// allows the caller to specify the LSTM layer count.
func NewDeepSeqFunc(s neuralstruct.RStruct, inSize, hiddenSize, hiddenCount, outHiddenSize,
	outCount int) *neuralstruct.SeqFunc {
	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  hiddenSize,
			OutputCount: outHiddenSize,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  outHiddenSize,
			OutputCount: outCount + s.ControlSize(),
		},
	}
	outNet.Randomize()
	outBlock := rnn.NewNetworkBlock(outNet, 0)
	var resBlock rnn.StackedBlock
	for i := 0; i < hiddenCount; i++ {
		if i == 0 {
			resBlock = append(resBlock, rnn.NewLSTM(inSize+s.DataSize(), hiddenSize))
		} else {
			resBlock = append(resBlock, rnn.NewLSTM(hiddenSize, hiddenSize))
		}
	}
	resBlock = append(resBlock, outBlock)
	return &neuralstruct.SeqFunc{
		Block:  resBlock,
		Struct: s,
	}
}

// A SeqFuncModel is a seqtasks.Model which uses a
// neuralstruct.SeqFunc and a cost function.
type SeqFuncModel struct {
	SeqFunc *neuralstruct.SeqFunc
	Cost    neuralnet.CostFunc

	// OutActivation is used to format outputs for Run.
	// This might be necessary in conjunction with a
	// cost function like neuralnet.SigmoidCECost.
	// See ../lstm/model.go for more.
	OutActivation autofunc.Func
}

// Train runs one epoch of SGD on the entire sample set.
func (s *SeqFuncModel) Train(samples sgd.SampleSet) {
	gradienter := &sgd.Adam{
		Gradienter: &seqtoseq.SeqFuncGradienter{
			SeqFunc:  s.SeqFunc,
			Learner:  s.SeqFunc,
			CostFunc: s.Cost,
		},
	}
	sgd.SGD(gradienter, samples, StepSize, 1, BatchSize)
}

// Run applies the SeqFunc to all the inputs in batch,
// then applies the optional activation function.
func (s *SeqFuncModel) Run(inputs [][]linalg.Vector) [][]linalg.Vector {
	runner := &neuralstruct.Runner{
		Block:  s.SeqFunc.Block,
		Struct: s.SeqFunc.Struct,
	}
	out := runner.RunAll(inputs)
	if s.OutActivation != nil {
		for _, seq := range out {
			for i, x := range seq {
				in := &autofunc.Variable{Vector: x}
				seq[i] = s.OutActivation.Apply(in).Output()
			}
		}
	}
	return out
}
