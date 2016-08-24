package main

import (
	"github.com/unixpickle/autofunc"
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

// NewBlock creates an rnn.Block which uses an LSTM to
// produce outputs given the inputs.
// The resulting network will output a scaler, so adding
// an activation function may be required.
func NewBlock(inSize, hiddenSize, outHiddenSize, outCount int) rnn.StackedBlock {
	return NewDeepBlock(inSize, hiddenSize, 1, outHiddenSize, outCount)
}

// NewDeepBlock is like NewBlock, except that it allows
// the caller to specify the number of LSTM layers.
func NewDeepBlock(inSize, hiddenSize, hiddenCount, outHiddenSize, outCount int) rnn.StackedBlock {
	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  hiddenSize,
			OutputCount: outHiddenSize,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  outHiddenSize,
			OutputCount: outCount,
		},
	}
	outNet.Randomize()
	outBlock := rnn.NewNetworkBlock(outNet, 0)
	var res rnn.StackedBlock
	for i := 0; i < hiddenCount; i++ {
		if i == 0 {
			res = append(res, rnn.NewLSTM(inSize, hiddenSize))
		} else {
			res = append(res, rnn.NewLSTM(hiddenSize, hiddenSize))
		}
	}
	res = append(res, outBlock)
	return res
}

// A BlockModel is a seqtasks.Model which uses an rnn.Block
// with the given cost function.
type BlockModel struct {
	Block rnn.StackedBlock
	Cost  neuralnet.CostFunc

	// OutActivation is used to format outputs for Run.
	// This might be necessary in conjunction with a
	// cost function like neuralnet.SigmoidCECost, since
	// the network won't apply sigmoid automatically.
	//
	// If this is nil, no activation will be applied for
	// calls to Run.
	OutActivation autofunc.Func

	gradienter sgd.Gradienter
}

// Train runs one epoch of SGD on the entire sample set.
func (b *BlockModel) Train(s sgd.SampleSet) {
	if b.gradienter == nil {
		b.gradienter = &sgd.Adam{
			Gradienter: &seqtoseq.BPTT{
				Block:    b.Block,
				Learner:  b.Block,
				CostFunc: b.Cost,
			},
		}
	}
	sgd.SGD(b.gradienter, s, StepSize, 1, BatchSize)
}

// Run applies the RNN to all the inputs in batch, then
// applies the optional activation function.
func (b *BlockModel) Run(inputs [][]linalg.Vector) [][]linalg.Vector {
	runner := &rnn.Runner{Block: b.Block}
	out := runner.RunAll(inputs)
	if b.OutActivation != nil {
		for _, s := range out {
			for i, x := range s {
				in := &autofunc.Variable{Vector: x}
				s[i] = b.OutActivation.Apply(in).Output()
			}
		}
	}
	return out
}
