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

// An AllRunner is a non-differentiable sequence function.
type AllRunner interface {
	RunAll(seqs [][]linalg.Vector) [][]linalg.Vector
}

// A Model is a seqtasks.Model which uses an rnn.SeqFunc
// and a cost function.
type Model struct {
	SeqFunc rnn.SeqFunc
	Learner sgd.Learner
	Runner  AllRunner
	Cost    neuralnet.CostFunc

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

func NewBlockModel(inSize, hiddenSize, hiddenCount, outHiddenSize, outCount int,
	blockMaker func(inSize, hiddenSize int) rnn.Block) *Model {
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
			res = append(res, blockMaker(inSize, hiddenSize))
		} else {
			res = append(res, blockMaker(hiddenSize, hiddenSize))
		}
	}
	res = append(res, outBlock)
	return &Model{
		SeqFunc:       &rnn.BlockSeqFunc{Block: res},
		Learner:       res,
		Runner:        &rnn.Runner{Block: res},
		Cost:          &neuralnet.SigmoidCECost{},
		OutActivation: &neuralnet.Sigmoid{},
	}
}

// UseSoftmax switches the output cost and activation
// to use softmax + cross entropy.
// It returns s for convenience.
func (s *Model) UseSoftmax() *Model {
	s.Cost = &neuralnet.DotCost{}
	s.OutActivation = &neuralnet.LogSoftmaxLayer{}
	return s
}

// Train runs one epoch of SGD on the entire sample set.
func (s *Model) Train(samples sgd.SampleSet) {
	if s.gradienter == nil {
		s.gradienter = &sgd.Adam{
			Gradienter: &seqtoseq.SeqFuncGradienter{
				SeqFunc:  s.SeqFunc,
				Learner:  s.SeqFunc.(sgd.Learner),
				CostFunc: s.Cost,
			},
		}
	}
	sgd.SGD(s.gradienter, samples, StepSize, 1, BatchSize)
}

// Run applies the RNN to all the inputs in batch,
// then applies the optional activation function.
func (s *Model) Run(inputs [][]linalg.Vector) [][]linalg.Vector {
	out := s.Runner.RunAll(inputs)
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
