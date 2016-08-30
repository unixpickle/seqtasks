package main

import (
	"github.com/unixpickle/neuralstruct"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

var Structs = map[string]neuralstruct.RStruct{
	"stack": &neuralstruct.Stack{VectorSize: 10},
	"queue": &neuralstruct.Queue{VectorSize: 10},
	"multiqueue": neuralstruct.RAggregate{
		&neuralstruct.Queue{VectorSize: 4},
		&neuralstruct.Queue{VectorSize: 4},
		&neuralstruct.Queue{VectorSize: 4},
		&neuralstruct.Queue{VectorSize: 4},
		&neuralstruct.Queue{VectorSize: 4},
		&neuralstruct.Queue{VectorSize: 4},
	},
	"multistack": neuralstruct.RAggregate{
		&neuralstruct.Stack{VectorSize: 4, NoReplace: true},
		&neuralstruct.Stack{VectorSize: 4, NoReplace: true},
		&neuralstruct.Stack{VectorSize: 4, NoReplace: true},
		&neuralstruct.Stack{VectorSize: 4, NoReplace: true},
		&neuralstruct.Stack{VectorSize: 4, NoReplace: true},
		&neuralstruct.Stack{VectorSize: 4, NoReplace: true},
	},
}

// NewStructLSTM creates a new struct RNN with an LSTM
// controller, controlling the given structure.
func NewStructLSTM(s neuralstruct.RStruct, inSize, hiddenSize, hiddenCount, outHiddenSize,
	outCount int) *Model {
	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  hiddenSize,
			OutputCount: outHiddenSize,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  outHiddenSize,
			OutputCount: outCount + s.ControlSize(),
		},
		structDataActivation(s),
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
	seqFunc := &neuralstruct.SeqFunc{
		Block:  resBlock,
		Struct: s,
	}
	return &Model{
		SeqFunc:       seqFunc,
		Learner:       seqFunc,
		Runner:        &neuralstruct.Runner{Block: resBlock, Struct: s},
		Cost:          &neuralnet.SigmoidCECost{},
		OutActivation: &neuralnet.Sigmoid{},
	}
}

func structDataActivation(s neuralstruct.RStruct) neuralnet.Layer {
	ag, ok := s.(neuralstruct.RAggregate)
	if !ok {
		ag = neuralstruct.RAggregate{s}
	}
	var activation neuralstruct.PartialActivation
	var idx int
	for _, subStruct := range ag {
		r := neuralstruct.ComponentRange{
			Start: idx + subStruct.ControlSize() - subStruct.DataSize(),
			End:   idx + subStruct.ControlSize(),
		}
		activation.Ranges = append(activation.Ranges, r)
		activation.Activations = append(activation.Activations,
			&neuralnet.HyperbolicTangent{})
		idx = r.End
	}
	return &activation
}
