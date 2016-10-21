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
	"ffstruct": neuralstruct.RAggregate{
		&neuralstruct.Queue{VectorSize: 6},
		&neuralstruct.Queue{VectorSize: 6},
		&neuralstruct.Queue{VectorSize: 6},
		&neuralstruct.Stack{VectorSize: 6, NoReplace: true},
		&neuralstruct.Stack{VectorSize: 6, NoReplace: true},
		&neuralstruct.Stack{VectorSize: 6, NoReplace: true},
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
	seqFunc := &rnn.BlockSeqFunc{
		B: &neuralstruct.Block{
			Block:  resBlock,
			Struct: s,
		},
	}
	return &Model{
		SeqFunc:       seqFunc,
		Learner:       seqFunc,
		Runner:        &rnn.Runner{Block: seqFunc.B},
		Cost:          &neuralnet.SigmoidCECost{},
		OutActivation: &neuralnet.Sigmoid{},
	}
}

// NewStructFeedforward creates a struct RNN with a
// feed-forward neural controller.
func NewStructFeedforward(s neuralstruct.RStruct, in, out int, hidden ...int) *Model {
	var network neuralnet.Network
	for i, size := range hidden {
		lastSize := in + s.DataSize()
		if i > 0 {
			lastSize = hidden[i-1]
		}
		network = append(network, &neuralnet.DenseLayer{
			InputCount:  lastSize,
			OutputCount: size,
		})
		network = append(network, &neuralnet.HyperbolicTangent{})
	}
	lastHiddenSize := in
	if len(hidden) > 0 {
		lastHiddenSize = hidden[len(hidden)-1]
	}
	network = append(network, &neuralnet.DenseLayer{
		InputCount:  lastHiddenSize,
		OutputCount: out + s.ControlSize(),
	})
	network = append(network, structDataActivation(s))
	network.Randomize()
	seqFunc := &rnn.BlockSeqFunc{
		B: &neuralstruct.Block{
			Block:  rnn.NewNetworkBlock(network, 0),
			Struct: s,
		},
	}
	return &Model{
		SeqFunc:       seqFunc,
		Learner:       seqFunc,
		Runner:        &rnn.Runner{Block: seqFunc.B},
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
