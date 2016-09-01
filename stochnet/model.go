package main

import (
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/stochnet"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// A Model is a seqtasks.Model which uses a stochastic
// RNN to learn sequence tasks.
type Model struct {
	StateSize   int
	StateLayer  stochnet.Layer
	OutputLayer stochnet.Layer
}

// NewModel creates a Model with the given dimensions.
func NewModel(in, state, outHidden, out int) *Model {
	return &Model{
		StateSize:  state,
		StateLayer: stochnet.NewDense(in+state, state).Randomize(),
		OutputLayer: stochnet.Network{
			stochnet.NewDense(state, outHidden).Randomize(),
			stochnet.NewDense(outHidden, out).Randomize(),
		},
	}
}

// Train runs one epoch of SGD on the entire sample set.
func (s *Model) Train(samples sgd.SampleSet) {
	indices := rand.Perm(samples.Len())
	for _, i := range indices {
		sample := samples.GetSample(i).(seqtoseq.Sample)
		state := stochnet.ConstBoolVec(make([]bool, s.StateSize))
		for t, in := range sample.Inputs {
			out := floatsToBools(sample.Outputs[t]).Activations()
			joinedIn := stochnet.Concat(state, floatsToBools(in))
			state = s.StateLayer.Apply(joinedIn)
			output := s.OutputLayer.Apply(state)
			change := make([]bool, len(output.Activations()))
			for j, o := range output.Activations() {
				change[j] = (out[j] != o)
			}
			output.Learn(change)
		}
	}
}

// Run applies the RNN to all the inputs.
func (s *Model) Run(inputs [][]linalg.Vector) [][]linalg.Vector {
	var resSequences [][]linalg.Vector
	for _, sequence := range inputs {
		state := stochnet.ConstBoolVec(make([]bool, s.StateSize))
		var outSeq []linalg.Vector
		for _, in := range sequence {
			joinedIn := stochnet.Concat(state, floatsToBools(in))
			state = s.StateLayer.Apply(joinedIn)
			output := s.OutputLayer.Apply(state)
			outSeq = append(outSeq, boolsToFloats(output))
		}
		resSequences = append(resSequences, outSeq)
	}
	return resSequences
}

// floatsToBools converts a numerical vector to a boolean
// vector by rounding to 0 (false) or 1 (true).
func floatsToBools(v linalg.Vector) stochnet.BoolVec {
	bools := make([]bool, len(v))
	for i, f := range v {
		bools[i] = f >= 0.5
	}
	return stochnet.ConstBoolVec(bools)
}

// boolsToFloats is the inverse of floatsToBools.
func boolsToFloats(b stochnet.BoolVec) linalg.Vector {
	res := make(linalg.Vector, len(b.Activations()))
	for i, x := range b.Activations() {
		if x {
			res[i] = 1
		}
	}
	return res
}
