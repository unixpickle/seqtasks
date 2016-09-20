package seqtasks

import (
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// RandomRecallTask measures a model's ability to learn
// long-term dependencies by feeding it input sequences
// with one input specially marked for later recall.
type RandomRecallTask struct {
	// Bits indicates the number of bits for each piece
	// of data in the sequence.
	Bits int

	// SeqLen specifies the size of the sequences.
	SeqLen int
}

// InputSize returns the size of each input vector.
func (r *RandomRecallTask) InputSize() int {
	return r.Bits + 2
}

// OutputSize returns the size of each output vector.
func (r *RandomRecallTask) OutputSize() int {
	return r.Bits
}

// NewSamples generates random sequences for the task.
func (r *RandomRecallTask) NewSamples(n int) sgd.SampleSet {
	var res sgd.SliceSampleSet
	for i := 0; i < n; i++ {
		var sample seqtoseq.Sample
		sample.Inputs = make([]linalg.Vector, r.SeqLen+1)
		sample.Outputs = make([]linalg.Vector, r.SeqLen+1)
		for i := 0; i < r.SeqLen; i++ {
			sample.Inputs[i] = make(linalg.Vector, r.InputSize())
			for j := 0; j < r.Bits; j++ {
				sample.Inputs[i][j] = float64(rand.Intn(2))
			}
			sample.Outputs[i] = make(linalg.Vector, r.OutputSize())
		}
		rememberIdx := rand.Intn(r.SeqLen)
		sample.Inputs[rememberIdx][r.Bits] = 1
		sample.Inputs[r.SeqLen] = make(linalg.Vector, r.InputSize())
		sample.Inputs[r.SeqLen][r.Bits+1] = 1
		sample.Outputs[r.SeqLen] = sample.Inputs[rememberIdx][:r.Bits]
		res = append(res, sample)
	}
	return res
}

// Score measures the fraction of output bits the model
// predicts correctly.
func (r *RandomRecallTask) Score(model Model, batchSize, batchCount int) float64 {
	return roundedBinaryTailScore(r, model, batchSize, batchCount, func(s []linalg.Vector) int {
		for i, x := range s {
			if x[r.Bits+1] == 1 {
				return i
			}
		}
		panic("no tail found")
	})
}
