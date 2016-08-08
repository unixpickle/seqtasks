package seqtasks

import (
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// MatchMultiTask requires the model to learn to close multiple
// kinds of open tags properly.
// For example, imagine the sequences of "[", "]", "{", and "}"
// where the brackets and curly-brackets needed to close in a
// matching fashion, e.g. "[{[[{}]]}]"
type MatchMultiTask struct {
	// TypeCount is the number of tag types.
	// If this is 1, this task is similar to MatchOpenTask.
	TypeCount int

	// MinLen is the minimum length of the input string of tags.
	MinLen int

	// MaxLen is the maximum length of the input string of tags.
	MaxLen int

	// CloseProb is the probability that the next symbol in the
	// input string will close the previous tag rather than
	// opening another tag.
	// The higher CloseProb, the lest nested tags input strings
	// are likely to have.
	CloseProb float64
}

// InputSize returns the number of input symbols into the model,
// which varies with m.TypeCount.
func (m *MatchMultiTask) InputSize() int {
	return 2*m.TypeCount + 1
}

// OutputSize returns the number of output symbols in the model,
// which varies with m.TypeCount.
func (m *MatchMultiTask) OutputSize() int {
	return m.TypeCount + 1
}

// NewSamples creates a set of samples.
func (m *MatchMultiTask) NewSamples(n int) sgd.SampleSet {
	var res sgd.SliceSampleSet
	zeroIn := make(linalg.Vector, m.InputSize())
	zeroOut := make(linalg.Vector, m.OutputSize())
	inDelimiter := make(linalg.Vector, m.InputSize())
	inDelimiter[len(inDelimiter)-1] = 1
	outDelimiter := make(linalg.Vector, m.OutputSize())
	outDelimiter[len(outDelimiter)-1] = 1
	for i := 0; i < n; i++ {
		var sample seqtoseq.Sample
		var symbolStack []int
		sampleLen := rand.Intn(m.MaxLen-m.MinLen+1) + m.MinLen
		for j := 0; j < sampleLen; j++ {
			sample.Outputs = append(sample.Outputs, zeroOut)
			if len(symbolStack) == 0 || rand.Float64() > m.CloseProb {
				newSym := rand.Intn(m.TypeCount)
				inVec := make(linalg.Vector, m.InputSize())
				inVec[newSym] = 1
				sample.Inputs = append(sample.Inputs, inVec)
			} else {
				lastSym := symbolStack[len(symbolStack)-1]
				symbolStack = symbolStack[:len(symbolStack)-1]
				inVec := make(linalg.Vector, m.InputSize())
				inVec[lastSym+m.TypeCount] = 1
				sample.Inputs = append(sample.Inputs, inVec)
			}
		}
		sample.Outputs = append(sample.Outputs, zeroOut)
		sample.Inputs = append(sample.Inputs, inDelimiter)
		for j := len(symbolStack) - 1; j >= 0; j-- {
			symbol := symbolStack[j]
			outVec := make(linalg.Vector, m.OutputSize())
			outVec[symbol] = 1
			sample.Outputs = append(sample.Outputs, outVec)
			sample.Inputs = append(sample.Inputs, zeroIn)
		}
		sample.Outputs = append(sample.Outputs, outDelimiter)
		sample.Inputs = append(sample.Inputs, zeroIn)
		res = append(res, sample)
	}
	return res
}

func (m *MatchMultiTask) Score(model Model, batchSize, batchCount int) float64 {
	return roundedBinaryTailScore(m, model, batchSize, batchCount, func(s []linalg.Vector) int {
		for i, x := range s {
			if x[len(x)-1] == 1 {
				return i + 1
			}
		}
		panic("no tail found")
	})
}
