package seqtasks

import (
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// AdditionTask requires the model to add two integers.
type AdditionTask struct {
	// MaxDigits is the maximum number of digits in a
	// number.
	MaxDigits int

	// Base is the base of the input numbers.
	// For instance, base 10 is decimal.
	Base int
}

// InputSize returns the number of input symbols, which
// varies with the base.
func (a *AdditionTask) InputSize() int {
	return a.Base + 1
}

// OutputSize returns the number of output symbols, which
// varies with the base.
func (a *AdditionTask) OutputSize() int {
	return a.Base
}

// NewSamples creates a set of samples.
func (a *AdditionTask) NewSamples(n int) sgd.SampleSet {
	var res sgd.SliceSampleSet
	for i := 0; i < n; i++ {
		var sample seqtoseq.Sample
		digitCount := rand.Intn(a.MaxDigits) + 1
		var leftOperand, rightOperand []int
		for j := 0; j < 2; j++ {
			for k := 0; k < digitCount; k++ {
				digit := rand.Intn(a.Base)
				if j == 0 {
					leftOperand = append(leftOperand, digit)
				} else {
					rightOperand = append(rightOperand, digit)
				}
				inVec := make(linalg.Vector, a.Base+1)
				inVec[digit] = 1
				sample.Inputs = append(sample.Inputs, inVec)
				sample.Outputs = append(sample.Outputs, make(linalg.Vector, a.Base))
			}
			delimiter := make(linalg.Vector, a.Base+1)
			delimiter[a.Base] = 1
			sample.Inputs = append(sample.Inputs, delimiter)
			sample.Outputs = append(sample.Outputs, make(linalg.Vector, a.Base))
		}

		// Needed to get carry flag.
		leftOperand = append(leftOperand, 0)
		rightOperand = append(rightOperand, 0)

		var carry int
		for j, x := range leftOperand {
			y := rightOperand[j]
			sum := (x + y + carry) % a.Base
			carry = (x + y + carry) / a.Base
			inVec := make(linalg.Vector, a.Base+1)
			outVec := make(linalg.Vector, a.Base)
			outVec[sum] = 1
			sample.Inputs = append(sample.Inputs, inVec)
			sample.Outputs = append(sample.Outputs, outVec)
		}
		res = append(res, sample)
	}
	return res
}

func (a *AdditionTask) Score(model Model, batchSize, batchCount int) float64 {
	return roundedBinaryTailScore(a, model, batchSize, batchCount, func(s []linalg.Vector) int {
		var seenBefore bool
		for i, x := range s {
			if x[len(x)-1] == 1 {
				if seenBefore {
					return i + 1
				}
				seenBefore = true
			}
		}
		panic("no tail found")
	})
}
