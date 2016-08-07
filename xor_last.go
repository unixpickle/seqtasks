package seqtasks

// An XORLastTask tests a model's ability to XOR the input bit
// with the previous input bit.
// All inputs and outputs are either 0 or 1.
type XORLastTask struct {
	// SeqLen is the length of test sequences.
	SeqLen int
}

// InputSize returns 1, since each timestep comes with one
// binary input.
func (x *XORLastTask) InputSize() int {
	return 1
}

// OutputSize returns 1, since each timestep requires one
// binary output from the model.
func (x *XORLastTask) OutputSize() int {
	return 1
}

// NewSamples creates a new set of training samples.
func (x *XORLastTask) NewSamples(count int) sgd.SampleSet {
	var set sgd.SliceSampleSet
	for i := 0; i < count; i++ {
		var seq seqtoseq.Sample
		for j := 0; j < x.SeqLen; j++ {
			input := float64(rand.Intn(2))
			seq.Inputs = append(seq.Inputs, []float64{input})
			if j == 0 {
				seq.Outputs = append(seq.Outputs, []float64{input})
			} else {
				last := seq.Inputs[j-1]
				xor1 := (last == 1 && input == 0) ||
					(last == 0 && input == 1)
				if xor1 {
					seq.Outputs = append(seq.Outputs, []float64{1})
				} else {
					seq.Outputs = append(seq.Outputs, []float64{0})
				}
			}
		}
		set = append(set, seq)
	}
	return set
}
