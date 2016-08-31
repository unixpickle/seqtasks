package main

import (
	"log"

	"github.com/unixpickle/seqtasks"
)

type Task struct {
	Name   string
	Task   seqtasks.Task
	Models map[string]seqtasks.Model

	MaxEpochs    int
	MaxScore     float64
	TrainingSize int
	TestingBatch int
	TestingCount int
}

func (t *Task) Run(modelName string) {
	model, ok := t.Models[modelName]
	if !ok {
		log.Printf("Model \"%s\" not implemented for task \"%s\"", modelName, t.Name)
		return
	}
	log.Printf("Running task \"%s\" with model \"%s\"", t.Name, modelName)
	samples := t.Task.NewSamples(t.TrainingSize)
	for i := 0; i < t.MaxEpochs; i++ {
		model.Train(samples)
		score := t.Task.Score(model, t.TestingBatch, t.TestingCount)
		log.Printf("epoch %d: score=%f", i, score)
		if score >= t.MaxScore {
			break
		}
	}
}

var Tasks = []*Task{
	{
		Name: "XOR last",
		Task: &seqtasks.XORLastTask{SeqLen: 50},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(1, 40, 1, 40, 1),
			"stack":      NewStructLSTM(Structs["stack"], 1, 40, 1, 40, 1),
			"queue":      NewStructLSTM(Structs["queue"], 1, 40, 1, 40, 1),
			"multistack": NewStructLSTM(Structs["multistack"], 1, 40, 1, 40, 1),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 1, 40, 1, 40, 1),
			"irnn":       NewIRNN(1, 40, 3, 40, 1, 1),
			"nprnn":      NewNPRNN(1, 40, 3, 40, 1),
		},
		MaxEpochs:    100,
		MaxScore:     1,
		TrainingSize: 100,
		TestingBatch: 10,
		TestingCount: 10,
	},
	{
		Name: "Repeat",
		Task: &seqtasks.RepeatTask{
			MinString: 2,
			MaxString: 5,
			MinGap:    0,
			MaxGap:    6,
		},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(3, 100, 1, 100, 1),
			"stack":      NewStructLSTM(Structs["stack"], 3, 40, 1, 40, 1),
			"queue":      NewStructLSTM(Structs["queue"], 3, 40, 1, 40, 1),
			"multistack": NewStructLSTM(Structs["multistack"], 3, 40, 1, 40, 1),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 3, 40, 1, 40, 1),
			"irnn":       NewIRNN(3, 100, 1, 100, 1, 0.1),
			"nprnn":      NewNPRNN(3, 40, 1, 40, 1),
		},
		MaxEpochs:    100,
		MaxScore:     1,
		TrainingSize: 300,
		TestingBatch: 10,
		TestingCount: 30,
	},
	{
		Name: "Match Open",
		Task: &seqtasks.MatchOpenTask{
			MinLen:  1,
			MaxLen:  15,
			MaxOpen: 6,
		},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(3, 40, 1, 40, 1),
			"stack":      NewStructLSTM(Structs["stack"], 3, 40, 1, 40, 1),
			"queue":      NewStructLSTM(Structs["queue"], 3, 40, 1, 40, 1),
			"multistack": NewStructLSTM(Structs["multistack"], 3, 40, 1, 40, 1),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 3, 40, 1, 40, 1),
			"irnn":       NewIRNN(3, 40, 3, 40, 1, 1),
			"nprnn":      NewNPRNN(3, 40, 3, 40, 1),
		},
		MaxEpochs:    50,
		MaxScore:     1,
		TrainingSize: 100,
		TestingBatch: 10,
		TestingCount: 30,
	},
	{
		Name: "Match Multi",
		Task: &seqtasks.MatchMultiTask{
			TypeCount: 4,
			MinLen:    1,
			MaxLen:    8,
			CloseProb: 0.3,
		},
		Models: map[string]seqtasks.Model{
			"lstm":       NewLSTM(4*2+1, 100, 2, 100, 4+1),
			"stack":      NewStructLSTM(Structs["stack"], 4*2+1, 40, 1, 40, 4+1),
			"queue":      NewStructLSTM(Structs["queue"], 4*2+1, 40, 1, 40, 4+1),
			"multistack": NewStructLSTM(Structs["multistack"], 4*2+1, 40, 1, 40, 4+1),
			"multiqueue": NewStructLSTM(Structs["multiqueue"], 4*2+1, 40, 1, 40, 4+1),
			"irnn":       NewIRNN(4*2+1, 40, 3, 40, 4+1, 1),
			"nprnn":      NewNPRNN(4*2+1, 40, 3, 40, 4+1),
		},
		MaxEpochs:    1000,
		MaxScore:     1,
		TrainingSize: 3000,
		TestingBatch: 20,
		TestingCount: 100,
	},
}
