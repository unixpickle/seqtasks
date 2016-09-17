package main

import (
	"log"
	"math/rand"
	"time"

	"github.com/unixpickle/seqtasks"
)

type Task struct {
	Name  string
	Task  seqtasks.Task
	Model *Model

	MaxEpochs    int
	MaxScore     float64
	TrainingSize int
	TestingBatch int
	TestingCount int
}

func (t *Task) Run() {
	rand.Seed(time.Now().UnixNano())
	log.Printf("Running task \"%s\"", t.Name)
	for i := 0; i < t.MaxEpochs; i++ {
		samples := t.Task.NewSamples(t.TrainingSize)
		t.Model.Train(samples)
		score := t.Task.Score(t.Model, t.TestingBatch, t.TestingCount)
		log.Printf("epoch %d: score=%f", i, score)
		if score >= t.MaxScore {
			break
		}
	}
}

var Tasks = []*Task{
	{
		Name:         "XOR last",
		Task:         &seqtasks.XORLastTask{SeqLen: 50},
		Model:        NewModel(1, 40, 40, 1),
		MaxEpochs:    200,
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
		Model:        NewModel(3, 40, 40, 1),
		MaxEpochs:    1000,
		MaxScore:     1,
		TrainingSize: 600,
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
		Model:        NewModel(3, 40, 40, 1),
		MaxEpochs:    300,
		MaxScore:     1,
		TrainingSize: 300,
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
		Model:        NewModel(4*2+1, 40, 40, 4+1),
		MaxEpochs:    1000,
		MaxScore:     1,
		TrainingSize: 3000,
		TestingBatch: 20,
		TestingCount: 100,
	},
}
