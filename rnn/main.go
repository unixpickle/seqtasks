package main

import (
	"fmt"
	"os"
	"sort"
)

func main() {
	if len(os.Args) != 2 && len(os.Args) != 3 {
		fmt.Fprint(os.Stderr, "Usage:", os.Args[0], " model [task]\n\n")
		fmt.Fprintln(os.Stderr, "Available models:")
		for _, model := range AllModels() {
			fmt.Fprintln(os.Stderr, " -", model)
		}
		fmt.Fprintln(os.Stderr, "\nAvailable tasks:")
		for _, task := range Tasks {
			fmt.Fprintln(os.Stderr, " -", task.Name)
		}
		fmt.Fprintln(os.Stderr)
		os.Exit(1)
	}

	model := os.Args[1]
	if !AllModelsSet()[model] {
		fmt.Fprintln(os.Stderr, "Unknown model:", model)
		os.Exit(1)
	}

	tasks := Tasks
	if len(os.Args) == 3 {
		for _, t := range tasks {
			if t.Name == os.Args[2] {
				tasks = []*Task{t}
				break
			}
		}
		if tasks[0].Name != os.Args[2] {
			fmt.Fprintln(os.Stderr, "Unknown task:", os.Args[2])
			os.Exit(1)
		}
	}

	for _, task := range tasks {
		task.Run(model)
	}
}

func AllModels() []string {
	models := AllModelsSet()
	res := make([]string, 0, len(models))
	for m := range models {
		res = append(res, m)
	}
	sort.Strings(res)
	return res
}

func AllModelsSet() map[string]bool {
	models := map[string]bool{}
	for _, m := range Tasks {
		for n := range m.Models {
			models[n] = true
		}
	}
	return models
}
