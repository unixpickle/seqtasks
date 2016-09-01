package main

func main() {
	for _, task := range Tasks {
		task.Run()
	}
}
