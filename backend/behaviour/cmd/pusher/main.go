package main

import (
	"ai-hack/behaviour/internal/platform"
	"log"
)

func main() {
	if err := platform.Serve("pusher"); err != nil {
		log.Fatal(err)
	}
}
