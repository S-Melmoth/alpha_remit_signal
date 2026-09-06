package platform

import (
	"testing"
	"time"
)

func TestMLSignalValidation(t *testing.T) {
	valid := MLSignal{
		Date: "2026-08-04", Corridor: "RUB→TJS", Currency: "TJS",
		Indicator: "ml_good_push", SignalSource: "ml", Strength: 0.74,
		RUBPerUnit: 8.66, TemplateID: "ml_neutral_fallback",
		PushTitle: "Обновился ориентир", PushText: "Проверьте условия перевода.",
	}
	if err := valid.validate(); err != nil {
		t.Fatalf("valid signal rejected: %v", err)
	}

	invalid := valid
	invalid.Corridor = "RUB→KGS"
	if err := invalid.validate(); err == nil {
		t.Fatal("currency/corridor mismatch accepted")
	}

	invalid = valid
	invalid.SignalSource = "calendar_reminder"
	if err := invalid.validate(); err == nil {
		t.Fatal("non-ML row accepted")
	}
}

func TestMLSignalBatchRequiresVersion(t *testing.T) {
	// The database function validates the version before opening a transaction.
	if _, err := SaveMLSignals(nil, nil, MLSignalBatch{}, time.Time{}); err == nil {
		t.Fatal("empty model version accepted")
	}
}
