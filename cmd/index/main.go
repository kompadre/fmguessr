package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

// optionalString is a flag.Value that distinguishes between "not provided"
// and "provided with empty value" (for --sync-timbre without argument).
type optionalString struct {
	value    string
	provided bool
}

func (o *optionalString) String() string { return o.value }
func (o *optionalString) Set(s string) error {
	o.value = s
	o.provided = true
	return nil
}

func main() {
	outPath := flag.String("o", "dx7_codebook.bin", "output codebook binary file")
	overwrite := flag.Bool("overwrite", false, "overwrite existing codebook instead of appending")
	renderDur := flag.Duration("d", 1*time.Second, "audio render duration per patch")
	noteNum := flag.Int("note", 48, "MIDI note number to render (default 48 = C3)")

	var syncTimbre optionalString
	flag.Var(&syncTimbre, "sync-timbre",
		"backfill timbre codebook from fingerprint codebook entries.\n"+
			"                         Provide a path to override the default (*_timbre.bin),\n"+
			"                         or use without a value for auto-derived path from -o flag.\n"+
			"                         No .syx files required — renders each entry through the synth.")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <syx_file_or_directory...>\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Build or update an audio fingerprint codebook index from DX7 .syx bank files.\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  %s banks/*.syx                          Index .syx files\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --sync-timbre -o dx7_codebook.bin    Backfill timbre from existing DXCB\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --sync-timbre=custom.bin -o dx7.bin  Backfill to custom path\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nFlags:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() < 1 && !syncTimbre.provided {
		flag.Usage()
		os.Exit(1)
	}

	var syxFiles []string
	for _, arg := range flag.Args() {
		info, err := os.Stat(arg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: cannot access %s: %v\n", arg, err)
			continue
		}
		if info.IsDir() {
			err := filepath.Walk(arg, func(path string, info os.FileInfo, err error) error {
				if err == nil && !info.IsDir() && filepath.Ext(path) == ".syx" {
					syxFiles = append(syxFiles, path)
				}
				return nil
			})
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error scanning directory %s: %v\n", arg, err)
			}
		} else if filepath.Ext(arg) == ".syx" {
			syxFiles = append(syxFiles, arg)
		}
	}

	if len(syxFiles) == 0 && !syncTimbre.provided {
		fmt.Fprintf(os.Stderr, "No .syx files found.\n")
		os.Exit(1)
	}

	// Determine timbre codebook path: --sync-timbre=value overrides auto-derived
	timbrePath := *outPath
	if len(*outPath) > 4 && (*outPath)[len(*outPath)-4:] == ".bin" {
		timbrePath = (*outPath)[:len(*outPath)-4] + "_timbre.bin"
	} else {
		timbrePath = *outPath + "_timbre.bin"
	}
	if syncTimbre.provided && syncTimbre.value != "" {
		timbrePath = syncTimbre.value
	}

	codebook := &match.Codebook{}
	timbreCodebook := &match.TimbreCodebook{}

	if !*overwrite {
		if loaded, err := match.LoadCodebook(*outPath); err == nil {
			codebook = loaded
			fmt.Printf("Loaded existing codebook %s (%d existing patches)\n", *outPath, len(codebook.Entries))
		}
		if loadedT, err := match.LoadTimbreCodebook(timbrePath); err == nil {
			timbreCodebook = loadedT
			fmt.Printf("Loaded existing timbre codebook %s (%d existing patches)\n", timbrePath, len(timbreCodebook.Entries))
		}
	}

	fmt.Printf("Indexing %d .syx file(s) into %s and %s...\n", len(syxFiles), *outPath, timbrePath)

	startTime := time.Now()
	addedTotal := 0
	skippedTotal := 0

	for i, file := range syxFiles {
		fmt.Printf("[%d/%d] Indexing %s...\n", i+1, len(syxFiles), file)
		entries, err := match.IndexSysexBank(file, uint8(*noteNum), *renderDur)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Error indexing fingerprint codebook for %s: %v\n", file, err)
			continue
		}
		tEntries, err := match.IndexSysexBankTimbre(file, uint8(*noteNum), *renderDur)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Error indexing timbre codebook for %s: %v\n", file, err)
			continue
		}

		added := codebook.AddEntriesDeduplicated(entries)
		timbreCodebook.AddEntriesDeduplicated(tEntries)

		skipped := len(entries) - added
		addedTotal += added
		skippedTotal += skipped
		if skipped > 0 {
			fmt.Printf("  -> Added %d new patches (%d duplicates skipped)\n", added, skipped)
		} else {
			fmt.Printf("  -> Added %d new patches\n", added)
		}
	}

	// Backfill timbre codebook from fingerprint codebook entries
	if syncTimbre.provided && len(codebook.Entries) > 0 {
		fmt.Printf("Syncing timbre codebook from %d fingerprint entries -> %s...\n", len(codebook.Entries), timbrePath)
		startBackfill := time.Now()
		added := timbreCodebook.BackfillFromCodebook(codebook, uint8(*noteNum), *renderDur)
		elapsed := time.Since(startBackfill)
		fmt.Printf("  -> Added %d new timbre entries (%d already existed) in %v\n",
			added, len(codebook.Entries)-added, elapsed)
	}

	if err := codebook.Save(*outPath); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving codebook: %v\n", err)
		os.Exit(1)
	}

	if err := timbreCodebook.Save(timbrePath); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving timbre codebook: %v\n", err)
		os.Exit(1)
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\nDone in %v: added %d new patches (%d duplicates skipped).\n  Fingerprint Index: %d patches -> %s\n  Timbre Index:      %d patches -> %s\n",
		elapsed, addedTotal, skippedTotal, len(codebook.Entries), *outPath, len(timbreCodebook.Entries), timbrePath)
}
