package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func main() {
	output := flag.String("o", "morphed.syx", "output .syx file")
	note := flag.Int("note", 60, "MIDI note for rendering comparison")
	duration := flag.Duration("dur", time.Second, "render duration")
	ratio := flag.Float64("ratio", 0.5, "target distance ratio (0.5 = halve distance)")
	iter := flag.Int("iter", 100, "optimization iterations after lerp (0 = lerp only)")
	verbose := flag.Bool("v", false, "print progress")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <source.syx> <target.syx>\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Morph a DX7 patch toward a target patch, reducing spectral distance.\n")
		fmt.Fprintf(os.Stderr, "If either .syx contains a bank (32 voices), only the first voice is used.\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() < 2 {
		flag.Usage()
		os.Exit(1)
	}
	syxSource := flag.Arg(0)
	syxTarget := flag.Arg(1)

	// Load source patch
	fmt.Printf("Loading source: %s\n", syxSource)
	source, err := match.LoadSyxPatch(syxSource)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading source: %v\n", err)
		os.Exit(1)
	}

	// Load target patch
	fmt.Printf("Loading target: %s\n", syxTarget)
	target, err := match.LoadSyxPatch(syxTarget)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading target: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Morphing source → target (ratio=%.2f, note=%d, dur=%.2fs, iter=%d)\n",
		*ratio, *note, duration.Seconds(), *iter)

	// Run morph
	result := match.Morph(match.MorphConfig{
		Source:  source,
		Target:  target,
		Note:    uint8(*note),
		Dur:     *duration,
		Ratio:   *ratio,
		MaxIter: *iter,
		Verbose: *verbose,
	})

	fmt.Printf("\nResults:\n")
	fmt.Printf("  Source distance:  %.6f\n", result.SourceDist)
	fmt.Printf("  Morph distance:   %.6f\n", result.MorphDist)
	fmt.Printf("  Achieved ratio:   %.4f (%.1f%% of source)\n", result.TargetRatio, result.TargetRatio*100)
	fmt.Printf("  Lerp alpha:       %.4f\n", result.Alpha)
	fmt.Printf("  Optimization iters: %d\n", result.Iterations)

	// Export
	syxData := result.Morphed.SysexBulkDump()
	if err := os.WriteFile(*output, syxData, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing %s: %v\n", *output, err)
		os.Exit(1)
	}
	fmt.Printf("\nWrote morphed patch to %s (%d bytes)\n", *output, len(syxData))
}
