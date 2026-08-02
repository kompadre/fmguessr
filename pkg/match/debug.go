//go:build !prod

package match

// bestCodebookName holds the name of the best codebook patch found during matching.
// Only populated in non-production builds for diagnostic output.
var bestCodebookName string
