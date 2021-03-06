// Copyright 2020 The Sequencer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"
	"strings"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tc128"
)

// Text for learning
var Text = `In the beginning God created the heaven and the earth.` /*
And the earth was without form, and void; and darkness was upon the face of the deep.
And the Spirit of God moved upon the face of the waters.
And God said, Let there be light: and there was light..`*/

const (
	// Width is the width of the network
	Width = 256
	// Middle size of the middle layer
	Middle = 512
	// Eta is the learning rate
	Eta = .6
)

func main() {
	Text = strings.ReplaceAll(Text, "\n", "")
	length := len(Text)
	fmt.Println(Text)

	rnd := rand.New(rand.NewSource(3))
	random128 := func(a, b, c float64) complex128 {
		return complex(((b-a)*rnd.Float64()+a)/math.Sqrt(c), ((b-a)*rnd.Float64()+a)/math.Sqrt(c))
	}

	parameters := make([]*tc128.V, 0, 4)
	w0, b0 := tc128.NewV(Width, Middle), tc128.NewV(Middle)
	w1, b1 := tc128.NewV(Middle, Middle), tc128.NewV(Middle)
	w2, b2 := tc128.NewV(Middle, Width), tc128.NewV(Width)
	parameters = append(parameters, &w0, &b0, &w1, &b1, &w2, &b2)
	for _, p := range parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, random128(-1, 1, float64(p.S[0])))
		}
	}

	input, output := tc128.NewV(Width, length), tc128.NewV(Width, length)
	for i, in := range Text {
		fmt.Println(i, in)
		encoding := make([]complex128, Width)
		for e := range encoding {
			encoding[e] = cmplx.Rect(0, math.Pi*float64(i)/float64(length))
		}
		encoding[in] = cmplx.Rect(.001, math.Pi*float64(i)/float64(length))
		input.X = append(input.X, encoding...)
		j := (i + 8) % length
		out := Text[j]
		encoding = make([]complex128, Width)
		for e := range encoding {
			encoding[e] = cmplx.Rect(0, math.Pi*float64(j)/float64(length))
		}
		encoding[out] = cmplx.Rect(.001, math.Pi*float64(j)/float64(length))
		output.X = append(output.X, encoding...)
	}

	l0 := tc128.TanH(tc128.Add(tc128.Mul(w0.Meta(), input.Meta()), b0.Meta()))
	l1 := tc128.TanH(tc128.Add(tc128.Mul(w1.Meta(), l0), b1.Meta()))
	l2 := tc128.TanH(tc128.Add(tc128.Mul(w2.Meta(), l1), b2.Meta()))
	cost := tc128.Avg(tc128.Quadratic(l2, output.Meta()))

	iterations := 32
	pointsAbs, pointsPhase := make(plotter.XYs, 0, iterations), make(plotter.XYs, 0, iterations)
	for i := 0; i < iterations; i++ {
		fmt.Println("iteration", i)
		for _, p := range parameters {
			p.Zero()
		}
		input.Zero()
		output.Zero()

		total := tc128.Gradient(cost).X[0]

		norm := float64(0)
		for _, p := range parameters {
			for _, d := range p.D {
				norm += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm = math.Sqrt(norm)
		if norm > 1 {
			scaling := 1 / norm
			for _, p := range parameters {
				for l, d := range p.D {
					p.X[l] -= Eta * d * complex(scaling, 0)
				}
			}
		} else {
			for _, p := range parameters {
				for l, d := range p.D {
					p.X[l] -= Eta * d
				}
			}
		}

		pointsAbs = append(pointsAbs, plotter.XY{X: float64(i), Y: float64(cmplx.Abs(total))})
		pointsPhase = append(pointsPhase, plotter.XY{X: float64(i), Y: float64(cmplx.Phase(total))})
	}

	plot := func(title, name string, points plotter.XYs) {
		p, err := plot.New()
		if err != nil {
			panic(err)
		}

		p.Title.Text = title
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, name)
		if err != nil {
			panic(err)
		}
	}
	plot("cost abs vs epochs", "cost_abs.png", pointsAbs)
	plot("cost phase vs epochs", "cost_phase.png", pointsPhase)

	{
		type Symbol struct {
			Symbol byte
			Order  float64
			Score  float64
		}
		input := tc128.NewV(Width)
		encoding := make([]complex128, Width)
		l0 := tc128.TanH(tc128.Add(tc128.Mul(w0.Meta(), input.Meta()), b0.Meta()))
		l1 := tc128.TanH(tc128.Add(tc128.Mul(w1.Meta(), l0), b1.Meta()))
		l2 := tc128.TanH(tc128.Add(tc128.Mul(w2.Meta(), l1), b2.Meta()))
		symbols := make([]Symbol, 0)
		for e := range encoding {
			encoding[e] = cmplx.Rect(0, math.Pi*float64(0)/float64(length))
		}
		sym := byte('I')
		encoding[sym] = cmplx.Rect(.001, math.Pi*float64(0)/float64(length))
		symbol := Symbol{
			Symbol: sym,
			Order:  math.Pi * float64(0) / float64(length),
		}
		symbols = append(symbols, symbol)
		for i := 0; i < length; i++ {
			search := func() {
				l2(func(a *tc128.V) bool {
					var max complex128
					sort.Slice(a.X, func(i, j int) bool {
						return math.Abs(cmplx.Phase(a.X[i])) < math.Abs(cmplx.Phase(a.X[j]))
					})
					for j, value := range a.X {
						//if i > 0 && math.Abs(cmplx.Phase(value)) > symbols[len(symbols)-1].Order {
						if j == 110 {
							fmt.Println("here 110", i, cmplx.Abs(value))
						}
						if cmplx.Abs(value) > cmplx.Abs(max) {
							max = value
							symbol.Symbol = byte(j)
							symbol.Order = math.Abs(cmplx.Phase(value))
							symbol.Score = cmplx.Abs(value)
						}
						//}
					}
					return true
				})
			}

			input.Set(encoding)
			search()

			encoding = make([]complex128, Width)
			for e := range encoding {
				encoding[e] = cmplx.Rect(0, math.Pi*float64(i+1)/float64(length)-2*math.Pi)
			}
			encoding[sym] = cmplx.Rect(.001, math.Pi*float64(i+1)/float64(length)-2*math.Pi)
			input.Set(encoding)
			search()

			encoding = make([]complex128, Width)
			for e := range encoding {
				encoding[e] = cmplx.Rect(0, math.Pi*float64(i+1)/float64(length))
			}
			symbol.Order = math.Pi * float64(i+1) / float64(length)
			symbols = append(symbols, symbol)
			encoding[symbol.Symbol] = cmplx.Rect(.001, symbol.Order)
			sym = symbol.Symbol
			symbol = Symbol{}
		}
		for _, symbol := range symbols {
			fmt.Println(symbol)
		}
		fmt.Println("sorting")
		sort.Slice(symbols, func(i, j int) bool {
			return symbols[i].Order < symbols[j].Order
		})
		sequenced := ""
		for _, symbol := range symbols {
			fmt.Println(symbol)
			sequenced += fmt.Sprintf("%c", symbol.Symbol)
		}
		fmt.Println(sequenced)
	}
}
