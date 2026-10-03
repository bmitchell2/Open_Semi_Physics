"""
Reusable schemdraw building blocks for circuit schematics and
block/flow diagrams. See Processing Instructions, Section 31: schemdraw
covers both circuit-symbol drawing (schemdraw.elements) and flowcharts
(schemdraw.flow) in one tool -- use this module instead of hand-placing
matplotlib patches for schematics.
"""
import schemdraw
import schemdraw.elements as elm
from schemdraw import flow


def lc_tank(filename, L_label="L", C_label="C", node_label="$V_{tank}$"):
    """Parallel LC tank: inductor and capacitor between a labeled tank
    node and ground. Visually verified (see chat history 2026-09-05)."""
    with schemdraw.Drawing(file=filename, show=False) as d:
        d.config(unit=3)
        d += elm.Inductor().up().label(L_label)
        d += elm.Line().right().length(3)
        d += elm.Capacitor().down().label(C_label)
        d += elm.Line().left().length(3).hold()
        d += elm.Dot().at((0, 3))
        d += elm.Line().up(0.8).at((0, 3)).idot()
        d += elm.Gap().right().at((0, 3.8)).to((3, 3.8)).label(node_label, fontsize=14)
        d += elm.Ground().at((0, 0))
    return d


def radar_transit_route(filename):
    """Full TX-to-target-and-back-through-RX signal chain for an FMCW
    radar, as a schemdraw flowchart. Visually verified (see chat history
    2026-09-05). TX: Chirp Synthesizer -> PA -> Circulator -> Antenna.
    RX: Antenna -> LNA -> Mixer (LO = chirp) -> LPF -> VGA -> ADC ->
    DSP/FFT -> Range/Velocity."""
    with schemdraw.Drawing(file=filename, show=False) as d:
        d.config(fontsize=10, unit=2)
        row1_y, row2_y = 3.5, 0.5

        synth = flow.Box(w=2.2, h=1.1).label('Chirp\nSynthesizer').at((0, row1_y))
        d += synth
        d += flow.Arrow().right(1.2).at(synth.E)
        pa = flow.Box(w=1.4, h=1.1).label('PA')
        d += pa
        d += flow.Arrow().right(1.2).at(pa.E)
        circ = flow.Box(w=1.8, h=1.1).label('Circulator')
        d += circ
        d += flow.Arrow().right(1.2).at(circ.E)
        ant = flow.Box(w=1.8, h=1.1).label('Antenna')
        d += ant

        d += flow.Arrow().right(1.4).at(ant.E).label('TX', fontsize=9, loc='top')
        tgt = flow.Box(w=1.6, h=1.1).label('Target').linestyle('--')
        d += tgt
        d += flow.Arrow().right(1.4).at(tgt.E).reverse().label('RX', fontsize=9, loc='top')

        lna = flow.Box(w=1.6, h=1.1).label('LNA').at((ant.S[0], row2_y))
        d += flow.Arrow().down(row1_y - row2_y - 1.1).at(ant.S)
        d += lna
        d += flow.Arrow().right(1.2).at(lna.E)
        mix = flow.Box(w=1.8, h=1.1).label('Mixer\n(LO = chirp)')
        d += mix
        d += flow.Arrow().right(1.2).at(mix.E)
        lpf = flow.Box(w=1.6, h=1.1).label('LPF')
        d += lpf
        d += flow.Arrow().right(1.2).at(lpf.E)
        vga = flow.Box(w=1.6, h=1.1).label('VGA')
        d += vga
        d += flow.Arrow().right(1.2).at(vga.E)
        adc = flow.Box(w=1.4, h=1.1).label('ADC')
        d += adc
        d += flow.Arrow().right(1.2).at(adc.E)
        dsp = flow.Box(w=1.8, h=1.1).label('DSP\n(FFT)')
        d += dsp
        d += flow.Arrow().right(1.2).at(dsp.E)
        out = flow.Box(w=2.0, h=1.1).label('Range /\nVelocity').linestyle('--')
        d += out
    return d


def inverting_charge_pump(filename):
    """Four-switch inverting charge pump (flying capacitor). Phase 1 (S1,S3)
    charges C_fly to V_in; phase 2 (S2,S4) grounds its top plate so its
    bottom plate, and C_out, go to -V_in. Visually verified 2026-10-03."""
    with schemdraw.Drawing(file=filename, show=False) as d:
        d.config(fontsize=11, unit=2.4)
        vin = d.add(elm.SourceV().up().label("$V_{in}$", loc="top"))
        d.add(elm.Ground().at(vin.start))
        d.add(elm.Switch().right().at(vin.end).label("S1 ($\\phi_1$)"))
        top = d.add(elm.Dot())
        d.add(elm.Capacitor().down().label("$C_{fly}$", loc="top", ofst=0.25))
        bot = d.add(elm.Dot())
        d.add(elm.Line().right(2.4).at(top.center))
        d.add(elm.Switch().down().label("S2 ($\\phi_2$)", loc="bottom"))
        d.add(elm.Ground())
        d.add(elm.Switch().down().at(bot.center).label("S3 ($\\phi_1$)", loc="top"))
        d.add(elm.Ground())
        d.add(elm.Line().right(1.8).at(bot.center))
        d.add(elm.Line().down(1.6))
        d.add(elm.Switch().right().label("S4 ($\\phi_2$)", loc="bottom"))
        out = d.add(elm.Dot().label("$V_{out}\\approx -V_{in}$", loc="right"))
        d.add(elm.Capacitor().down().at(out.center).label("$C_{out}$", loc="bottom"))
        d.add(elm.Ground())
    return d
