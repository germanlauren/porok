// Manuscrito GRL en .docx. Formato de AGU: 12 unidades de publicacion, donde
// una unidad son 500 palabras o un elemento grafico. Con cuatro figuras quedan
// ocho unidades de texto, es decir 4000 palabras; el cuerpo apunta a ~3200.
//
// Resumen <= 150 palabras, resumen en lenguaje llano <= 200, y hasta tres
// "key points" de 140 caracteres cada uno sin abreviaturas.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  ImageRun, PageOrientation, Footer, PageNumber, LineRuleType, TabStopType,
  Table, TableRow, TableCell, WidthType, BorderStyle, VerticalAlign,
} = require("docx");

const RAIZ = path.resolve(__dirname, "..");
// --- ecuacion numerada, compuesta con STIX por scripts/ecuaciones.py -------
// Se insertan como imagen de 400 dpi y no como OMML: el XML de OMML que
// genera esta libreria es valido, pero no hay aqui ningun visor que lo
// renderice para comprobarlo, y una ecuacion que no se puede verificar no se
// entrega. Asi se ve igual en Word, en LibreOffice y en el PDF.
const EQ = path.join(RAIZ, "figuras_eq");

function eq(n) {
  const archivo = path.join(EQ, "eq" + String(n).padStart(2, "0") + ".png");
  const datos = fs.readFileSync(archivo);
  const w = datos.readUInt32BE(16), h = datos.readUInt32BE(20);
  // 400 dpi -> ~96 dpi con el cuerpo a 12 pt, y nunca mas ancha que la caja
  const esc = Math.min(0.31, 500 / w);
  const sinBorde = {
    top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  };
  const celda = (hijos, ancho, alineacion) => new TableCell({
    borders: sinBorde, verticalAlign: VerticalAlign.CENTER,
    width: { size: ancho, type: WidthType.DXA },
    children: [new Paragraph({ alignment: alineacion, children: hijos,
                               spacing: { line: 240,
                                          lineRule: LineRuleType.AUTO } })],
  });
  // Tabla sin bordes: la ecuacion centrada y el numero a la derecha quedan
  // alineados en cualquier visor, cosa que con tabuladores no ocurre cuando
  // la imagen es mas alta que la linea.
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [8200, 1160],
    borders: sinBorde,
    rows: [new TableRow({ children: [
      celda([new ImageRun({ data: datos, type: "png",
                            transformation: { width: Math.round(w * esc),
                                              height: Math.round(h * esc) } })],
            8200, AlignmentType.CENTER),
      celda([new TextRun({ text: "(" + n + ")", size: 24,
                           font: "Times New Roman" })],
            1160, AlignmentType.RIGHT),
    ] })],
  });
}

// Fraccion dentro del texto corrido, con barra horizontal.
function inl(nombre) {
  const datos = fs.readFileSync(path.join(EQ, "in_" + nombre + ".png"));
  const w = datos.readUInt32BE(16), h = datos.readUInt32BE(20);
  const esc = 0.20;
  return new ImageRun({ data: datos, type: "png",
                        transformation: { width: Math.max(4, Math.round(w * esc)),
                                          height: Math.round(h * esc) } });
}

// Parrafo que mezcla texto y fracciones en linea. Cada elemento es una
// cadena, un par [texto, estilo] como en pr(), o {eq: "nombre"}.
function pm(partes) {
  return new Paragraph({
    spacing: INTER,
    children: partes.map((x) => {
      if (typeof x === "object" && x.eq) return inl(x.eq);
      const [t, est] = Array.isArray(x) ? x : [x, ""];
      return new TextRun({ text: t, size: 24, font: "Times New Roman",
                           italics: est.includes("i"),
                           bold: est.includes("b"),
                           subScript: est.includes("_") });
    }),
  });
}

// Una tabla no lleva espacio propio: se enmarca con parrafos vacios cortos.
function eqp(n) {
  return [new Paragraph({ spacing: { before: 0, after: 0, line: 120 },
                          children: [] }), eq(n),
          new Paragraph({ spacing: { before: 0, after: 0, line: 120 },
                          children: [] })];
}

const FIG = path.join(RAIZ, "figuras_fd");

// ---------------------------------------------------------------- helpers --
const INTER = { line: 480, lineRule: LineRuleType.AUTO, after: 0 };   // doble

function p(texto, opts = {}) {
  return new Paragraph({
    spacing: opts.spacing || INTER,
    alignment: opts.alignment,
    indent: opts.indent,
    children: [new TextRun({ text: texto, size: 24, font: "Times New Roman",
                             italics: opts.italics, bold: opts.bold })],
  });
}

// Parrafo con tramos en cursiva/negrita: se pasa un arreglo [texto, estilo].
function pr(tramos, opts = {}) {
  return new Paragraph({
    spacing: opts.spacing || INTER,
    alignment: opts.alignment,
    children: tramos.map(([t, est]) => new TextRun({
      text: t, size: 24, font: "Times New Roman",
      italics: (est || "").includes("i"), bold: (est || "").includes("b"),
      subScript: (est || "").includes("_"), superScript: (est || "").includes("^"),
    })),
  });
}

function h(texto, nivel = HeadingLevel.HEADING_1) {
  return new Paragraph({
    heading: nivel,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text: texto, size: 24, bold: true,
                             font: "Times New Roman", color: "000000" })],
  });
}

function figura(archivo, ancho, alto, titulo, pie) {
  const datos = fs.readFileSync(path.join(FIG, archivo));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 120 },
      children: [new ImageRun({ data: datos, type: "png",
                                transformation: { width: ancho, height: alto } })],
    }),
    new Paragraph({
      spacing: { line: 240, lineRule: LineRuleType.AUTO, after: 240 },
      children: [
        new TextRun({ text: titulo, size: 22, bold: true, font: "Times New Roman" }),
        new TextRun({ text: pie, size: 22, font: "Times New Roman" }),
      ],
    }),
  ];
}

function ref(texto) {
  return new Paragraph({
    spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 },
    indent: { left: 720, hanging: 720 },
    children: [new TextRun({ text: texto, size: 24, font: "Times New Roman" })],
  });
}

// ------------------------------------------------------------------ texto --
const cuerpo = [];

// --- portada ---
cuerpo.push(new Paragraph({
  spacing: { after: 240 },
  children: [new TextRun({
    text: "A Second-Order Damage Tensor Is Sufficient for Damage-Induced "
        + "Permeability; Scalar Crack Density Is Not",
    size: 28, bold: true, font: "Times New Roman" })],
}));

cuerpo.push(p("German Orlando Romero Suárez¹, Diego Fernando Villegas "
              + "Bermúdez¹, Wilmer Velilla Díaz²",
              { spacing: { after: 60 } }));
cuerpo.push(p("¹ Universidad Industrial de Santander, Bucaramanga, Colombia",
              { spacing: { after: 20 } }));
cuerpo.push(p("² Universidad de La Serena, La Serena, Chile",
              { spacing: { after: 60 } }));
cuerpo.push(p("Corresponding author: German Orlando Romero Suárez "
              + "(german2218069@correo.uis.edu.co)",
              { spacing: { after: 240 } }));

// --- key points ---
cuerpo.push(h("Key Points"));
[
  "Permeability of mechanically damaged rock spreads by up to 43 percent "
  + "among loading paths that share one scalar crack density",
  "Matching the full second-order damage tensor collapses that spread to 3 "
  + "percent, in two and in three dimensions",
  "An exponential damage-permeability law at its own best fit still errs by "
  + "57 percent in the median, with a sign that flips as damage grows",
].forEach((t, i) => cuerpo.push(p(`${i + 1}. ${t}`,
  { spacing: { line: 240, lineRule: LineRuleType.AUTO, after: 120 } })));

// --- abstract ---
cuerpo.push(h("Abstract"));
cuerpo.push(p(
  "Reservoir simulators close the damage-permeability coupling with a state "
  + "variable, yet it has never been tested whether that variable carries "
  + "enough information, because laboratory experiments cannot hold damage "
  + "fixed while the stress path varies. We build a numerical reference that "
  + "can: phase-field fracture under crack-length control, so the softening "
  + "branch is traversed, coupled to FFT-Galerkin homogenization of Darcy "
  + "flow on the same periodic cell. Across five loading paths and 210 damage "
  + "states on one microstructure, permeability at matched scalar crack "
  + "density spreads by up to 43.5 percent, and the spread grows with damage. "
  + "Matching the full second-order damage tensor instead collapses the "
  + "spread to 3.0 percent, with no residual floor: load-path memory enters "
  + "permeability only through the tensor. The exponential damage law in "
  + "common use, granted its own best-fit exponent, still errs by 57 percent "
  + "in the median, underpredicting at moderate damage and overpredicting "
  + "beyond tr D = 0.3."));

// --- plain language summary ---
cuerpo.push(h("Plain Language Summary"));
cuerpo.push(p(
  "When rock deep underground is squeezed, it cracks, and the cracks let "
  + "fluid move more easily. Engineers who forecast how much oil, gas, water "
  + "or stored carbon dioxide will flow need a rule that turns “how damaged "
  + "is the rock” into “how easily does fluid flow”. The rules in use "
  + "summarize damage with a single number. Nobody has been able to check "
  + "whether one number is enough, because in the laboratory you cannot "
  + "create two rock samples that are damaged by the same amount but in "
  + "different ways. In a computer you can. We simulated cracks growing under "
  + "five different loading histories and computed the flow through each "
  + "resulting crack network directly. Rock states that share the same single "
  + "damage number can differ in permeability by more than forty percent, and the "
  + "disagreement widens as damage grows. But when we compare states that "
  + "share a richer description of damage, one that also records direction, "
  + "the disagreement almost vanishes. The practical message is that the "
  + "directional description is worth its cost, and that it must be updated "
  + "by the mechanics rather than guessed from porosity."));

// --- 1 introduction ---
cuerpo.push(h("1 Introduction"));
cuerpo.push(p(
  "Fluid transport in damaged rock is controlled by a crack network that the "
  + "stress history creates. Reservoir and repository simulators cannot "
  + "resolve that network, so they close the coupling: a damage variable is "
  + "advanced by a constitutive law and permeability is read off a function "
  + "of that variable. The closures in use are scalar and isotropic, most "
  + "commonly the exponential form k = k₀ exp(α D) calibrated on "
  + "core measurements (Souley et al., 2001), or a power law in porosity of "
  + "fractal inspiration. Anisotropic alternatives exist and are built on a "
  + "second-order damage tensor (Shao et al., 2005), following the fabric- "
  + "and crack-tensor tradition of Oda (1982, 1985) and Snow (1969)."));
cuerpo.push(p(
  "Whether a state variable of either kind carries enough information is an "
  + "open question, and the reason it stays open is experimental. Laboratory "
  + "work has established that permeability depends on the stress path and "
  + "not only on the final state (Ren et al., 2026; Wang et al., 2021; Liu et "
  + "al., 2021), and that unloading is less stress-sensitive than loading "
  + "(Zhang et al., 2020). But a core sample cannot be brought to a "
  + "prescribed damage tensor by two different routes, so the observation "
  + "cannot be turned into a statement about closures. It is absorbed instead "
  + "as case-dependent recalibration."));
cuerpo.push(p(
  "Synthetic crack networks close part of the gap. Zhou et al. (2011) and Li "
  + "and Li (2015) showed with statistically generated networks that crack "
  + "density alone does not determine permeability and that connectivity is "
  + "the missing control. Their cracks are inserted rather than grown, and "
  + "their comparison is against a scalar. Both choices matter here: the "
  + "topology a loading path produces is a small and biased subset of what a "
  + "generator can produce, and simulators carry the tensor, not the "
  + "scalar."));
cuerpo.push(p(
  "The numerical ingredients now exist. Phase-field fracture (Bourdin et "
  + "al., 2000, 2008) on a periodic cell gives mechanically consistent crack "
  + "populations, crack-length control (Aranda & Segurado, 2025) traverses "
  + "the unstable branch on which they coalesce, and the same spectral "
  + "machinery solves the Darcy problem on the same grid (Vondřejc et al., "
  + "2014). Damage and permeability are then measured on one cell at one "
  + "instant, with the loading path free and the damage state an observable "
  + "rather than an input."));
cuerpo.push(p(
  "We use that setup to ask three questions with parameter-free answers. "
  + "First, how much can permeability differ between states that a scalar "
  + "closure is obliged to treat as identical? That difference is the "
  + "irreducible error of the entire scalar family, independent of "
  + "functional form and calibration. Second, does the same test applied to "
  + "the full second-order tensor also show a floor, which would mean no "
  + "second-order closure can work? Third, given those bounds, how far off "
  + "are the closures actually in use?"));

// --- 2 methods ---
cuerpo.push(h("2 Reference Model"));
cuerpo.push(h("2.1 Microstructure and fracture", HeadingLevel.HEADING_2));
cuerpo.push(pr([
  ["We work on a periodic unit cell discretized on a 96 × 96 grid. The "
   + "matrix is linear elastic and isotropic, ", ""],
  ["E", "i"], [" = 1, ", ""], ["ν", "i"], [" = 0.2 in nondimensional "
   + "units. Fracture toughness ", ""],
  ["G", "i"], ["c", "i_"], [" is heterogeneous: a lognormal field of mean "
   + "2 × 10⁻³ with 15% dispersion and a six-pixel correlation "
   + "length, into which 40 pre-existing microcracks are seeded as short "
   + "segments of length 0.10", ""],
  ["L", "i"], [" and random orientation where ", ""],
  ["G", "i"], ["c", "i_"], [" is reduced to 12% of its mean. The seeds are "
   + "not cosmetic. With a smooth toughness field alone, AT2 phase-field "
   + "damage under strain control grows diffuse and never localizes, which "
   + "produces a damage tensor that is not a crack density at all; the "
   + "seeds put the model in the regime where cracks nucleate, propagate and "
   + "link, and which seeds are selected is exactly the mechanism under "
   + "test.", ""],
]));
cuerpo.push(pm([
  ["Fracture follows the AT2 regularization with the volumetric-deviatoric "
   + "split of Amor et al. (2009) and the irreversibility of Miehe et al. "
   + "(2010), in which the driving field is the running maximum of the "
   + "positive elastic energy density. The regularization length is "
   + "ℓ = 4 pixels = ", ""],
  { eq: "L24" },
  [". Equilibrium and damage are solved alternately on the "
   + "spectral grid, with the gradient term of Equation 3 discretized by "
   + "forward differences: the rotated scheme used for elasticity and flow "
   + "(Willot, 2015) has symbols that vanish on whole lines of the "
   + "three-dimensional lattice, where damage would go unregularized. In "
   + "two dimensions the choice moves permeability by under 1%.", ""],
]));
cuerpo.push(p(
  "The regularized energy, the AT2 surface density and the irreversibility "
  + "condition are, in that order:", { spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 } }));
eqp(1).forEach((x) => cuerpo.push(x));
eqp(2).forEach((x) => cuerpo.push(x));
eqp(3).forEach((x) => cuerpo.push(x));
cuerpo.push(p(
  "Here η = 10⁻⁶ is the residual stiffness that keeps the elastic problem "
  + "invertible inside a fully broken zone, and the running maximum in "
  + "Equation 3 is what makes damage irreversible and, as Section 3.1 "
  + "shows, what erases sub-dominant excursions."));
cuerpo.push(p(
  "Under load control the response reaches a limit point beyond which lies "
  + "everything of interest: coalescence, and the transition from diffuse "
  + "damage to a connected crack. We therefore prescribe the crack length "
  + "and solve for the load factor (Aranda & Segurado, 2025), which converts "
  + "that limit point into a regular point. Figure 1a shows the branch: the "
  + "load factor peaks at λ = 0.063 and falls by 14% while the crack grows. "
  + "A load-controlled sweep would have stopped at the peak, tr D = 0.375, "
  + "missing the states where the closures diverge most."));
cuerpo.push(p(
  "Formally, each increment solves for the load multiplier λ that delivers a "
  + "prescribed increase of crack density:", { spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 } }));
eqp(4).forEach((x) => cuerpo.push(x));

cuerpo.push(h("2.2 Damage tensor", HeadingLevel.HEADING_2));
cuerpo.push(p(
  "From the phase field we extract a second-order damage tensor as a "
  + "magnitude times a direction:", { spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 } }));
eqp(5).forEach((x) => cuerpo.push(x));
cuerpo.push(pr([
  ["Taking the magnitude from the φ² term and the direction from "
   + "the normalized gradient tensor removes a mesh-orientation bias: the "
   + "naive gradient-only estimator varies by 14.6% between a crack aligned "
   + "with the grid and one at 45°, whereas the split estimator varies "
   + "by 0.52%. We also record a localization ratio ", ""],
  ["R", "i"], [", the ratio of gradient to bulk contributions to the "
   + "regularized crack surface. The two are equal for the optimal AT2 "
   + "profile, so the continuum value is one; diffuse damage falls below it "
   + "and a formed crack approaches it from above. Evaluated discretely the "
   + "ratio is scheme-dependent — our spectral gradient returns values about "
   + "twice those of second-order finite differences, uniformly across the "
   + "sweep — so we read ", ""],
  ["R", "i"], [" as a relative indicator between paths and anchor statements "
   + "about crack formation on the connectivity of the damaged phase, which "
   + "no choice of derivative can shift.", ""],
]));

cuerpo.push(h("2.3 Permeability", HeadingLevel.HEADING_2));
cuerpo.push(pr([
  ["Permeability is computed on the same grid by FFT-Galerkin homogenization "
   + "of the Darcy problem (Vondřejc et al., 2014), using the rotated "
   + "discrete gradient of Willot (2015), which suppresses the spurious "
   + "oscillations that the spectral operator produces at high contrast. "
   + "The local conductivity interpolates between matrix and crack as "
   + "k(φ) = ", ""],
  ["k", "i"], ["m", "i_"], [" times a factor given in Equation 6, with "
   + "contrast χ = 10⁴; all permeabilities below are reported in units "
   + "of ", ""],
  ["k", "i"], ["m", "i_"], [". The linear system is solved by conjugate "
   + "gradients to a relative residual of 10⁻¹⁰, normalized "
   + "against the physical load rather than the right-hand side, which "
   + "matters when the exact solution is nearly trivial. Primal and dual "
   + "formulations agree to 10⁻¹³, as they must when both use "
   + "the same discrete gradient symbol, and we use that identity as a "
   + "correctness test rather than as a discretization bound.", ""],
]));
eqp(6).forEach((x) => cuerpo.push(x));
cuerpo.push(p(
  "The effective tensor follows from the cell average of the flux under a "
  + "prescribed macroscopic gradient G:", { spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 } }));
eqp(7).forEach((x) => cuerpo.push(x));
cuerpo.push(p(
  "This is a Darcy-scale surrogate, not Stokes flow in an open fracture with "
  + "a physical aperture. It is the right level for the questions posed here, "
  + "which are about whether a given state variable determines permeability "
  + "at all; the map from geometry to permeability is local and isotropic by "
  + "construction, so any path dependence we find is a property of the "
  + "geometry, not of the flow model. Section 5 returns to what changes when "
  + "the cubic law is used instead."));

cuerpo.push(h("2.4 Loading paths", HeadingLevel.HEADING_2));
cuerpo.push(p(
  "Five paths act on the identical microstructure, so the microstructure is "
  + "the control and the path is the variable. Three are proportional and "
  + "differ in confinement, which is the effective-stress path of depletion: "
  + "uniaxial (P1), pure shear (P2) and confined (P3). Two are "
  + "non-proportional: the loading direction is rotated by 90° partway "
  + "through, after an excursion to Γ = 0.15 (P4) or Γ = 0.32 (P5). The "
  + "first is sub-dominant and the second dominant in stored energy. P4 and P5 end loading in the same direction as P1 "
  + "and reach the same crack density, so any difference from P1 is "
  + "attributable to history and not to the final loading direction. In "
  + "total the two-dimensional sweep contains 210 damage states spanning "
  + "tr D from 0.015 to 0.63 (Figure 1b). The same five paths are then run "
  + "on a 96³ cell with the seeding density held fixed in the sense of "
  + "flaw length over mean spacing, which fixes the number of penny-shaped "
  + "flaws at 253 rather than the 40 segments used in two dimensions; that "
  + "sweep contains 403 states and is analyzed in Section 3.4."));

// --- figura 1 ---
figura("fig1_referencia.png", 612, 254,
  "Figure 1. ",
  "The reference. (a) Load factor against prescribed crack density for the "
  + "uniaxial path. Crack-length control converts the limit point into a "
  + "regular point and follows the softening branch, on which the load falls "
  + "by 14% while the crack grows. (b) Effective permeability along the "
  + "loading axis for the five paths, in units of the matrix permeability."
).forEach((x) => cuerpo.push(x));

// --- 3 results ---
cuerpo.push(h("3 Results"));
cuerpo.push(h("3.1 Load-path memory is real but selective", HeadingLevel.HEADING_2));
cuerpo.push(p(
  "P4 and P5 differ from the monotonic path P1 only by an excursion in the "
  + "orthogonal direction before the final loading leg. The two behave "
  + "oppositely (Figure 2a). P4, whose excursion is sub-dominant, departs "
  + "from P1 by at most 5.25% during the excursion and then converges: by "
  + "tr D = 0.36 the discrepancy is 0.002%, more than three orders of "
  + "magnitude below its peak, and it stays below 0.012% up to tr D = 0.63; "
  + "the two paths have become indistinguishable. P5, whose excursion is "
  + "dominant, never converges: from tr D = 0.405 to the end of the sweep "
  + "it holds on a plateau of 12.6 to 14.2%, and at tr D = 0.405 its damage "
  + "tensor has a visibly different shape: the ratio of its axial component "
  + "to its trace is 0.529, against 0.644 for P1."));
cuerpo.push(p(
  "The asymmetry is structural, not numerical: by Equation 3 the driving "
  + "field is a running maximum, so an excursion later exceeded by the final "
  + "leg is overwritten pointwise and leaves no trace, while a dominant one "
  + "cannot be erased. A closure calibrated only on monotonic tests will "
  + "therefore look adequate on a large class of non-proportional paths and "
  + "fail on the rest, which is a plausible reading of why laboratory "
  + "stress-path effects are reported inconsistently."));
cuerpo.push(p(
  "Figure 2b shows what distinguishes the states, and the connectivity of the "
  + "damaged phase makes it unambiguous. On the uniaxial path the damaged "
  + "phase becomes a single connected cluster at tr D = 0.375 and stays one; "
  + "at that damage the shear path has not yet formed any cluster, and even "
  + "at tr D = 0.63 its largest holds only 59% of the damaged phase. Its "
  + "localization ratio stays lower throughout. One path has formed "
  + "a crack and the other has not. No state in the sweep percolates, so all "
  + "comparisons below are pre-percolation."));

figura("fig2_camino.png", 612, 254,
  "Figure 2. ",
  "Load-path memory. (a) Relative departure of permeability from the "
  + "monotonic path, at matched crack density, for a sub-dominant excursion "
  + "(P4) and a dominant one (P5); dotted lines mark where each path rotates "
  + "its loading direction. The sub-dominant excursion is erased; the "
  + "dominant one is not. (b) The localization ratio for the uniaxial and "
  + "shear paths over the same states."
).forEach((x) => cuerpo.push(x));

cuerpo.push(h("3.2 No scalar closure escapes a floor that reaches 43%", HeadingLevel.HEADING_2));
cuerpo.push(p(
  "A scalar closure asserts that permeability is some function of crack "
  + "density. Whatever that function is, it must return one value for one "
  + "value of tr D. The spread of the reference permeability across states "
  + "that share tr D is therefore a lower bound on the error of the whole "
  + "family: no functional form, no calibration, no extra fitted parameter "
  + "can reduce it."));
cuerpo.push(p(
  "We measure it as the relative spread at matched crack density:",
  { spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 } }));
eqp(8).forEach((x) => cuerpo.push(x));
cuerpo.push(p(
  "That bound is not small and it is not constant (Figure 3a). It grows "
  + "monotonically with damage, from 0.8% at tr D = 0.015 to 43.5% at "
  + "tr D = 0.63, with a median of 9.0% over the sweep. The growth matters "
  + "more than the peak value: the closure is most reliable exactly where "
  + "damage is negligible and least reliable where it drives the flow. "
  + "This reproduces, for mechanically grown cracks and against a tensor-"
  + "consistent measure of damage, the non-uniqueness that Zhou et al. "
  + "(2011) and Li and Li (2015) found for inserted networks, and it "
  + "serves here as a consistency check on the setup."));

cuerpo.push(h("3.3 The second-order tensor closes the gap", HeadingLevel.HEADING_2));
cuerpo.push(p(
  "We then applied the identical test to the tensor. Among all pairs of "
  + "states drawn from different loading paths with tr D > 0.10, we "
  + "selected those whose full damage tensors agree to within 2% in relative "
  + "Frobenius norm, and asked how much their permeabilities differ. There "
  + "are 59 such pairs, spanning tr D from 0.105 to 0.63 and every "
  + "combination of paths; P4 and P5 before P4 rotates are excluded, since "
  + "they share one history and are the same state. The largest "
  + "discrepancy is 2.97% and the median is 0.21%."));
cuerpo.push(p(
  "The pairing criterion and the quantity reported for each pair are:",
  { spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 } }));
eqp(9).forEach((x) => cuerpo.push(x));
cuerpo.push(p(
  "The stronger statement is in the scaling (Figure 3b). Plotted against "
  + "tensor mismatch, the permeability discrepancy follows a line of unit "
  + "slope (fitted log-log slope 1.2) over three decades and continues down "
  + "to mismatches of 0.007%, with no floor. A hidden variable — connectivity, tortuosity, anything "
  + "the second-order tensor cannot see — would appear as a plateau: pairs "
  + "with identical tensors but persistently different permeabilities. "
  + "There is none. In this regime the second-order damage tensor is a "
  + "sufficient state variable, and the load-path memory documented in "
  + "Section 3.1 enters permeability entirely through it."));
cuerpo.push(p(
  "We had expected the opposite: that mechanically grown topology would "
  + "defeat any second-order descriptor, as it defeats the scalar. The "
  + "refutation is the more useful result, because it says the tensor "
  + "already implemented in anisotropic damage simulators is in principle "
  + "enough, and the error lies in the functional forms bolted onto it and "
  + "in evolving it from porosity rather than from the mechanics."));

figura("fig3_escalar_vs_tensor.png", 612, 490,
  "Figure 3. ",
  "Scalar closures fail where tensor closures need not. (a) Spread of "
  + "permeability among states sharing a crack density, which lower-bounds "
  + "the error of any closure of the form K = f(tr D); only levels reached by "
  + "at least three paths are shown. (b) The same comparison for the full "
  + "tensor: every cross-path pair with tr D > 0.10, permeability discrepancy "
  + "against tensor mismatch. Green marks pairs matched to within 2%. The "
  + "one-to-one scaling continues to the smallest mismatches, so there is no "
  + "residual attributable to a variable the tensor cannot see. (c) The "
  + "scalar spread in two and three dimensions, with crack density staged by "
  + "its value at localization so the two sweeps are commensurate. (d) The "
  + "tensor test in three dimensions, where the tensor has six independent "
  + "components; the two-dimensional bound is quoted for comparison."
).forEach((x) => cuerpo.push(x));

cuerpo.push(h("3.4 What the closures in use actually cost", HeadingLevel.HEADING_2));
cuerpo.push(p(
  "Those two bounds frame the closures in service. The two we test are the "
  + "exponential damage law and a two-parameter linear tensor law:",
  { spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 } }));
eqp(10).forEach((x) => cuerpo.push(x));
cuerpo.push(pr([
  ["We fit the exponential law to the entire reference by least squares in "
   + "the logarithm, granting it its best exponent rather than a literature "
   + "value. The optimum, α = 6.7, sits inside the range of 5 to 25 used in "
   + "practice, and still errs by 57.1% in the median (63.7% at α = 5). "
   + "Figure 4 shows the structure of the failure: underprediction by about "
   + "60% for 0.1 < tr D < 0.3, then a crossover to 38% overprediction "
   + "beyond 0.3 and a factor of 2.3 at the largest damage. The failure is "
   + "one of functional form, since the reference grows sub-linearly in "
   + "tr ", ""],
  ["D", "ib"], [" (power-law exponent 0.88), not exponentially.", ""],
]));
cuerpo.push(pr([
  ["The linear tensor closure of Equation 10", ""],
  [", fitted to the reference over 0.15 < tr ", ""], ["D", "ib"],
  [" ≤ 0.405, gives a median error of 1.3% and a maximum of 11.8%, with "
   + "K", ""],
  ["0", "_"], [" = 11.0, ", ""], ["a", "i"], [" = 362 and ", ""],
  ["b", "i"], [" = −150 in matrix-permeability units. The negative sign on "
   + "the tensor term reproduces the projection structure of Oda–Snow, in "
   + "which flow runs along the crack plane and not across it, at less than "
   + "half the strength of the pure projection. Extended to tr D = 0.63 the "
   + "same form degrades to 8.1%: since the matched-tensor bound stays at 3% "
   + "there, this is model-form error in the linear ansatz, not path memory, "
   + "and a better function of the same tensor exists. Dropping the tensor "
   + "term roughly doubles the error below tr D = 0.4; the gain is bounded "
   + "by the modest anisotropy of this two-dimensional cell, and should be "
   + "larger in three dimensions.", ""],
]));

figura("fig4_cierres.png", 400, 273,
  "Figure 4. ",
  "Error of three closures against the reference, over all 210 states. The "
  + "exponential damage law is shown at its own best-fit exponent, not at a "
  + "literature value; the two linear closures are fitted over the whole "
  + "sweep. The vertical axis is symmetric-logarithmic with a "
  + "linear band below 10%."
).forEach((x) => cuerpo.push(x));

cuerpo.push(h("3.5 The same two bounds hold in three dimensions", HeadingLevel.HEADING_2));
cuerpo.push(p(
  "Both tests were repeated on a 96³ cell. Localization is later in three "
  + "dimensions but identical in kind: the load factor peaks at Γ = 0.855 "
  + "against 0.375, the maximum of the phase field jumps from 0.70 to 0.97 "
  + "in one increment, and the largest cluster then holds 97% of the damaged "
  + "volume. The sweeps are commensurate only when staged by the ratio of "
  + "crack density to its value at localization (Figure 3c)."));
cuerpo.push(p(
  "The scalar floor behaves the same way in both dimensions: below 3% while "
  + "damage is diffuse, then steep once a crack exists, reaching 14.5% at "
  + "the last state common to the five paths, 1.14 times its value at "
  + "localization, where two "
  + "dimensions give 21%. The tensor test survives the added components: "
  + "nineteen cross-path pairs match the six-component tensor to within 2% "
  + "and agree in permeability to 0.61% at worst and 0.22% in the median, so "
  + "the scalar floor exceeds the tensor bound by 24 times here against 15 "
  + "in two dimensions (Figure 3d). Path memory reproduces with a wider "
  + "margin: at the last common state the sub-dominant excursion departs by "
  + "0.007% and the dominant one by 3.50%, growing through localization. No "
  + "three-dimensional state percolates either."));
cuerpo.push(p(
  "In three dimensions one exclusion is needed that two dimensions did not "
  + "require: once the sub-dominant excursion is erased, P4 and P1 agree to "
  + "0.02% in the tensor, so pairs drawn from them are one history, not two, "
  + "and are excluded."));

cuerpo.push(h("3.6 Aperture-based closures do not apply before percolation",
              HeadingLevel.HEADING_2));
cuerpo.push(p(
  "Reservoir practice more often writes the closure on aperture than on "
  + "damage. The two standards are the scalar cubic law and the Oda-Snow "
  + "tensor, the latter built from the aperture-cubed moment of the crack "
  + "density:", { spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 } }));
eqp(11).forEach((x) => cuerpo.push(x));
cuerpo.push(p(
  "Both need an aperture measured from the displacement field rather than "
  + "postulated. We take the crack volume and the mean aperture as",
  { spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 } }));
eqp(12).forEach((x) => cuerpo.push(x));
cuerpo.push(pm([
  "and verify them against Sneddon's plane-strain solution for a straight "
  + "crack of half-length a under remote tension σ, refining the mesh with ",
  { eq: "l_h" },
  " held fixed so that the crack stays resolved:",
]));
eqp(13).forEach((x) => cuerpo.push(x));
cuerpo.push(pm([
  "The measured ratio extrapolates to 1.003 as ",
  { eq: "l_a" }, " → 0, and the aperture-cubed moment, which is what the "
  + "cubic law weights, agrees to within 5% for ", { eq: "l_a" },
  " ≤ 0.12, a range that brackets the sweep.",
]));
cuerpo.push(p(
  "Judging those closures against the Darcy reference would compare two flow "
  + "models rather than two descriptors, so we solve a second, "
  + "aperture-consistent problem on the same geometry. The conductivity is "
  + "the cubic-law transmissivity spread over the thickness of the band "
  + "where φ exceeds one half, which for the AT2 profile is w = 2ℓ ln 2:",
  { spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 } }));
eqp(14).forEach((x) => cuerpo.push(x));
cuerpo.push(p(
  "Λ is the only group that enters and it is physical: for a 10 cm cell and "
  + "a 1 mD matrix, Λ = 10¹³, and a formed crack then gives a local contrast "
  + "of 10⁴, the value the Darcy surrogate uses. Against this reference "
  + "Oda-Snow overpredicts by three to six orders of magnitude and misses "
  + "the directional ratio by a factor of two on the uniaxial path and "
  + "eight on the shear path."));
cuerpo.push(p(
  "Two features of the sweep set the scope. Aperture closures are undefined "
  + "over most of it: no state below tr D = 0.375 has any phase above the "
  + "half-damage level, "
  + "so 63% of the 212 states admit no aperture, while permeability has "
  + "already risen a hundredfold. And where they are defined the "
  + "overprediction is not a constant: it runs from 10² to 3 × 10⁶, median "
  + "1.2 × 10⁵, and grows with damage. Fitting the cracked states gives a "
  + "one-parameter correction,", { spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 } }));
eqp(15).forEach((x) => cuerpo.push(x));
cuerpo.push(p(
  "with a median residual of 16%. The informative number is the exponent: "
  + "were Oda-Snow merely miscalibrated, m would be 1. That m = 0.26 is the "
  + "measure of the miss, because a moment of aperture cannot see whether "
  + "the crack spans the cell. Aperture closures are closures for connected "
  + "networks, which leaves the pre-percolation regime to a damage tensor."));

// --- 4 implications ---
cuerpo.push(h("4 Implications"));
cuerpo.push(p(
  "For simulation practice the results separate cleanly: the variable is not "
  + "the problem, the function is. Replacing an exponential scalar law by a "
  + "linear tensor law cuts the median error from 57% to 8% over the whole "
  + "sweep and to about 1% below tr D = 0.4, at the cost of two constants "
  + "and of carrying a tensor that anisotropic damage codes already carry. "
  + "No amount of recalibration rescues a scalar closure: its floor rises "
  + "with damage and reaches 43% here."));
cuerpo.push(p(
  "The second implication concerns how the tensor is obtained. Because "
  + "permeability follows the tensor faithfully, a prediction is only as "
  + "good as the tensor it is given, and the path-memory results show what "
  + "that demands: two histories at one crack density can carry tensors that "
  + "differ enough to move permeability by 15%. A tensor inferred from "
  + "porosity, or advanced by a law calibrated on monotonic loading, will "
  + "not reproduce that. The mechanics has to be in the loop."));
cuerpo.push(p(
  "The third is where this work sits relative to the roughness literature. "
  + "Perez et al. (2025) corrected the cubic law and Darcy for a static "
  + "rough fracture by Bayesian inference on the aperture field: the "
  + "complementary error, since they hold the geometry fixed and ask what "
  + "the flow model gets wrong while we hold the flow model fixed and ask "
  + "what the descriptor gets wrong. The two corrections compose."));

// --- 5 limitations ---
cuerpo.push(h("5 Limitations"));
cuerpo.push(p(
  "Four limits bound the claims. Each sweep uses one microstructural "
  + "realization, so the 43.5% and 3.0% figures are this cell's values, "
  + "though the separation between the two tests is robust to that since "
  + "both are internal comparisons. The three-dimensional sweep is truncated "
  + "at 1.14 Γ_loc, where the staggered solve reaches its iteration cap on "
  + "three paths. Percolation is reached in neither sweep, and the "
  + "sufficiency of the tensor is asserted only below it — precisely where a "
  + "topological variable would be expected to escape a second-order "
  + "descriptor."));
cuerpo.push(p(
  "Third, permeability is computed at the Darcy level with a local "
  + "conductivity interpolated in the phase field, not by solving Stokes "
  + "flow through an open fracture of physical aperture, so absolute "
  + "permeabilities are surrogate values and only the internal comparisons "
  + "are claimed. Fourth, the damage model is AT2 with a "
  + "maximum-based history field; the erasure of sub-dominant excursions is "
  + "a property of that irreversibility condition, and a model with a "
  + "different memory structure would behave differently."));

// --- 6 conclusions ---
cuerpo.push(h("6 Conclusions"));
cuerpo.push(p(
  "A coupled phase-field and FFT-homogenization reference, run over five "
  + "loading paths on one microstructure, bounds two families of "
  + "damage-permeability closure without any fitting. Any closure written as "
  + "a function of scalar crack density carries an irreducible error that "
  + "grows with damage and reaches 43.5%. Any closure written as a function "
  + "of the second-order damage tensor is not similarly bounded: matched "
  + "tensors give permeabilities agreeing to 3.0%, with unit-slope scaling "
  + "and no residual floor, so the tensor is a sufficient state variable "
  + "before percolation. Load-path memory is real — a dominant "
  + "non-proportional excursion shifts permeability by 13 to 14% at matched crack "
  + "density — but it acts on permeability only through the tensor, and a "
  + "sub-dominant excursion is erased entirely. The exponential damage law "
  + "in common use, given its best-fit exponent, still errs by 57% in the "
  + "median, while a two-parameter linear tensor closure fitted to the same "
  + "data errs by 8% over the whole sweep and about 1% below tr D = 0.4. A "
  + "96³ sweep reproduces all three findings at the damage it reaches: a "
  + "scalar floor of 14.5% against a tensor bound of 0.61%, and a "
  + "sub-dominant excursion erased to 0.007% while a dominant one holds at "
  + "3.50%. Aperture-based closures, tested against an aperture-consistent "
  + "reference, are not an alternative in this regime: before percolation "
  + "Oda-Snow overpredicts by orders of magnitude, because a moment of "
  + "aperture cannot see connectivity."));

// --- open research ---
cuerpo.push(h("Open Research"));
cuerpo.push(p(
  "The solver, the analysis scripts and the complete state tables for the "
  + "two-dimensional and three-dimensional sweeps are archived under a DOI "
  + "assigned on acceptance. The "
  + "code depends only on NumPy and SciPy; all figures in this paper are "
  + "regenerated by a single script from the archived state table."));

cuerpo.push(h("Acknowledgments"));
cuerpo.push(p(
  "The computations were run on the high-performance computing facility of "
  + "Universidad Industrial de Santander. The authors declare no competing "
  + "financial interests."));

// --- references ---
cuerpo.push(h("References"));
[
  "Amor, H., Marigo, J.-J., & Maurini, C. (2009). Regularized formulation of "
  + "the variational brittle fracture with unilateral contact: Numerical "
  + "experiments. Journal of the Mechanics and Physics of Solids, 57(8), "
  + "1209–1229. https://doi.org/10.1016/j.jmps.2009.04.011",

  "Aranda, P., & Segurado, J. (2025). A crack-length control technique for "
  + "phase-field fracture in FFT homogenization. International Journal for "
  + "Numerical Methods in Engineering, 126(2), e7664. "
  + "https://doi.org/10.1002/nme.7664",

  "Bourdin, B., Francfort, G. A., & Marigo, J.-J. (2000). Numerical "
  + "experiments in revisited brittle fracture. Journal of the Mechanics and "
  + "Physics of Solids, 48(4), 797–826. "
  + "https://doi.org/10.1016/S0022-5096(99)00028-9",

  "Bourdin, B., Francfort, G. A., & Marigo, J.-J. (2008). The variational "
  + "approach to fracture. Journal of Elasticity, 91(1–3), 5–148. "
  + "https://doi.org/10.1007/s10659-007-9107-3",

  "Chen, Y. (2023). High-performance computational homogenization of "
  + "Stokes–Brinkman flow with an Anderson-accelerated FFT method. "
  + "International Journal for Numerical Methods in Fluids, 95(9), "
  + "1441–1467. https://doi.org/10.1002/fld.5199",

  "Kachanov, M. (1993). Elastic solids with many cracks and related problems. "
  + "Advances in Applied Mechanics, 30, 259–445. "
  + "https://doi.org/10.1016/S0065-2156(08)70176-5",

  "Li, L., & Li, K. (2015). Permeability of microcracked solids with random "
  + "crack networks: Role of connectivity and opening aperture. Transport in "
  + "Porous Media, 109(1), 217–237. "
  + "https://doi.org/10.1007/s11242-015-0510-0",

  "Liu, C., Song, Z., Zhang, D., & Zhao, H. (2021). Mechanical response of "
  + "permeability evolution to anisotropic structure of reservoir rock under "
  + "true triaxial stress path. Geomechanics and Geophysics for Geo-Energy "
  + "and Geo-Resources, 7(3), 59. https://doi.org/10.1007/s40948-021-00249-2",

  "Miehe, C., Hofacker, M., & Welschinger, F. (2010). A phase field model for "
  + "rate-independent crack propagation: Robust algorithmic implementation "
  + "based on operator splits. Computer Methods in Applied Mechanics and "
  + "Engineering, 199(45–48), 2765–2778. "
  + "https://doi.org/10.1016/j.cma.2010.04.011",

  "Moulinec, H., & Suquet, P. (1998). A numerical method for computing the "
  + "overall response of nonlinear composites with complex microstructure. "
  + "Computer Methods in Applied Mechanics and Engineering, 157(1–2), "
  + "69–94. https://doi.org/10.1016/S0045-7825(97)00218-1",

  "Oda, M. (1982). Fabric tensor for discontinuous geological materials. "
  + "Soils and Foundations, 22(4), 96–108. "
  + "https://doi.org/10.3208/sandf1972.22.4_96",

  "Oda, M. (1985). Permeability tensor for discontinuous rock masses. "
  + "Géotechnique, 35(4), 483–495. "
  + "https://doi.org/10.1680/geot.1985.35.4.483",

  "Perez, S., Doster, F., Maes, J., Menke, H., ElSheikh, A., & Busch, A. "
  + "(2025). When cubic law and Darcy fail: Bayesian correction of model "
  + "misspecification in fracture conductivities. Geophysical Research "
  + "Letters, 52(18), e2025GL117776. https://doi.org/10.1029/2025GL117776",

  "Ren, S., Liang, W., Chen, Y., Wang, Z., Luo, H., & Zou, L. (2026). "
  + "Experimental study on damage and permeability evolution in coal measure "
  + "rocks under variable deviatoric stress paths. Rock Mechanics and Rock "
  + "Engineering. Advance online publication. "
  + "https://doi.org/10.1007/s00603-026-05901-5",

  "Shao, J. F., Zhou, H., & Chau, K. T. (2005). Coupling between anisotropic "
  + "damage and permeability variation in brittle rocks. International "
  + "Journal for Numerical and Analytical Methods in Geomechanics, 29(12), "
  + "1231–1247. https://doi.org/10.1002/nag.457",

  "Snow, D. T. (1969). Anisotropic permeability of fractured media. Water "
  + "Resources Research, 5(6), 1273–1289. "
  + "https://doi.org/10.1029/WR005i006p01273",

  "Souley, M., Homand, F., Pepa, S., & Hoxha, D. (2001). Damage-induced "
  + "permeability changes in granite: A case example at the URL in Canada. "
  + "International Journal of Rock Mechanics and Mining Sciences, 38(2), "
  + "297–310. https://doi.org/10.1016/S1365-1609(01)00002-8",

  "Vondřejc, J., Zeman, J., & Marek, I. (2014). An FFT-based Galerkin method "
  + "for homogenization of periodic media. Computers & Mathematics with "
  + "Applications, 68(3), 156–173. "
  + "https://doi.org/10.1016/j.camwa.2014.05.014",

  "Wang, Z., Li, W., & Hu, Y. (2021). Experimental study on mechanical "
  + "behavior, permeability, and damage characteristics of Jurassic sandstone "
  + "under varying stress paths. Bulletin of Engineering Geology and the "
  + "Environment, 80(6), 4423–4439. "
  + "https://doi.org/10.1007/s10064-021-02214-5",

  "Willot, F. (2015). Fourier-based schemes for computing the mechanical "
  + "response of composites with accurate local fields. Comptes Rendus "
  + "Mécanique, 343(3), 232–245. "
  + "https://doi.org/10.1016/j.crme.2014.12.005",

  "Zhang, C., Bai, Q., & Chen, Y. (2020). Using stress path-dependent "
  + "permeability law to evaluate permeability enhancement and coalbed "
  + "methane flow in protected coal seam: A case study. Geomechanics and "
  + "Geophysics for Geo-Energy and Geo-Resources, 6(3), 53. "
  + "https://doi.org/10.1007/s40948-020-00177-7",

  "Zhou, C., Li, K., & Pang, X. (2011). Effect of crack density and "
  + "connectivity on the permeability of microcracked solids. Mechanics of "
  + "Materials, 43(12), 969–978. "
  + "https://doi.org/10.1016/j.mechmat.2011.08.011",
].forEach((t) => cuerpo.push(ref(t)));

// ------------------------------------------------------------------- doc --
const doc = new Document({
  creator: "German Orlando Romero Suárez; Diego Fernando Villegas Bermúdez; "
         + "Wilmer Velilla Díaz",
  title: "A Second-Order Damage Tensor Is Sufficient for Damage-Induced "
       + "Permeability; Scalar Crack Density Is Not",
  description: "Manuscript prepared for Geophysical Research Letters.",
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 24 } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },          // US Letter, en DXA
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
      orientation: PageOrientation.PORTRAIT,
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 20,
                                   font: "Times New Roman" })],
        })],
      }),
    },
    children: cuerpo,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const salida = path.join(RAIZ, "manuscrito_GRL.docx");
  fs.writeFileSync(salida, buf);
  console.log("->", salida);
});
