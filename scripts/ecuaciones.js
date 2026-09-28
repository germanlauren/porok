// Ecuaciones del manuscrito como OMML nativo de Word (no imagenes, no texto
// plano). Tres reglas que impone el encargo:
//
//   1. Toda division va con BARRA HORIZONTAL (MathFraction), nunca con "/".
//   2. Subindices y superindices son OMML de verdad (MathSubScript,
//      MathSuperScript, MathSubSuperScript), no caracteres Unicode.
//   3. Cada ecuacion va centrada y numerada a la derecha, con tabuladores,
//      que es como las maqueta AGU.
//
// Word renderiza esto con su propio motor matematico, asi que el autor puede
// editarlo despues sin volver a generarlo.

const {
  Paragraph, TextRun, AlignmentType, TabStopType, LineRuleType,
  Math: OMath, MathRun, MathFraction, MathSubScript, MathSuperScript,
  MathSubSuperScript, MathIntegral, MathRadical, MathRoundBrackets,
  MathSquareBrackets, MathAngledBrackets, MathSum,
} = require("docx");

// --- atomos ---------------------------------------------------------------
const r = (t) => new MathRun(t);
const frac = (num, den) => new MathFraction({
  numerator: Array.isArray(num) ? num : [num],
  denominator: Array.isArray(den) ? den : [den],
});
const sub = (base, s) => new MathSubScript({
  children: Array.isArray(base) ? base : [base],
  subScript: Array.isArray(s) ? s : [s],
});
const sup = (base, s) => new MathSuperScript({
  children: Array.isArray(base) ? base : [base],
  superScript: Array.isArray(s) ? s : [s],
});
const subsup = (base, sb, sp) => new MathSubSuperScript({
  children: Array.isArray(base) ? base : [base],
  subScript: Array.isArray(sb) ? sb : [sb],
  superScript: Array.isArray(sp) ? sp : [sp],
});
const par = (x) => new MathRoundBrackets({ children: Array.isArray(x) ? x : [x] });
const cor = (x) => new MathSquareBrackets({ children: Array.isArray(x) ? x : [x] });
const ang = (x) => new MathAngledBrackets({ children: Array.isArray(x) ? x : [x] });
const integral = (hijos, sb) => new MathIntegral({
  children: Array.isArray(hijos) ? hijos : [hijos],
  subScript: sb ? (Array.isArray(sb) ? sb : [sb]) : undefined,
});
const raiz = (x) => new MathRadical({ children: Array.isArray(x) ? x : [x] });

// --- una ecuacion numerada ------------------------------------------------
// Centro a 3.25", numero a la derecha del bloque de texto (6.5").
function ecuacion(hijos, numero) {
  return new Paragraph({
    spacing: { line: 480, lineRule: LineRuleType.AUTO, before: 120, after: 120 },
    tabStops: [
      { type: TabStopType.CENTER, position: 3.25 * 1440 },
      { type: TabStopType.RIGHT, position: 6.5 * 1440 },
    ],
    children: [
      new TextRun({ text: "\t", size: 24, font: "Times New Roman" }),
      new OMath({ children: hijos }),
      new TextRun({ text: "\t(" + numero + ")", size: 24,
                    font: "Times New Roman" }),
    ],
  });
}

module.exports = {
  r, frac, sub, sup, subsup, par, cor, ang, integral, raiz, ecuacion,
};
