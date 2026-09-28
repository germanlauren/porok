// Resumen ejecutivo en español, para ANH / Ecopetrol / UIS.
//
// No es una traducción del manuscrito. El lector aquí decide si el trabajo
// merece respaldo, no si el método es correcto: va primero qué cambia, después
// con qué evidencia, y al final qué falta. Las cifras son las mismas del
// manuscrito, sin redondear hacia arriba.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  ImageRun, Footer, PageNumber, LineRuleType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle,
} = require("docx");

const RAIZ = path.resolve(__dirname, "..");
const FIG = path.join(RAIZ, "figuras_fd");
const INTER = { line: 300, lineRule: LineRuleType.AUTO, after: 140 };

function p(texto, opts = {}) {
  return new Paragraph({
    spacing: opts.spacing || INTER,
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    indent: opts.indent,
    children: [new TextRun({ text: texto, size: 22, font: "Calibri",
                             italics: opts.italics, bold: opts.bold,
                             color: opts.color })],
  });
}

function vinieta(texto) {
  return new Paragraph({
    spacing: INTER,
    alignment: AlignmentType.JUSTIFIED,
    numbering: { reference: "puntos", level: 0 },
    children: [new TextRun({ text: texto, size: 22, font: "Calibri" })],
  });
}

function h(texto, nivel = HeadingLevel.HEADING_1) {
  return new Paragraph({
    heading: nivel,
    spacing: { before: 280, after: 120 },
    children: [new TextRun({ text: texto, size: nivel === HeadingLevel.HEADING_1 ? 26 : 23,
                             bold: true, font: "Calibri", color: "1F3864" })],
  });
}

function celda(texto, opts = {}) {
  return new TableCell({
    width: { size: opts.ancho, type: WidthType.DXA },
    shading: opts.fondo ? { type: ShadingType.CLEAR, fill: opts.fondo } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      spacing: { line: 260, lineRule: LineRuleType.AUTO },
      alignment: opts.alignment,
      children: [new TextRun({ text: texto, size: 20, font: "Calibri",
                               bold: opts.bold, color: opts.color })],
    })],
  });
}

const ANCHOS = [3400, 2000, 3600];

function fila(celdas, opts = {}) {
  return new TableRow({
    children: celdas.map((t, i) => celda(t, {
      ancho: ANCHOS[i], fondo: opts.fondo, bold: opts.bold, color: opts.color,
      alignment: i === 1 ? AlignmentType.CENTER : undefined,
    })),
  });
}

const cuerpo = [];

cuerpo.push(new Paragraph({
  spacing: { after: 80 },
  children: [new TextRun({
    text: "Permeabilidad de roca dañada: qué tan lejos está la práctica actual "
        + "de la respuesta correcta",
    size: 32, bold: true, font: "Calibri", color: "1F3864" })],
}));
cuerpo.push(p("German Orlando Romero Suárez y Diego Fernando Villegas "
              + "Bermúdez, Universidad Industrial de Santander · "
              + "Wilmer Velilla Díaz, Universidad de La Serena",
              { spacing: { after: 40 } }));
cuerpo.push(p("Resumen ejecutivo · septiembre de 2026 · contacto: "
              + "german2218069@correo.uis.edu.co",
              { italics: true, color: "595959", spacing: { after: 240 } }));

// ---------------------------------------------------------------- el punto --
cuerpo.push(h("En una frase"));
cuerpo.push(p(
  "Construimos una referencia numérica que permite hacer un experimento "
  + "imposible en laboratorio —llevar la misma roca al mismo nivel de daño por "
  + "caminos de carga distintos— y con ella medimos cuánto se equivocan las "
  + "reglas que la industria usa hoy para convertir daño mecánico en "
  + "permeabilidad. El error de la regla más común es del 57% en la mediana; "
  + "una regla alternativa que usa información que los simuladores ya cargan "
  + "baja ese error al 8% en todo el rango, y a cerca de 1% por debajo de "
  + "un daño moderado."));

cuerpo.push(h("Por qué importa para el país"));
cuerpo.push(p(
  "La meta de recuperar autosuficiencia en gas y petróleo descansa sobre "
  + "yacimientos que ya no son fáciles: naturalmente fracturados, apretados, o "
  + "sometidos a estimulación. En todos ellos el flujo lo gobierna una red de "
  + "grietas que la historia de esfuerzos va creando durante la depletación. "
  + "Los simuladores no pueden resolver esa red grieta por grieta, así que la "
  + "resumen en una variable de daño y leen la permeabilidad de una función de "
  + "esa variable. Esa función es la que nadie había podido auditar, porque "
  + "para auditarla hay que comparar dos rocas dañadas igual pero por caminos "
  + "distintos, y en un núcleo real eso no se puede preparar."));
cuerpo.push(p(
  "El mismo cálculo es la base técnica del almacenamiento geológico de CO₂ "
  + "y de la evaluación de sellos, donde el error no se paga en producción sino "
  + "en riesgo de fuga."));

// ---------------------------------------------------------------- hallazgos --
cuerpo.push(h("Los tres resultados"));

cuerpo.push(h("1. La variable escalar que se usa hoy no alcanza, y no es "
              + "cuestión de calibrar mejor", HeadingLevel.HEADING_2));
cuerpo.push(p(
  "Una regla escalar afirma que la permeabilidad es función de un solo número "
  + "—la densidad de grieta—. Eso la obliga a dar el mismo resultado para dos "
  + "estados con ese número igual. Medimos cuánto difieren de verdad esos "
  + "estados: hasta 43,5%, y la diferencia crece con el daño. Ese es un piso: "
  + "ninguna forma funcional, ninguna constante ajustada, ningún parámetro "
  + "adicional puede bajarlo. La regla es más confiable justo donde el daño no "
  + "importa, y menos confiable donde sí gobierna el flujo."));

cuerpo.push(h("2. El tensor de daño sí alcanza — y es el resultado útil",
              HeadingLevel.HEADING_2));
cuerpo.push(p(
  "Repetimos la prueba con el tensor de daño completo, que además de cuánto "
  + "daño hay registra en qué dirección. Entre los 59 pares de estados de "
  + "trayectorias distintas cuyo tensor coincide dentro del 2%, la "
  + "permeabilidad difiere a lo sumo 2,97%, con mediana de 0,21%. Y no hay "
  + "piso residual: al reducir el desajuste del tensor, la diferencia de "
  + "permeabilidad baja proporcionalmente durante tres décadas. Si existiera "
  + "una variable oculta que el tensor no ve, aparecería como una meseta. No "
  + "aparece."));
cuerpo.push(p(
  "Esto contradice la hipótesis con la que arrancamos —esperábamos que la "
  + "topología de la red venciera también al tensor— y por eso es más valioso: "
  + "significa que la variable que los simuladores anisótropos ya llevan es, "
  + "en principio, suficiente. El error está en la fórmula que se le cuelga "
  + "encima, no en la variable.", { bold: true }));

cuerpo.push(h("3. La historia de carga importa, pero de manera selectiva",
              HeadingLevel.HEADING_2));
cuerpo.push(p(
  "Comparamos una carga monótona contra dos historias que rotan 90° la "
  + "dirección de carga a mitad de camino. Cuando la excursión previa es menor "
  + "que la carga final, el material la olvida por completo: la diferencia cae "
  + "de 5,3% a 0,002%, más de mil veces menos, es decir, "
  + "las dos historias se vuelven indistinguibles. Cuando la excursión domina, "
  + "no la olvida: se queda en una meseta de 13 a 14% "
  + "hasta el final. Esto explica por qué los efectos de trayectoria de esfuerzo se "
  + "reportan de forma inconsistente en laboratorio, y advierte que una regla "
  + "calibrada solo con ensayos monótonos parecerá adecuada en muchos casos y "
  + "fallará en el resto sin previo aviso."));

// ---------------------------------------------------------------- tabla --
cuerpo.push(h("Qué cuesta cada regla, en números"));
cuerpo.push(new Table({
  columnWidths: ANCHOS,
  width: { size: ANCHOS.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  borders: {
    top: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" },
    bottom: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" },
    left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
    insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "E0E0E0" },
    insideVertical: { style: BorderStyle.NONE },
  },
  rows: [
    fila(["Regla", "Error mediano", "Observación"],
         { fondo: "1F3864", bold: true, color: "FFFFFF" }),
    fila(["Exponencial de daño, k = k₀ e^(αD), con su mejor "
          + "exponente posible", "57%",
          "Subestima ~60% con daño moderado y sobreestima ~40% con daño alto"]),
    fila(["La misma, con el valor típico de literatura (α = 5)", "64%",
          "El problema es la forma exponencial, no la calibración"]),
    fila(["Cualquier regla escalar, sea cual sea su forma", "piso de 43,5%",
          "Cota inferior medida, no estimada"]),
    fila(["Tensorial lineal de dos parámetros", "8% (1% con daño moderado)",
          "Usa el tensor que los códigos anisótropos ya llevan"]),
    fila(["Cota de lo alcanzable con el tensor", "3,0%",
          "Lo que la ley lineal pierde con daño alto es forma funcional, no memoria de camino"]),
  ],
}));
cuerpo.push(p(""));

// ---------------------------------------------------------------- figura --
cuerpo.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 80 },
  children: [new ImageRun({
    data: fs.readFileSync(path.join(FIG, "fig3_escalar_vs_tensor.png")),
    type: "png", transformation: { width: 460, height: 368 } })],
}));
cuerpo.push(p(
  "Arriba: dispersión de la permeabilidad entre estados que comparten "
  + "densidad de grieta (izquierda); es el error irreducible de cualquier "
  + "regla escalar, y crece con el daño. A la derecha, la misma prueba con el "
  + "tensor completo; los puntos verdes son los pares con tensor igualado "
  + "dentro del 2%. Abajo: las mismas dos pruebas en el barrido "
  + "tridimensional de 96³.",
  { spacing: { after: 240 }, italics: true }));

// ---------------------------------------------------------------- práctica --
cuerpo.push(h("Qué cambia en la práctica"));
[
  "Cambiar la ley exponencial escalar por una ley tensorial lineal reduce el "
  + "error mediano de 57% a 8%, con dos constantes de calibración y sin "
  + "cambiar la variable de estado que el simulador ya transporta.",
  "El tensor de daño tiene que evolucionar con la mecánica. Inferirlo de la "
  + "porosidad, o avanzarlo con una ley ajustada solo a carga monótona, no "
  + "reproduce las diferencias de 13 a 14% que dejan las trayectorias no "
  + "proporcionales.",
  "Para caracterización de laboratorio: los ensayos monótonos no bastan para "
  + "calibrar. Hay que incluir al menos una trayectoria con rotación de "
  + "dirección de carga, y que la excursión sea dominante, porque una "
  + "subdominante se borra y no informa nada.",
].forEach((t) => cuerpo.push(vinieta(t)));

// ---------------------------------------------------------------- honesto --
cuerpo.push(h("Confirmación en tres dimensiones"));
cuerpo.push(p(
  "Las dos pruebas se repitieron en una celda de 96³, con las mismas cinco "
  + "trayectorias. La grieta se forma más tarde que en 2D (Γ = 0,855 frente "
  + "a 0,375), pero de la misma manera. En el último estado común a las cinco "
  + "trayectorias, el piso escalar es de 14,5% y la cota tensorial de 0,61%: "
  + "el tensor sirve 24 veces mejor que el escalar, frente a 15 veces en 2D. "
  + "La memoria de camino también se repite y con más margen: la excursión "
  + "subdominante se borra hasta 0,007% y la dominante se queda en 3,50%."));
cuerpo.push(p(
  "El barrido 3D está truncado: pasado el punto de localización, tres de las "
  + "cinco trayectorias agotan el tope de iteraciones del esquema escalonado "
  + "y cada incremento cuesta unas cinco horas, así que cubre el inicio del "
  + "agrietamiento y no todo el rango de ablandamiento que sí cubre el 2D."));

cuerpo.push(h("Ley cúbica y Oda-Snow: dónde sí y dónde no"));
cuerpo.push(p(
  "Los cierres que la industria escribe sobre APERTURA, y no sobre daño, son "
  + "la ley cúbica y el tensor de Oda-Snow. Para poder juzgarlos hubo que "
  + "medir la apertura del campo de desplazamiento y verificarla contra la "
  + "solución clásica de Sneddon: el volumen de grieta converge con error "
  + "1.8 (l/a) y extrapola a 1.003, y el momento de apertura al cubo, que es "
  + "el que pesa la ley cúbica, queda dentro del 5%."));
cuerpo.push(p(
  "Comparados contra una referencia construida con su propia física — "
  + "conductividad local dada por la apertura al cubo — Oda-Snow sobrepredice "
  + "entre tres y seis órdenes de magnitud, y se equivoca en la dirección por "
  + "un factor de 2 en carga uniaxial y de 8 en cizalla. La razón es la "
  + "CONECTIVIDAD: ninguna de las grietas del barrido atraviesa la celda, y "
  + "una grieta que no atraviesa no puede conducir lo que sugiere la suma de "
  + "sus aperturas al cubo. En otras palabras, los cierres de apertura son "
  + "cierres para redes ya conectadas. El régimen previo a la percolación, "
  + "donde el daño ya es medible y la permeabilidad ya está cambiando, es "
  + "justamente el que cubre un cierre por tensor de daño."));
cuerpo.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 120 },
  children: [new ImageRun({
    data: fs.readFileSync(path.join(RAIZ, "figuras_eq", "eq15.png")),
    type: "png", transformation: { width: 405, height: 43 } })],
}));
cuerpo.push(p(
  "Factor de corrección calibrado contra la referencia: con él, el error "
  + "mediano de Oda-Snow baja al 16%.",
  { italics: true, spacing: { after: 200 } }));
cuerpo.push(p(
  "Sobre el barrido completo, 212 estados y cinco trayectorias, el mapa "
  + "queda así: en el 63% de los estados la ley cúbica y Oda-Snow ni "
  + "siquiera están definidas, porque todavía no hay grieta abierta, "
  + "mientras la permeabilidad real ya subió cien veces sobre la de la "
  + "matriz. Donde sí están definidas, la sobrepredicción va de 10² a "
  + "3×10⁶, con mediana 1.2×10⁵, y crece con el daño. Un factor de "
  + "corrección de un solo parámetro, K_b − 1 proporcional a "
  + "(Λ K_Oda) elevado a 0.26, deja un error mediano del 16%; que el "
  + "exponente sea 0.26 y no 1 es la medida de cuánto se pierde por ignorar "
  + "la conectividad."));

cuerpo.push(h("Qué todavía no podemos afirmar"));
cuerpo.push(p(
  "Cada barrido usa una sola realización de la "
  + "microestructura, así que las cifras son de esta celda; la separación "
  + "cualitativa entre lo escalar y lo tensorial sí es robusta, porque ambas "
  + "pruebas son comparaciones internas. Ningún estado de ninguno de los dos "
  + "barridos alcanza percolación, y la suficiencia del tensor se afirma solo "
  + "antes de percolar — es justamente ahí donde cabe esperar que una "
  + "variable topológica se escape. La permeabilidad "
  + "se calcula a nivel de Darcy, no resolviendo Stokes en una fractura "
  + "abierta con apertura física; la verificación de nuestro campo de apertura "
  + "contra la solución de Sneddon sigue abierta, y por eso no reportamos el "
  + "tensor de Oda-Snow, que la requiere."));

cuerpo.push(h("Estado y siguiente paso"));
cuerpo.push(p(
  "El manuscrito está escrito en formato de Geophysical Research Letters "
  + "(carta corta, factor de impacto en el primer cuartil de geociencias) con "
  + "cuatro figuras, el barrido 2D de 210 estados y el 3D de 403. El código depende "
  + "solo de NumPy y SciPy, corre en el HPC disponible y es reanudable."));
cuerpo.push(p(
  "Lo que falta para cerrar la versión publicable, en orden de importancia: "
  + "(1) repetir el barrido 3D con varias semillas y llevarlo más allá de la "
  + "localización, lo que exige acelerar el esquema escalonado; (2) cerrar la verificación de "
  + "apertura y habilitar la comparación contra la ley cúbica y el tensor de "
  + "Oda-Snow; (3) verificación cruzada con FEniCSx en dos o tres puntos. El "
  + "cálculo está dimensionado: ocho procesos de cuatro hilos rinden unas "
  + "cinco veces más que un solo proceso de treinta y dos, porque la FFT "
  + "satura hacia los ocho hilos."));

// ------------------------------------------------------------------ doc --
const doc = new Document({
  numbering: {
    config: [{
      reference: "puntos",
      levels: [{
        level: 0, format: "bullet", text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 460, hanging: 240 } } },
      }],
    }],
  },
  styles: { default: { document: { run: { font: "Calibri", size: 22 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1300, bottom: 1300, left: 1300, right: 1300 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 18,
                                   font: "Calibri", color: "808080" })],
        })],
      }),
    },
    children: cuerpo,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const salida = path.join(RAIZ, "resumen_ejecutivo_ES.docx");
  fs.writeFileSync(salida, buf);
  console.log("->", salida);
});
