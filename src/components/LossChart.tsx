interface Point {
  step: number
  loss: number
  psnr: number
}

interface Props {
  history: Point[]
}

const WIDTH = 520
const HEIGHT = 150
const PAD = { left: 40, right: 40, top: 12, bottom: 22 }

/**
 * Loss (log scale) and PSNR on one pair of axes. Log scale matters here: the
 * loss drops by two orders of magnitude in the first few hundred steps, and a
 * linear axis would flatline immediately.
 */
export function LossChart({ history }: Props) {
  if (history.length < 2) {
    return (
      <div className="chart-empty">
        Eğitim başladığında kayıp (loss) ve PSNR eğrisi burada belirecek.
      </div>
    )
  }

  const steps = history.map((h) => h.step)
  const minStep = steps[0]
  const maxStep = steps[steps.length - 1]
  const spanStep = Math.max(1, maxStep - minStep)

  const losses = history.map((h) => Math.max(h.loss, 1e-6))
  const minLoss = Math.min(...losses)
  const maxLoss = Math.max(...losses)
  const logMin = Math.log10(minLoss)
  const logMax = Math.log10(maxLoss)
  const logSpan = Math.max(1e-6, logMax - logMin)

  const psnrs = history.map((h) => h.psnr)
  const minPsnr = Math.min(...psnrs)
  const maxPsnr = Math.max(...psnrs)
  const psnrSpan = Math.max(1e-6, maxPsnr - minPsnr)

  const plotW = WIDTH - PAD.left - PAD.right
  const plotH = HEIGHT - PAD.top - PAD.bottom
  const x = (step: number) => PAD.left + ((step - minStep) / spanStep) * plotW
  const yLoss = (loss: number) =>
    PAD.top + plotH - ((Math.log10(Math.max(loss, 1e-6)) - logMin) / logSpan) * plotH
  const yPsnr = (psnr: number) => PAD.top + plotH - ((psnr - minPsnr) / psnrSpan) * plotH

  const lossPath = history.map((h, i) => `${i === 0 ? 'M' : 'L'}${x(h.step)},${yLoss(h.loss)}`).join(' ')
  const psnrPath = history.map((h, i) => `${i === 0 ? 'M' : 'L'}${x(h.step)},${yPsnr(h.psnr)}`).join(' ')
  const last = history[history.length - 1]

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Eğitim kayıp ve PSNR grafiği">
        <line
          x1={PAD.left} y1={PAD.top + plotH} x2={PAD.left + plotW} y2={PAD.top + plotH}
          className="chart-axis"
        />
        <path d={psnrPath} className="chart-line chart-line--psnr" />
        <path d={lossPath} className="chart-line chart-line--loss" />
        <text x={PAD.left} y={HEIGHT - 6} className="chart-tick">adım {minStep}</text>
        <text x={PAD.left + plotW} y={HEIGHT - 6} className="chart-tick" textAnchor="end">
          {maxStep}
        </text>
        <text x={4} y={PAD.top + 8} className="chart-tick chart-tick--loss">
          {maxLoss.toExponential(1)}
        </text>
        <text x={4} y={PAD.top + plotH} className="chart-tick chart-tick--loss">
          {minLoss.toExponential(1)}
        </text>
        <text x={WIDTH - 4} y={PAD.top + 8} className="chart-tick chart-tick--psnr" textAnchor="end">
          {maxPsnr.toFixed(0)} dB
        </text>
        <text
          x={WIDTH - 4} y={PAD.top + plotH} className="chart-tick chart-tick--psnr" textAnchor="end"
        >
          {minPsnr.toFixed(0)} dB
        </text>
      </svg>
      <figcaption>
        <span className="legend legend--loss">kayıp {last.loss.toExponential(2)}</span>
        <span className="legend legend--psnr">PSNR {last.psnr.toFixed(2)} dB</span>
      </figcaption>
    </figure>
  )
}
