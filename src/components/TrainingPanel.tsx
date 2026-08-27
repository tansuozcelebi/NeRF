import { CanvasImage } from './CanvasImage'
import { LossChart } from './LossChart'
import type { NerfWorkerApi } from '../hooks/useNerfWorker'
import { QUALITY_PRESETS, type QualityPreset } from '../nerf/types'

interface Props {
  nerf: NerfWorkerApi
  preset: QualityPreset
  onPresetChange: (preset: QualityPreset) => void
  /**
   * Re-prepares the model, which is required after a preset change. The preset
   * is passed explicitly because the caller's state update has not landed yet.
   */
  onRebuild: (preset?: QualityPreset) => void
}

export function TrainingPanel({ nerf, preset, onPresetChange, onRebuild }: Props) {
  const { status, stats, history, preview, previewStep, paramCount, viewCount, pointsPerStep } = nerf
  const training = status === 'egitiliyor'
  const ready = status === 'hazir' || training

  const stepsPerSecond = stats && stats.stepMs > 0 ? 1000 / stats.stepMs : 0

  return (
    <div className="training">
      <div className="training-controls">
        <div className="field">
          <label htmlFor="preset">Kalite ön ayarı</label>
          <select
            id="preset"
            value={preset}
            disabled={training}
            onChange={(e) => {
              const next = e.target.value as QualityPreset
              onPresetChange(next)
              onRebuild(next)
            }}
          >
            {Object.entries(QUALITY_PRESETS).map(([value, definition]) => (
              <option key={value} value={value}>
                {definition.label}
              </option>
            ))}
          </select>
          <p className="hint">{QUALITY_PRESETS[preset].description}</p>
        </div>

        <div className="button-row">
          <button
            type="button"
            className="btn btn--primary"
            disabled={!ready}
            onClick={() => (training ? nerf.pause() : nerf.start())}
          >
            {training ? 'Duraklat' : stats ? 'Devam et' : 'Eğitimi başlat'}
          </button>
          <button
            type="button"
            className="btn"
            disabled={training || !ready}
            onClick={() => onRebuild()}
          >
            Sıfırla ve yeniden kur
          </button>
        </div>
      </div>

      <p className="hint">
        Sonuçlar birkaç yüz adımda tanınır hâle gelir, 1000–3000 adım arasında belirgin şekilde
        keskinleşir. Eğitim arka planda sürerken diğer sekmelere geçebilirsiniz.
      </p>

      <dl className="stat-grid">
        <Stat label="Adım" value={stats ? stats.step.toLocaleString('tr') : '0'} />
        <Stat label="Kayıp (MSE)" value={stats ? stats.loss.toExponential(2) : '—'} />
        <Stat
          label="Eğitim PSNR"
          value={stats ? `${stats.psnr.toFixed(2)} dB` : '—'}
          hint="Modelin gördüğü karelerdeki keskinlik. Tek başına ezberlemeyi de gösterebilir."
        />
        <Stat
          label="Doğrulama PSNR"
          value={
            stats && stats.validationPsnr !== null ? `${stats.validationPsnr.toFixed(2)} dB` : '—'
          }
          hint="Eğitimde hiç kullanılmayan karelerdeki keskinlik. Asıl bakılması gereken sayı budur."
        />
        <Stat
          label="Hız"
          value={stepsPerSecond > 0 ? `${stepsPerSecond.toFixed(1)} adım/sn` : '—'}
        />
        <Stat
          label="Dolu hacim"
          value={stats ? `%${(stats.occupancy * 100).toFixed(0)}` : '—'}
          hint="Boş alan atlama ızgarasında dolu sayılan hücrelerin oranı"
        />
        <Stat
          label="Örnek/adım"
          value={pointsPerStep ? pointsPerStep.toLocaleString('tr') : '—'}
          hint="Ağın her adımda değerlendirdiği 3B nokta sayısı"
        />
        <Stat label="Parametre" value={paramCount.toLocaleString('tr')} />
        <Stat label="Eğitim görüntüsü" value={viewCount.toLocaleString('tr')} />
      </dl>

      <div className="training-bottom">
        <LossChart history={history} />
        <div className="live-preview">
          <CanvasImage
            image={preview}
            className="preview-canvas"
            placeholder="canlı önizleme"
            smooth={false}
          />
          <span className="hint">
            {preview ? `canlı önizleme · adım ${previewStep}` : 'eğitim başlayınca güncellenir'}
          </span>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="stat" title={hint}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}
