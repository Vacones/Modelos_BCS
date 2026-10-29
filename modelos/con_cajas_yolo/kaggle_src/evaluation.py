"""Métricas continuas y cobertura explícita; ningún test durante fit."""
import collections
import math
from pathlib import Path
from common import (artifacts, contract, digest, load, require, save, sha, stage, write_csv)


def metrics(records):
    eligible = [r for r in records if r.get('true_bcs') is not None]
    predicted = [r for r in eligible if r.get('pred_bcs') is not None]
    errors = [r['pred_bcs']-r['true_bcs'] for r in predicted]
    result = {'eligible_images': len(eligible), 'predicted_images': len(predicted),
              'abstained_images': len(eligible)-len(predicted),
              'coverage': len(predicted)/len(eligible) if eligible else None,
              'mae_predicted': sum(abs(e) for e in errors)/len(errors) if errors else None,
              'rmse_predicted': math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None}
    for tolerance in (.25, .5):
        count = sum(abs(e) <= tolerance for e in errors)
        result[f'within_{tolerance}_all'] = count/len(eligible) if eligible else None
        result[f'within_{tolerance}_predicted'] = count/len(predicted) if predicted else None
    return result


def report(cfg, records, tag):
    root = artifacts(cfg)
    dest = root / 'reports' / tag
    dest.mkdir(parents=True, exist_ok=True)
    save(dest / 'metrics.json', metrics(records))
    write_csv(dest / 'predictions.csv', records,
              ['key', 'cow_id', 'true_bcs', 'pred_bcs', 'status'])
    bycow = collections.defaultdict(list)
    for r in records:
        if r['true_bcs'] is not None:
            bycow[r['cow_id']].append(r)
    cows = [{'cow_id': cow, **metrics(items)} for cow, items in sorted(bycow.items())]
    write_csv(dest / 'by_cow.csv', cows, ['cow_id', *metrics([])])
    save(dest / 'cow_macro.json', {k: sum(c[k] for c in cows if c[k] is not None)/sum(c[k] is not None for c in cows)
                                  if any(c[k] is not None for c in cows) else None
                                  for k in ('coverage', 'mae_predicted', 'rmse_predicted', 'within_0.5_all')})
    predicted = [r for r in records if r['pred_bcs'] is not None and r['true_bcs'] is not None]
    if predicted:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter([r['true_bcs'] for r in predicted], [r['pred_bcs'] for r in predicted])
        axes[0].plot([1, 5], [1, 5], '--', color='gray')
        axes[0].set(xlabel='BCS referencia', ylabel='BCS predicho (continuo)')
        axes[1].hist([r['pred_bcs']-r['true_bcs'] for r in predicted])
        axes[1].set(xlabel='Error BCS', ylabel='Imágenes')
        fig.tight_layout()
        fig.savefig(dest / 'errors.png', dpi=150)
        plt.close(fig)
    print(tag, metrics(records))


def predict(cfg, split=None, external=False):
    from regressor import runtime, trained
    tf = runtime(cfg)
    root = artifacts(cfg)
    model_path = trained(cfg)
    model = tf.keras.models.load_model(model_path)
    samples = load(root / 'predicted' / ('external' if external else split) / 'predictions.json')
    from detector import trained as detector_trained
    settings = samples['settings']
    require(all(settings[k] == v for k, v in contract(cfg).items()), 'Recortes predichos de otros datos.')
    require(settings['detector_hash'] == sha(detector_trained(cfg)) and settings['inference'] == cfg['INFERENCE']
            and settings['size'] == cfg['REGRESSOR']['size'], 'Recortes automáticos incompatibles.')

    def predict_rows(data):
        output = []
        for r in data:
            value = None
            if r['crop'] is not None:
                im = tf.io.decode_png(tf.io.read_file(str(root / r['crop'])), channels=3)
                im = tf.ensure_shape(im, [cfg['REGRESSOR']['size'], cfg['REGRESSOR']['size'], 3])
                value = float(model(tf.cast(im[None], tf.float32), training=False).numpy()[0, 0])
                require(math.isfinite(value), 'Predicción BCS no finita.')
            output.append({'key': r['key'], 'cow_id': r['cow_id'], 'true_bcs': r['true_bcs'],
                           'pred_bcs': value, 'status': r['status']})
        return output

    predicted = predict_rows(samples['rows'])
    tag = 'inference_external' if external else 'end_to_end_'+split
    report(cfg, predicted, tag)
    if external:
        save(root / 'reports/inference_external/prediction.json', predicted)
        return
    gt = [{'key': r['key'], 'cow_id': r['cow_id'], 'true_bcs': r['bcs'], 'crop': r['crop'], 'status': 'ok'}
          for r in load(root / 'manifests/regression_samples.json') if r['split'] == split]
    report(cfg, predict_rows(gt), 'regressor_gt_'+split)
    # Baseline robusta calculada SOLO con entrenamiento, nunca con validación/test.
    import statistics
    baseline = statistics.median(r['bcs'] for r in load(root / 'manifests/regression_samples.json') if r['split'] == 'train')
    report(cfg, [{'key': r['key'], 'cow_id': r['cow_id'], 'true_bcs': r['true_bcs'],
                  'pred_bcs': baseline, 'status': 'train_median_baseline'} for r in gt], 'baseline_'+split)
    stage(cfg, 'evaluation_'+split,
          digest({**contract(cfg), 'model': sha(model_path), 'detector': settings['detector_hash'],
                  'inference': cfg['INFERENCE']}),
          [root / 'reports' / tag / 'metrics.json', root / 'reports' / ('regressor_gt_'+split) / 'metrics.json'])
