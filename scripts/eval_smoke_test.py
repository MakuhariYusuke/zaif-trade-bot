from ztb.evaluation.evaluator.evaluator import TradingEvaluator

if __name__ == '__main__':
    try:
        ev = TradingEvaluator(model_path='models/test_model.zip', data_path='data/btc_jpy_real_dataset.csv')
        res = ev.evaluate_model()
        print('Evaluator ran, returned keys:', sorted(list(res.keys())))
    except Exception as e:
        print('Evaluator smoke test failed:', e)
