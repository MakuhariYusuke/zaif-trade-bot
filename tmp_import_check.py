import importlib
m = importlib.import_module('ztb.analysis.evaluation.walk_forward_integration_pipeline')
print('module file:', getattr(m,'__file__',None))
print('HasBacktestReporterAttr:', hasattr(m, 'BacktestReporter'))
print('HasBacktestReporter name in module dict:', 'BacktestReporter' in m.__dict__)
print('Top lines of file:')
with open(m.__file__, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 80:
            print(i+1, line.rstrip())
        else:
            break
