import json
d = json.load(open('notebooks/01_datasheet_curves_extraction.ipynb'))
errs = 0; figs = 0; streams = []
for i, c in enumerate(d['cells']):
    for o in c.get('outputs', []):
        if o.get('output_type') == 'error':
            errs += 1; print('ERROR cell', i, o['ename'], o['evalue'])
        elif o.get('output_type') == 'display_data' and 'image/png' in o.get('data', {}):
            figs += 1
        elif o.get('output_type') == 'stream' and o.get('name') == 'stdout':
            streams.append((i, ''.join(o['text'])))
print('cells:', len(d['cells']), '| errors:', errs, '| figures:', figs)
for i, t in streams:
    print(f'--- stdout cell {i} ---')
    print(t.strip()[:700])
