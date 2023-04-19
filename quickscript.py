import json

f = open('train_inspec.json', 'r')
samples = json.load(f)
f.close()

out = []

for sample in samples:
    sample["document"] = sample["document"].split(' ')
    out.append(sample)

f = open('train_inspec.json', 'w+')
json.dump(out, f)
f.close()
