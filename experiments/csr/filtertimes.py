import json
import sys

csvsep = " "

raw_data = {}
with open(sys.argv[1]) as f:
  raw_data = json.load(f)

formatted_data = dict()
def safeAppend(outerkey, innerkey, val):
  if(not outerkey in formatted_data):
    formatted_data[outerkey] = dict()
  formatted_data[outerkey][innerkey] = val

matrixname = sys.argv[2]

aggregates = True
if(len(sys.argv) > 3):
  aggregates = False

threadnums = set()
for benchmark in raw_data["benchmarks"]:
  if(benchmark["run_type"] == "aggregate" or not aggregates):
    if(not aggregates or benchmark["aggregate_name"] in ("mean","stddev")):
      settings = benchmark["name"].split("/")
      if(matrixname == settings[1]):
        threadnums.add(int(settings[2]))
        safeAppend("%s_%s_%s"%(settings[0],settings[1],settings[3]),int(settings[2]),benchmark["cpu_time"])

settings = list(formatted_data.keys())
print("nth"+csvsep+csvsep.join(settings))
for nth in sorted(threadnums):
  line = list()
  line.append(str(nth))
  for setting in settings:
    if(nth in formatted_data[setting]):
      line.append(str(formatted_data[setting][nth]))
    elif("BM_spmvt_serial" in setting):
      line.append(str(formatted_data[setting][1]))
    else:
      line.append("nan")
  print(csvsep.join(line))

