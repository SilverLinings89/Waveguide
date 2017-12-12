from pathlib import Path
p = Path('.')
q = p/'Solutions'
if not q.is_dir():
	sys.exit(0)

i = 0
while (q/('run'+str(i))).is_dir():
	i = i+1
print(i)
