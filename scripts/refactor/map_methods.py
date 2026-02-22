from pathlib import Path
s=Path('ztb/training/callbacks/core/callback_implementations.py').read_text()
lines=s.splitlines()
class_stack=[]
methods=[]
for i,line in enumerate(lines, start=1):
    stripped=line.lstrip()
    indent=len(line)-len(stripped)
    if stripped.startswith('class '):
        class_name=stripped.split('(')[0].replace('class ','').strip()
        class_stack.append((class_name,indent))
    if stripped.startswith('def '):
        method_name=stripped.split('(')[0].replace('def ','').strip()
        # find current class from stack based on indent
        cur_class=None
        for name, cls_indent in reversed(class_stack):
            if indent > cls_indent:
                cur_class=name
                break
        methods.append((i,method_name,cur_class,indent))
# Filter on 'on_training_end'
for item in methods:
    if item[1]=='on_training_end':
        print(item)
