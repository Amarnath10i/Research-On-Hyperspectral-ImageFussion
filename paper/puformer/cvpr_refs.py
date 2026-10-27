"""Write cvpr_mainrefs.tex: the numbers of the labels in cvpr_main.aux, so that cvpr_supp.tex can refer to the
theorems, equations and tables of the main paper with \\mref{label} (run after compiling cvpr_main.tex with
--keep-intermediates)."""
import re

aux = open("cvpr_main.aux", encoding="utf-8", errors="replace").read()
out = []
for lab, num in re.findall(r"\\newlabel\{([^}]*)\}\{\{([^{}]*)\}", aux):
    if "@cref" not in lab:
        out.append("\\expandafter\\def\\csname mref@%s\\endcsname{%s}" % (lab, num))
with open("cvpr_mainrefs.tex", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print(len(out), "labels of cvpr_main.tex")
