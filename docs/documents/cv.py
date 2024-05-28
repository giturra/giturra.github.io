
cv_file="../cv.md"
cv_md = open(cv_file, "r", encoding='utf-8')

for cv_line in cv_md.readlines():
    if(cv_line.find("[Full list here]")==-1): 
        print(cv_line)
       
