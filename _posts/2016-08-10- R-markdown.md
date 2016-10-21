---
layout: post
title: R markdown in Sublime Text3
description: Support R markdown in Sublime Text3
modified: 2016-08-11
tags: [R markdown, Sublime Text, Latex]
---

### 1. Install Sublime Text plugin [randy3k/R-box](https://github.com/randy3k/R-Box)

### 2 Windows

#### 2.1. Create my own `sublime-build` for R markdown files

The default build system of R-box doesn't work, and get the error

    Error: '\G' is an unrecognized escape in character string starting "'C:\G"
    Execution halted
    [Finished in 0.4s with exit code 1]

since the windows path escape is not correctly handled.

This issue can be resolved by regular expression replacement [^1]

```json
{
    "selector": "text.html.markdown.rmarkdown",
    "working_dir": "${project_path:${folder}}",
    "cmd": [
        "Rscript", "-e",
        "library(rmarkdown); render('${file/\\\\/\\/\/g}')"
    ]
}
```

#### 2.2. Apply fix for pandoc 1.4 [^2]

To support latex in R markdown document, I added to the file ...\Anaconda3\R\library\rmarkdown\rmd\latex\default.tex:

```
    %fix for pandoc 1.14
    \providecommand{\tightlist}{
      \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
```

### 3. Mac
The default `R markdown` build system gets the error

    str expected, not list

it is because the PATH variable used by sublime text3 is `/usr/bin:/bin:/usr/sbin:/sbin` while my TexLive is installed at `/usr/local/texlive/2016/bin/x86_64-darwin`. This can be hacked by editing the PATH variable of sublime text3, described in the post [Hacking the PATH variable in Sublime Text](http://robdodson.me/hacking-the-path-variable-in-sublime-text/)

[^1]: <http://stackoverflow.com/questions/20752890/error-in-custom-sublime-build-for-knitr-markdown/>
[^2]: <https://github.com/tompollard/markdown-cv/issues/1/>