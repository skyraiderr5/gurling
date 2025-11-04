#  Gurling

**Gurling** is an experimental programming language with **global variables** and **modular libraries**, written in Python.  
It’s designed for learning and experimenting with interpreters, language design, and modular systems.

>  *Work in Progress (WIP)* - Gurling is still under development!

---

## Overview

- **Author:** [Skyraiderr5](https://github.com/skyraiderr5)  
- **Year:** 2025  
- **Influences:** [Python](https://www.python.org/), [Flug](https://github.com/Truttle1/Flug)

---

## Language Description

Gurling is a **Python-inspired** interpreted language designed for experimentation and educational use.  
It uses a **modular library system**, where libraries are loaded via the `include` statement.

### Features
- Global variables (no local scope)
- Function definitions
- Expressions and conditionals (`if` / `else`)
- `while` loops (no `for` or `do-while`)
- Line comments using `#`
- Modular libraries (`include io;`, etc.)

Example:
```gurling
include io;
io.prn("Hello, World!");
```
