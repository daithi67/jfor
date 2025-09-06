# jfor

A tiny interpreter for a **Johnson-style FOR loop DSL**, written in Python. This is just a hobby project I created, after reading about Steve Johnson’s flexible `for` loop ideas from Bell Labs, built as a toy language.

## Features

- Counter loops (inclusive end, positive or negative step):

```jfor
for i = 1 to 10 by 2 do
    print i
end
```

- Iterator loops:

```jfor
for fruit in ["apple", "banana", "cherry"] do
    print fruit
end
```

- Assignments and expressions:

```jfor
total = 0
for n = 5 to 1 by -2 do
    total = total + n
end
print total
```

- If/else blocks:

```jfor
x = 7
if x % 2 == 0 then
    print "even"
else
    print "odd"
end
```

Expressions support numbers, strings, lists, arithmetic, comparisons, boolean logic, and power (`**`).

## Usage

Run the demo program:

```bash
python jfor.py demo
```

Run your own script:

```bash
python jfor.py myprog.jfor
```

_Not production-grade — just a fun learning project in parsing and interpreters._

## License

MIT License. See [LICENSE](LICENSE).
